from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


GPU_TELEMETRY_FIELDS = [
    "timestamp_utc",
    "gpu_index",
    "gpu_name",
    "utilization_gpu_pct",
    "utilization_mem_pct",
    "memory_used_mb",
    "memory_total_mb",
    "power_draw_w",
    "power_limit_w",
    "temperature_c",
]


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NONE"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _series_stats(series: pd.Series) -> dict[str, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "avg": None,
            "p95": None,
            "max": None,
        }
    return {
        "avg": float(s.mean()),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


class GpuTelemetrySampler:
    def __init__(self, csv_path: Path, interval_sec: float) -> None:
        self.csv_path = csv_path
        self.interval_sec = max(0.2, float(interval_sec))
        self.errors: list[str] = []
        self.started_at_utc: str | None = None
        self.stopped_at_utc: str | None = None
        self.sample_rows: int = 0
        self.enabled = False

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def is_supported() -> bool:
        return shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self.is_supported():
            self.errors.append("nvidia-smi not found on PATH; GPU telemetry disabled.")
            return

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GPU_TELEMETRY_FIELDS)
            writer.writeheader()

        self.started_at_utc = _utc_now_iso()
        self.enabled = True
        self._thread = threading.Thread(target=self._run_loop, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self, join_timeout_sec: float = 15.0) -> None:
        if self._thread is None:
            self.stopped_at_utc = _utc_now_iso()
            return
        self._stop_event.set()
        self._thread.join(timeout=max(0.1, float(join_timeout_sec)))
        self.stopped_at_utc = _utc_now_iso()

    def _run_loop(self) -> None:
        query = ",".join(
            [
                "index",
                "name",
                "utilization.gpu",
                "utilization.memory",
                "memory.used",
                "memory.total",
                "power.draw",
                "power.limit",
                "temperature.gpu",
            ]
        )
        cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]

        while not self._stop_event.is_set():
            started = time.time()
            rows: list[dict[str, Any]] = []
            stamp = _utc_now_iso()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=20,
                )
                for line in proc.stdout.splitlines():
                    text = line.strip()
                    if not text:
                        continue
                    parts = [p.strip() for p in text.split(",")]
                    if len(parts) < 9:
                        continue
                    rows.append(
                        {
                            "timestamp_utc": stamp,
                            "gpu_index": parts[0],
                            "gpu_name": parts[1],
                            "utilization_gpu_pct": parts[2],
                            "utilization_mem_pct": parts[3],
                            "memory_used_mb": parts[4],
                            "memory_total_mb": parts[5],
                            "power_draw_w": parts[6],
                            "power_limit_w": parts[7],
                            "temperature_c": parts[8],
                        }
                    )
            except Exception as exc:
                self.errors.append(f"{stamp}: {exc}")

            if rows:
                with self.csv_path.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=GPU_TELEMETRY_FIELDS)
                    writer.writerows(rows)
                self.sample_rows += len(rows)

            wait_sec = self.interval_sec - (time.time() - started)
            if wait_sec > 0:
                self._stop_event.wait(wait_sec)


def summarize_gpu_telemetry_csv(
    csv_path: Path,
    interval_sec: float | None,
    sampler_errors: list[str] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "csv_path": str(csv_path),
        "interval_sec": float(interval_sec) if interval_sec is not None else None,
        "sample_rows": 0,
        "gpu_count": 0,
        "overall": {},
        "per_gpu": [],
        "errors": list(sampler_errors) if sampler_errors else [],
    }
    if not csv_path.exists():
        summary["errors"].append("telemetry csv missing; no samples recorded.")
        return summary

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        summary["errors"].append(f"failed to parse telemetry csv: {exc}")
        return summary

    if df.empty:
        summary["errors"].append("telemetry csv is empty.")
        return summary

    summary["sample_rows"] = int(len(df))

    numeric_cols = [
        "utilization_gpu_pct",
        "utilization_mem_pct",
        "memory_used_mb",
        "memory_total_mb",
        "power_draw_w",
        "power_limit_w",
        "temperature_c",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "memory_total_mb" in df.columns and "memory_used_mb" in df.columns:
        total = pd.to_numeric(df["memory_total_mb"], errors="coerce")
        used = pd.to_numeric(df["memory_used_mb"], errors="coerce")
        mem_util = (used / total.replace(0, pd.NA)) * 100.0
        mem_util = mem_util.replace([float("inf"), float("-inf")], pd.NA)
        df["memory_utilization_pct"] = mem_util

    overall = {
        "gpu_utilization_pct": _series_stats(df.get("utilization_gpu_pct", pd.Series(dtype=float))),
        "memory_utilization_pct": _series_stats(df.get("memory_utilization_pct", pd.Series(dtype=float))),
        "power_draw_w": _series_stats(df.get("power_draw_w", pd.Series(dtype=float))),
        "temperature_c": _series_stats(df.get("temperature_c", pd.Series(dtype=float))),
        "memory_used_mb": _series_stats(df.get("memory_used_mb", pd.Series(dtype=float))),
    }
    summary["overall"] = overall

    per_gpu: list[dict[str, Any]] = []
    if "gpu_index" in df.columns:
        for gpu_index, group in df.groupby("gpu_index", dropna=False):
            gpu_name = None
            if "gpu_name" in group.columns:
                names = group["gpu_name"].dropna()
                if not names.empty:
                    gpu_name = str(names.iloc[0])
            parsed_idx = _coerce_float(gpu_index)
            per_gpu.append(
                {
                    "gpu_index": int(parsed_idx) if parsed_idx is not None else str(gpu_index),
                    "gpu_name": gpu_name,
                    "sample_rows": int(len(group)),
                    "gpu_utilization_pct": _series_stats(
                        group.get("utilization_gpu_pct", pd.Series(dtype=float))
                    ),
                    "memory_utilization_pct": _series_stats(
                        group.get("memory_utilization_pct", pd.Series(dtype=float))
                    ),
                    "power_draw_w": _series_stats(group.get("power_draw_w", pd.Series(dtype=float))),
                    "temperature_c": _series_stats(group.get("temperature_c", pd.Series(dtype=float))),
                    "memory_used_mb": _series_stats(group.get("memory_used_mb", pd.Series(dtype=float))),
                }
            )
    summary["per_gpu"] = per_gpu
    summary["gpu_count"] = int(len(per_gpu))
    return summary
