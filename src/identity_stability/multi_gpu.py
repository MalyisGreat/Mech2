from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import RunConfig


DONE_PREFIX = "[done] run directory:"


def _to_jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _split_round_robin(items: list[str], buckets: int) -> list[list[str]]:
    out = [[] for _ in range(max(1, buckets))]
    for i, item in enumerate(items):
        out[i % len(out)].append(item)
    return out


def _parse_run_dir(stdout_text: str) -> Path | None:
    for line in stdout_text.splitlines():
        s = line.strip()
        if s.startswith(DONE_PREFIX):
            return Path(s[len(DONE_PREFIX) :].strip())
    return None


def _fallback_latest_run(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    candidates = [p for p in output_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _merge_model_dirs(worker_run_dir: Path, merged_run_dir: Path) -> None:
    for child in worker_run_dir.iterdir():
        if not child.is_dir():
            continue
        dest = merged_run_dir / child.name
        if dest.exists():
            continue
        shutil.copytree(child, dest)


def _write_merged_summaries(df: pd.DataFrame, merged_run_dir: Path) -> None:
    summary = (
        df.groupby(["concept_name", "model_id", "vector_method", "alpha"], as_index=False)
        .agg(
            peak_drift_mean=("peak_drift", "mean"),
            peak_drift_relative_mean=("peak_drift_relative", "mean"),
            end_drift_mean=("end_drift", "mean"),
            end_drift_relative_mean=("end_drift_relative", "mean"),
            drift_auc_mean=("drift_auc", "mean"),
            drift_auc_relative_mean=("drift_auc_relative", "mean"),
            recovery_fraction_mean=("recovery_fraction", "mean"),
            recovery_slope_mean=("recovery_slope", "mean"),
            cad_mean=("cad", "mean"),
            persistence_mean=("persistence", "mean"),
            degradation_mean=("degradation", "mean"),
            overshoot_index_mean=("overshoot_index", "mean"),
            crossed_baseline_rate=("crossed_baseline", "mean"),
            next_token_kl_mean=("next_token_kl", "mean"),
            n=("prompt_index", "count"),
        )
        .sort_values(["concept_name", "model_id", "vector_method", "alpha"])
    )
    summary.to_csv(merged_run_dir / "metrics_summary.csv", index=False)

    layer_summary = (
        df.groupby(["concept_name", "model_id", "layer_index", "vector_method"], as_index=False)
        .agg(
            peak_drift_mean=("peak_drift", "mean"),
            peak_drift_relative_mean=("peak_drift_relative", "mean"),
            drift_auc_mean=("drift_auc", "mean"),
            drift_auc_relative_mean=("drift_auc_relative", "mean"),
            recovery_fraction_mean=("recovery_fraction", "mean"),
            recovery_slope_mean=("recovery_slope", "mean"),
            cad_mean=("cad", "mean"),
            persistence_mean=("persistence", "mean"),
            degradation_mean=("degradation", "mean"),
            crossed_baseline_rate=("crossed_baseline", "mean"),
            n=("prompt_index", "count"),
        )
        .sort_values(["concept_name", "model_id", "layer_index", "vector_method"])
    )
    layer_summary.to_csv(merged_run_dir / "layer_summary.csv", index=False)

    with (merged_run_dir / "quick_report.md").open("w", encoding="utf-8") as f:
        f.write("# Quick Report\n\n")
        f.write(f"- Rows: `{len(df)}`\n")
        f.write(f"- Models with outputs: `{df['model_id'].nunique()}`\n\n")
        f.write("## Mean Recovery by Model\n\n")
        by_model = (
            df.groupby("model_id", as_index=False)
            .agg(
                recovery_fraction_mean=("recovery_fraction", "mean"),
                peak_drift_mean=("peak_drift", "mean"),
                peak_drift_relative_mean=("peak_drift_relative", "mean"),
                cad_mean=("cad", "mean"),
                persistence_mean=("persistence", "mean"),
                overshoot_index_mean=("overshoot_index", "mean"),
                crossed_baseline_rate=("crossed_baseline", "mean"),
            )
            .sort_values("model_id")
        )
        for _, row in by_model.iterrows():
            f.write(
                "- "
                f"{row['model_id']}: "
                f"recovery={row['recovery_fraction_mean']:.4f}, "
                f"peak={row['peak_drift_mean']:.4f}, "
                f"peak_rel={row['peak_drift_relative_mean']:.6f}, "
                f"cad={row['cad_mean']:.4f}, "
                f"persist={row['persistence_mean']:.4f}, "
                f"overshoot={row['overshoot_index_mean']:.4f}, "
                f"crossed={row['crossed_baseline_rate']:.4f}\n"
            )


def run_experiment_multi_gpu(
    config: RunConfig,
    gpu_ids: list[int],
    source_config_path: Path | None = None,
    run_label: str = "multi_gpu_experiment",
) -> Path:
    if len(gpu_ids) < 2:
        raise ValueError("run_experiment_multi_gpu requires at least 2 gpu ids.")

    repo_root = Path(__file__).resolve().parents[2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orchestrator_dir = config.output_root / f"{run_label}_{stamp}"
    workers_root = orchestrator_dir / "worker_runs"
    logs_root = orchestrator_dir / "logs"
    merged_run_dir = orchestrator_dir / "merged_run"
    workers_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    merged_run_dir.mkdir(parents=True, exist_ok=True)

    model_shards = _split_round_robin(list(config.model_ids), len(gpu_ids))
    worker_specs = [(gpu, shard) for gpu, shard in zip(gpu_ids, model_shards) if shard]
    worker_rows: list[dict[str, Any]] = []
    launched_workers: list[dict[str, Any]] = []

    overall_started = time.time()
    for worker_idx, (gpu_id, shard_models) in enumerate(worker_specs):
        worker_root = workers_root / f"gpu{gpu_id}"
        worker_cfg_path = worker_root / "worker_config.yaml"
        worker_output_root = worker_root / "runs"
        worker_output_root.mkdir(parents=True, exist_ok=True)

        shard_cfg = asdict(config)
        shard_cfg["model_ids"] = shard_models
        shard_cfg["output_root"] = str(worker_output_root)
        with worker_cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(_to_jsonable(shard_cfg), f, sort_keys=False)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_experiment.py"),
            "--config",
            str(worker_cfg_path),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        launched_workers.append(
            {
                "worker_index": worker_idx,
                "gpu_id": gpu_id,
                "models": shard_models,
                "worker_output_root": worker_output_root,
                "process": proc,
                "started_at": time.time(),
            }
        )

    for launched in launched_workers:
        proc = launched["process"]
        stdout, stderr = proc.communicate()
        gpu_id = launched["gpu_id"]
        worker_output_root = launched["worker_output_root"]

        (logs_root / f"gpu{gpu_id}_stdout.log").write_text(stdout, encoding="utf-8")
        (logs_root / f"gpu{gpu_id}_stderr.log").write_text(stderr, encoding="utf-8")

        run_dir = _parse_run_dir(stdout)
        if run_dir is None and proc.returncode == 0:
            run_dir = _fallback_latest_run(worker_output_root)

        worker_rows.append(
            {
                "worker_index": launched["worker_index"],
                "gpu_id": gpu_id,
                "models": launched["models"],
                "return_code": int(proc.returncode),
                "run_dir": str(run_dir) if run_dir is not None else "",
                "elapsed_seconds": float(time.time() - launched["started_at"]),
            }
        )

    with (orchestrator_dir / "worker_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(worker_rows, f, indent=2)
    pd.DataFrame(worker_rows).to_csv(orchestrator_dir / "worker_manifest.csv", index=False)

    any_fail = any(row["return_code"] != 0 or not row["run_dir"] for row in worker_rows)
    if any_fail:
        raise RuntimeError(
            "At least one multi-GPU worker failed. Check logs under "
            f"{logs_root} and manifest at {orchestrator_dir / 'worker_manifest.json'}."
        )

    merged_failures: list[dict[str, Any]] = []
    metrics_frames: list[pd.DataFrame] = []
    vector_frames: list[pd.DataFrame] = []
    model_frames: list[pd.DataFrame] = []
    topk_lines: list[str] = []
    worker_prov_dir = merged_run_dir / "worker_provenance"
    worker_prov_dir.mkdir(parents=True, exist_ok=True)

    worker_run_dirs = [Path(row["run_dir"]) for row in worker_rows]
    for i, worker_run_dir in enumerate(worker_run_dirs):
        _merge_model_dirs(worker_run_dir=worker_run_dir, merged_run_dir=merged_run_dir)

        metrics_path = worker_run_dir / "metrics_full.csv"
        if metrics_path.exists():
            metrics_frames.append(pd.read_csv(metrics_path))

        vec_path = worker_run_dir / "vector_registry.csv"
        if vec_path.exists():
            vector_frames.append(pd.read_csv(vec_path))

        model_path = worker_run_dir / "model_registry.csv"
        if model_path.exists():
            model_frames.append(pd.read_csv(model_path))

        topk_path = worker_run_dir / "layer_topk_records.jsonl"
        if topk_path.exists():
            topk_lines.extend(topk_path.read_text(encoding="utf-8").splitlines())

        fail_path = worker_run_dir / "failures.json"
        if fail_path.exists():
            try:
                merged_failures.extend(json.loads(fail_path.read_text(encoding="utf-8")))
            except Exception:
                pass

        prov_path = worker_run_dir / "run_provenance.json"
        if prov_path.exists():
            shutil.copy2(prov_path, worker_prov_dir / f"worker_{i:02d}_run_provenance.json")

    merged_cfg = asdict(config)
    merged_cfg["_multi_gpu"] = {
        "gpu_ids": gpu_ids,
        "source_config_path": str(source_config_path) if source_config_path else None,
        "worker_runs": [str(p) for p in worker_run_dirs],
    }
    with (merged_run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(merged_cfg), f, indent=2)

    if worker_run_dirs:
        sample_prompt = worker_run_dirs[0] / "prompt_set.json"
        if sample_prompt.exists():
            shutil.copy2(sample_prompt, merged_run_dir / "prompt_set.json")

    if metrics_frames:
        merged_df = pd.concat(metrics_frames, ignore_index=True)
        merged_df.to_csv(merged_run_dir / "metrics_full.csv", index=False)
        _write_merged_summaries(merged_df, merged_run_dir=merged_run_dir)

    if vector_frames:
        pd.concat(vector_frames, ignore_index=True).to_csv(merged_run_dir / "vector_registry.csv", index=False)
    if model_frames:
        pd.concat(model_frames, ignore_index=True).to_csv(merged_run_dir / "model_registry.csv", index=False)
    if topk_lines:
        with (merged_run_dir / "layer_topk_records.jsonl").open("w", encoding="utf-8") as f:
            for line in topk_lines:
                f.write(line + "\n")

    with (merged_run_dir / "failures.json").open("w", encoding="utf-8") as f:
        json.dump(merged_failures, f, indent=2)

    run_summary = {
        "mode": "multi_gpu",
        "gpu_ids": gpu_ids,
        "workers": len(worker_rows),
        "models_attempted": len(config.model_ids),
        "worker_manifest_json": str(orchestrator_dir / "worker_manifest.json"),
        "elapsed_seconds": float(time.time() - overall_started),
        "worker_run_dirs": [str(p) for p in worker_run_dirs],
    }
    with (merged_run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    return merged_run_dir
