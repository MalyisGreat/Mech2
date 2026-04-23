from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from identity_battery_common import add_src_to_path, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the redesigned identity master suite.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _resolve_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def _validate_self_other(step_dir: Path) -> tuple[bool, dict[str, object]]:
    summary = _safe_read_csv(step_dir / "summary_by_model_frame.csv")
    control = summary[summary["item_type"] == "control"].copy()
    descriptive = summary[summary["item_type"] == "descriptive"].copy()
    metrics = {
        "control_row_count": int(len(control)),
        "descriptive_row_count": int(len(descriptive)),
        "max_control_accuracy": float(control["control_accuracy_mean_mean"].fillna(0.0).max()) if len(control) else 0.0,
        "max_control_valid_orientation": float(control["valid_orientation_rate_mean"].fillna(0.0).max()) if len(control) else 0.0,
        "max_descriptive_non_tie": float(descriptive["non_tie_rate_mean"].fillna(0.0).max()) if len(descriptive) else 0.0,
        "max_descriptive_structure": float(descriptive["structure_score_mean"].fillna(0.0).max()) if len(descriptive) else 0.0,
        "max_descriptive_contradiction": float(descriptive["contradiction_rate_mean"].fillna(1.0).min()) if len(descriptive) else 1.0,
    }
    passed = (
        metrics["control_row_count"] > 0
        and metrics["descriptive_row_count"] > 0
        and metrics["max_control_accuracy"] >= 0.5
        and metrics["max_control_valid_orientation"] >= 0.9
        and metrics["max_descriptive_contradiction"] < 0.5
        and (metrics["max_descriptive_non_tie"] > 0.0 or metrics["max_descriptive_structure"] > 0.0)
    )
    return passed, metrics


def _validate_self_prediction(step_dir: Path) -> tuple[bool, dict[str, object]]:
    summary = _safe_read_csv(step_dir / "summary_by_model_frame.csv")
    metrics = {
        "row_count": int(len(summary)),
        "max_valid_choice_rate": float(summary["valid_choice_rate_mean"].fillna(0.0).max()) if len(summary) else 0.0,
        "max_predicted_gap_rate": float(summary["predicted_gap_rate_mean"].fillna(0.0).max()) if len(summary) else 0.0,
        "max_actual_gap_rate": float(summary["actual_gap_rate_mean"].fillna(0.0).max()) if len(summary) else 0.0,
        "max_gap_direction_accuracy": float(summary["gap_direction_accuracy_mean_mean"].fillna(0.0).max()) if len(summary) else 0.0,
    }
    passed = (
        metrics["row_count"] > 0
        and metrics["max_valid_choice_rate"] >= 0.95
        and metrics["max_actual_gap_rate"] >= 0.05
        and metrics["max_gap_direction_accuracy"] >= 0.4
    )
    return passed, metrics


def _validate_commitment(step_dir: Path) -> tuple[bool, dict[str, object]]:
    summary = _safe_read_csv(step_dir / "summary_by_model_frame.csv")
    label_bias = _safe_read_csv(step_dir / "label_bias_summary.csv")
    label_gap = (
        label_bias["label_a_rate_mean"].fillna(0.0) - label_bias["target_on_label_a_rate_mean"].fillna(0.0)
        if len(label_bias)
        else pd.Series(dtype=float)
    )
    metrics = {
        "row_count": int(len(summary)),
        "label_bias_row_count": int(len(label_bias)),
        "max_valid_choice_rate": float(summary["valid_choice_rate_mean"].fillna(0.0).max()) if len(summary) else 0.0,
        "max_reveal_pair_accuracy": float(summary["reveal_pair_accuracy_mean"].fillna(0.0).max()) if len(summary) else 0.0,
        "max_abs_label_bias_gap": float(label_gap.abs().max()) if len(label_gap) else float("inf"),
    }
    passed = (
        metrics["row_count"] > 0
        and metrics["label_bias_row_count"] > 0
        and metrics["max_valid_choice_rate"] >= 0.95
        and metrics["max_reveal_pair_accuracy"] >= 0.5
        and metrics["max_abs_label_bias_gap"] <= 0.35
    )
    return passed, metrics


def _validate_self_recognition(step_dir: Path) -> tuple[bool, dict[str, object]]:
    summary = _safe_read_csv(step_dir / "summary_by_model.csv")
    quality = _safe_read_csv(step_dir / "quality_summary.csv")
    choose_values = summary["choose_self_baseline_mean"].dropna().astype(float).tolist() if "choose_self_baseline_mean" in summary.columns else []
    metrics = {
        "summary_row_count": int(len(summary)),
        "quality_row_count": int(len(quality)),
        "non_nan_choose_rows": int(len(choose_values)),
        "max_pair_valid_rate": float(quality["pair_valid_rate"].fillna(0.0).max()) if len(quality) else 0.0,
    }
    passed = (
        metrics["summary_row_count"] > 0
        and metrics["quality_row_count"] > 0
        and metrics["non_nan_choose_rows"] > 0
        and metrics["max_pair_valid_rate"] >= 0.5
    )
    return passed, metrics


def _validate_step(step_name: str, expected_output: Path) -> tuple[bool, dict[str, object]]:
    step_dir = expected_output.parent
    if step_name == "self_other_boundary_transfer_v5":
        return _validate_self_other(step_dir)
    if step_name == "self_prediction_transfer_v3":
        return _validate_self_prediction(step_dir)
    if step_name == "commitment_persistence_v2":
        return _validate_commitment(step_dir)
    if step_name == "self_recognition_nearfoil_v2":
        return _validate_self_recognition(step_dir)
    if step_name == "report_identity_master_suite":
        return True, {"validation": "report_exists"}
    return True, {"validation": "not_configured"}


def main() -> None:
    repo_root = add_src_to_path()
    args = parse_args()
    config = load_yaml_config(args.config)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "run.log"
    status_path = output_root / "status.json"
    manifest_path = output_root / "run_manifest.md"
    python_exe = str(config.get("python_exe", sys.executable))
    steps = [dict(step) for step in config["steps"]]

    status: dict[str, object] = {
        "config": str(args.config),
        "started_at": datetime.now().isoformat(),
        "python_exe": python_exe,
        "steps": [],
    }

    with manifest_path.open("w", encoding="utf-8") as manifest:
        manifest.write("# Identity Master Suite\n\n")
        manifest.write(f"- Config: `{args.config}`\n")
        manifest.write(f"- Started: `{status['started_at']}`\n")
        manifest.write(f"- Python: `{python_exe}`\n\n")
        manifest.write("## Steps\n\n")
        for step in steps:
            command = [
                python_exe,
                str(_resolve_path(repo_root, str(step["script"]))),
                "--config",
                str(_resolve_path(repo_root, str(step["config"]))),
            ]
            manifest.write(f"- `{step['name']}`: `{' '.join(command)}`\n")

    with log_path.open("a", encoding="utf-8") as log_file:
        for step in steps:
            script_path = _resolve_path(repo_root, str(step["script"]))
            step_config_path = _resolve_path(repo_root, str(step["config"]))
            expected_output = _resolve_path(repo_root, str(step["expected_output"]))
            command = [python_exe, str(script_path), "--config", str(step_config_path)]
            step_started_at = datetime.now().isoformat()
            step_status = {
                "name": str(step["name"]),
                "script": str(script_path),
                "config": str(step_config_path),
                "expected_output": str(expected_output),
                "command": command,
                "started_at": step_started_at,
                "status": "running",
            }

            reused_existing_output = False
            if expected_output.exists():
                validation_ok, validation_metrics = _validate_step(str(step["name"]), expected_output)
                if validation_ok:
                    reused_existing_output = True
                    step_status["finished_at"] = step_started_at
                    step_status["return_code"] = 0
                    step_status["output_exists"] = True
                    step_status["validation_ok"] = True
                    step_status["validation_metrics"] = validation_metrics
                    step_status["status"] = "completed"
                    step_status["reused_existing_output"] = True
                    status["steps"].append(step_status)
                    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                    log_file.write(
                        f"\n[{step_started_at}] SKIP {step['name']} using validated existing output: {expected_output}\n"
                    )
                    log_file.flush()

            if reused_existing_output:
                continue

            status["steps"].append(step_status)
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            log_file.write(f"\n[{step_status['started_at']}] START {step['name']}: {' '.join(command)}\n")
            log_file.flush()

            process = subprocess.Popen(
                command,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
            return_code = int(process.wait())

            step_status["finished_at"] = datetime.now().isoformat()
            step_status["return_code"] = return_code
            step_status["output_exists"] = bool(expected_output.exists())
            if return_code == 0 and expected_output.exists():
                validation_ok, validation_metrics = _validate_step(str(step["name"]), expected_output)
                step_status["validation_ok"] = bool(validation_ok)
                step_status["validation_metrics"] = validation_metrics
            else:
                step_status["validation_ok"] = False
                step_status["validation_metrics"] = {}
            if return_code == 0 and expected_output.exists() and step_status["validation_ok"]:
                step_status["status"] = "completed"
                log_file.write(f"[{step_status['finished_at']}] END {step['name']} OK\n")
            else:
                step_status["status"] = "failed"
                log_file.write(
                    f"[{step_status['finished_at']}] END {step['name']} FAILED rc={return_code} "
                    f"validation_ok={step_status['validation_ok']} metrics={json.dumps(step_status['validation_metrics'])}\n"
                )
                status["finished_at"] = step_status["finished_at"]
                status["status"] = "failed"
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                raise SystemExit(return_code or 1)
            log_file.flush()
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    status["finished_at"] = datetime.now().isoformat()
    status["status"] = "completed"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
