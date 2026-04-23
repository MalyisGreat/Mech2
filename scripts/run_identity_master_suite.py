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


def _validate_commitment_runtime(step_dir: Path) -> tuple[bool, dict[str, object]]:
    summary = _safe_read_csv(step_dir / "summary_by_model_frame.csv")
    label_bias = _safe_read_csv(step_dir / "label_bias_summary.csv")
    label_gap = (
        label_bias["label_a_rate_mean"].fillna(0.0) - label_bias["target_on_label_a_rate_mean"].fillna(0.0)
        if len(label_bias)
        else pd.Series(dtype=float)
    )
    reveal_values = (
        summary["reveal_pair_accuracy_mean"].dropna().astype(float).tolist()
        if "reveal_pair_accuracy_mean" in summary.columns
        else []
    )
    metrics = {
        "row_count": int(len(summary)),
        "label_bias_row_count": int(len(label_bias)),
        "max_valid_choice_rate": float(summary["valid_choice_rate_mean"].fillna(0.0).max()) if len(summary) else 0.0,
        "reveal_pair_accuracy_row_count": int(len(reveal_values)),
        "max_reveal_pair_accuracy": max(reveal_values) if reveal_values else 0.0,
        "max_abs_label_bias_gap": float(label_gap.abs().max()) if len(label_gap) else float("inf"),
    }
    passed = (
        metrics["row_count"] > 0
        and metrics["label_bias_row_count"] > 0
        and metrics["max_valid_choice_rate"] >= 0.95
        and metrics["max_abs_label_bias_gap"] <= 0.6
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


def _validate_diachronic_graft(step_dir: Path) -> tuple[bool, dict[str, object]]:
    summary = _safe_read_csv(step_dir / "summary_by_condition.csv")
    results = _safe_read_csv(step_dir / "results.csv")
    expected_controls = {"primary", "adjacent", "very_early", "random_same_norm", "shuffled_prompt", "name_only"}
    observed_controls = set(results["control_kind"].dropna().astype(str).tolist()) if "control_kind" in results.columns else set()
    observed_modes = set(results["graft_mode"].dropna().astype(str).tolist()) if "graft_mode" in results.columns else set()
    observed_tokens = set(results["token_position_label"].dropna().astype(str).tolist()) if "token_position_label" in results.columns else set()
    metrics = {
        "summary_row_count": int(len(summary)),
        "result_row_count": int(len(results)),
        "observed_control_count": int(len(observed_controls)),
        "missing_controls": sorted(expected_controls - observed_controls),
        "observed_modes": sorted(observed_modes),
        "observed_token_positions": sorted(observed_tokens),
        "max_donor_identity_fraction": float(summary["donor_identity_fraction_mean"].fillna(0.0).max()) if len(summary) else 0.0,
        "max_next_token_kl": float(summary["next_token_kl_mean"].fillna(0.0).max()) if len(summary) else 0.0,
    }
    passed = (
        metrics["summary_row_count"] > 0
        and metrics["result_row_count"] > 0
        and not metrics["missing_controls"]
        and {"single_layer", "prefix", "suffix"}.issubset(observed_modes)
        and {"last_prompt_token", "first_prompt_token"}.issubset(observed_tokens)
    )
    return passed, metrics


def _validate_temporal_authorship_matrix(step_dir: Path) -> tuple[bool, dict[str, object]]:
    results = _safe_read_csv(step_dir / "results.csv")
    pair_summary = _safe_read_csv(step_dir / "summary_by_pair.csv")
    self_pref = _safe_read_csv(step_dir / "self_preference_summary.csv")
    matrix = _safe_read_csv(step_dir / "authorship_preference_matrix.csv")
    expected_revisions: set[str] = set()
    manifest_path = step_dir / "run_manifest.md"
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("- Revisions: `") and line.endswith("`"):
                payload = line[len("- Revisions: `") : -1]
                expected_revisions = {part.strip() for part in payload.split(",") if part.strip()}
                break
    if not expected_revisions:
        expected_revisions = {"step4000", "step16000", "step64000", "step128000", "step143000"}
    observed_evaluators = set(results["evaluator_revision"].dropna().astype(str).tolist()) if "evaluator_revision" in results.columns else set()
    observed_sources = set(results["source_revision"].dropna().astype(str).tolist()) if "source_revision" in results.columns else set()
    metrics = {
        "result_row_count": int(len(results)),
        "pair_summary_row_count": int(len(pair_summary)),
        "self_preference_row_count": int(len(self_pref)),
        "matrix_row_count": int(len(matrix)),
        "observed_evaluator_count": int(len(observed_evaluators)),
        "observed_source_count": int(len(observed_sources)),
        "missing_evaluators": sorted(expected_revisions - observed_evaluators),
        "missing_sources": sorted(expected_revisions - observed_sources),
        "max_self_preference_rate": float(self_pref["self_preference_rate"].fillna(0.0).max()) if len(self_pref) else 0.0,
        "mean_diagonal_margin_logprob": float(self_pref["mean_diagonal_margin_logprob"].dropna().astype(float).mean())
        if ("mean_diagonal_margin_logprob" in self_pref.columns and self_pref["mean_diagonal_margin_logprob"].notna().any())
        else float("nan"),
    }
    passed = (
        metrics["result_row_count"] > 0
        and metrics["pair_summary_row_count"] > 0
        and metrics["self_preference_row_count"] > 0
        and metrics["matrix_row_count"] > 0
        and not metrics["missing_evaluators"]
        and not metrics["missing_sources"]
    )
    return passed, metrics


def _validate_checkpoint_age_recognition(step_dir: Path) -> tuple[bool, dict[str, object]]:
    results = _safe_read_csv(step_dir / "results.csv")
    pair_summary = _safe_read_csv(step_dir / "summary_by_pair.csv")
    comparison_summary = _safe_read_csv(step_dir / "summary_by_comparison.csv")
    generation_quality = _safe_read_csv(step_dir / "generation_quality_summary.csv")
    observed_comparisons = set(results["comparison_revision"].dropna().astype(str).tolist()) if "comparison_revision" in results.columns else set()
    observed_evaluators = set(results["evaluator_revision"].dropna().astype(str).tolist()) if "evaluator_revision" in results.columns else set()
    metrics = {
        "result_row_count": int(len(results)),
        "pair_summary_row_count": int(len(pair_summary)),
        "comparison_summary_row_count": int(len(comparison_summary)),
        "generation_quality_row_count": int(len(generation_quality)),
        "observed_comparison_count": int(len(observed_comparisons)),
        "observed_evaluator_count": int(len(observed_evaluators)),
        "max_choose_anchor_centered_rate": float(comparison_summary["choose_anchor_centered_mean"].fillna(0.0).max())
        if len(comparison_summary)
        else 0.0,
        "max_generation_valid_rate": float(generation_quality["generation_valid_rate"].fillna(0.0).max())
        if len(generation_quality)
        else 0.0,
    }
    passed = (
        metrics["result_row_count"] > 0
        and metrics["pair_summary_row_count"] > 0
        and metrics["comparison_summary_row_count"] > 0
        and metrics["generation_quality_row_count"] > 0
        and metrics["observed_comparison_count"] > 0
        and metrics["observed_evaluator_count"] > 0
    )
    return passed, metrics


def _validate_behavioral_fingerprint(step_dir: Path) -> tuple[bool, dict[str, object]]:
    results = _safe_read_csv(step_dir / "results.csv")
    summary = _safe_read_csv(step_dir / "summary_by_model_frame.csv")
    feature_summary = _safe_read_csv(step_dir / "summary_by_profile_source.csv")
    metrics = {
        "result_row_count": int(len(results)),
        "summary_row_count": int(len(summary)),
        "feature_summary_row_count": int(len(feature_summary)),
        "max_self_valid_choice_rate": float(summary["self_profile_valid_choice_rate_mean"].fillna(0.0).max())
        if "self_profile_valid_choice_rate_mean" in summary.columns and len(summary)
        else 0.0,
        "max_self_margin_vs_scrambled": float(summary["self_margin_vs_scrambled_mean"].fillna(0.0).max())
        if "self_margin_vs_scrambled_mean" in summary.columns and len(summary)
        else 0.0,
    }
    passed = (
        metrics["result_row_count"] > 0
        and metrics["summary_row_count"] > 0
        and metrics["feature_summary_row_count"] > 0
        and metrics["max_self_valid_choice_rate"] >= 0.9
    )
    return passed, metrics


def _validate_kinship_ladder(step_dir: Path) -> tuple[bool, dict[str, object]]:
    results = _safe_read_csv(step_dir / "results.csv")
    summary = _safe_read_csv(step_dir / "summary_by_foil.csv")
    pair_summary = _safe_read_csv(step_dir / "summary_by_pair.csv")
    metrics = {
        "result_row_count": int(len(results)),
        "summary_row_count": int(len(summary)),
        "pair_summary_row_count": int(len(pair_summary)),
        "max_pair_valid_rate": float(summary["pair_valid_rate_mean"].fillna(0.0).max())
        if "pair_valid_rate_mean" in summary.columns and len(summary)
        else 0.0,
        "observed_foil_count": int(summary["foil_kind"].nunique()) if "foil_kind" in summary.columns else 0,
    }
    passed = (
        metrics["result_row_count"] > 0
        and metrics["summary_row_count"] > 0
        and metrics["pair_summary_row_count"] > 0
        and metrics["max_pair_valid_rate"] >= 0.9
        and metrics["observed_foil_count"] >= 3
    )
    return passed, metrics


def _validate_source_monitoring(step_dir: Path) -> tuple[bool, dict[str, object]]:
    results = _safe_read_csv(step_dir / "results.csv")
    summary = _safe_read_csv(step_dir / "summary_by_pair_type.csv")
    pair_summary = _safe_read_csv(step_dir / "summary_by_pair.csv")
    metrics = {
        "result_row_count": int(len(results)),
        "summary_row_count": int(len(summary)),
        "pair_summary_row_count": int(len(pair_summary)),
        "max_pair_valid_rate": float(summary["pair_valid_rate_mean"].fillna(0.0).max())
        if "pair_valid_rate_mean" in summary.columns and len(summary)
        else 0.0,
        "observed_pair_type_count": int(summary["pair_type"].nunique()) if "pair_type" in summary.columns else 0,
    }
    passed = (
        metrics["result_row_count"] > 0
        and metrics["summary_row_count"] > 0
        and metrics["pair_summary_row_count"] > 0
        and metrics["max_pair_valid_rate"] >= 0.9
        and metrics["observed_pair_type_count"] >= 2
    )
    return passed, metrics


def _validate_longform_return_v3(step_dir: Path) -> tuple[bool, dict[str, object]]:
    results = _safe_read_csv(step_dir / "results.csv")
    chunks = _safe_read_csv(step_dir / "chunk_curves.csv")
    summary = _safe_read_csv(step_dir / "summary.csv")
    style_values = results["final_chunk_style_preference"].dropna().astype(float).tolist() if "final_chunk_style_preference" in results.columns else []
    metrics = {
        "result_row_count": int(len(results)),
        "chunk_row_count": int(len(chunks)),
        "summary_row_count": int(len(summary)),
        "max_forced_shift_magnitude": float(results["forced_shift_magnitude"].fillna(0.0).max()) if len(results) else 0.0,
        "max_chunk_index": int(chunks["chunk_index"].max()) if len(chunks) and "chunk_index" in chunks.columns else 0,
        "style_preference_row_count": int(len(style_values)),
        "max_abs_final_style_preference": float(max(abs(value) for value in style_values)) if style_values else 0.0,
    }
    passed = (
        metrics["result_row_count"] > 0
        and metrics["chunk_row_count"] > 0
        and metrics["summary_row_count"] > 0
        and metrics["max_chunk_index"] >= 2
        and metrics["style_preference_row_count"] > 0
    )
    return passed, metrics


def _validate_step(step_name: str, expected_output: Path) -> tuple[bool, dict[str, object]]:
    step_dir = expected_output.parent
    if step_name == "self_other_boundary_transfer_v5":
        return _validate_self_other(step_dir)
    if step_name == "self_prediction_transfer_v3" or step_name.startswith("self_prediction_transfer_v3_"):
        return _validate_self_prediction(step_dir)
    if step_name == "commitment_persistence_v2_adversarial":
        return _validate_commitment_runtime(step_dir)
    if step_name == "commitment_persistence_v2" or step_name.startswith("commitment_persistence_v2_"):
        return _validate_commitment(step_dir)
    if step_name == "self_recognition_nearfoil_v2":
        return _validate_self_recognition(step_dir)
    if step_name == "diachronic_ship_of_theseus_graft":
        return _validate_diachronic_graft(step_dir)
    if step_name == "temporal_authorship_matrix" or step_name.startswith("temporal_authorship_matrix_"):
        return _validate_temporal_authorship_matrix(step_dir)
    if step_name == "checkpoint_age_recognition_v2":
        return _validate_checkpoint_age_recognition(step_dir)
    if step_name == "behavioral_fingerprint_transfer" or step_name.startswith("behavioral_fingerprint_transfer_"):
        return _validate_behavioral_fingerprint(step_dir)
    if step_name == "kinship_ladder_dissociation":
        return _validate_kinship_ladder(step_dir)
    if step_name == "source_monitoring_attribution":
        return _validate_source_monitoring(step_dir)
    if step_name == "longform_return_v3":
        return _validate_longform_return_v3(step_dir)
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
