from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from identity_battery_common import add_src_to_path, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a compact inferential report for the identity master suite.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _bootstrap_mean_ci(values: list[float]) -> tuple[float, float]:
    add_src_to_path()
    from identity_stability.identity_analysis import bootstrap_mean_ci

    return bootstrap_mean_ci(values, iters=2000, seed=123)


def _metric_row(name: str, values: list[float]) -> dict[str, object]:
    finite = [float(value) for value in values if np.isfinite(value)]
    mean = float(np.mean(finite)) if finite else float("nan")
    ci_low, ci_high = _bootstrap_mean_ci(finite) if finite else (float("nan"), float("nan"))
    return {
        "metric": name,
        "n": len(finite),
        "mean": mean,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def _paired_contrast_rows(
    df: pd.DataFrame,
    *,
    frame_col: str,
    baseline_frame: str,
    compare_frame: str,
    unit_cols: list[str],
    metric_cols: list[str],
    prefix: str,
) -> list[dict[str, object]]:
    subset = df[df[frame_col].isin([baseline_frame, compare_frame])].copy()
    if subset.empty:
        return []
    wide = subset.pivot_table(index=unit_cols, columns=frame_col, values=metric_cols, aggfunc="mean")
    rows: list[dict[str, object]] = []
    for metric in metric_cols:
        if metric not in wide.columns.get_level_values(0):
            continue
        metric_wide = wide[metric]
        if baseline_frame not in metric_wide.columns or compare_frame not in metric_wide.columns:
            continue
        diffs = (metric_wide[compare_frame] - metric_wide[baseline_frame]).dropna().astype(float).tolist()
        rows.append(_metric_row(f"{prefix}_{compare_frame}_minus_{baseline_frame}_{metric}", diffs))
    return rows


def _optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    repo_root = add_src_to_path()
    args = parse_args()
    config = load_yaml_config(args.config)
    output_root = Path(config["output_root"])
    report_path = output_root / "master_suite_report.md"
    stats_path = output_root / "master_suite_stats.json"

    rows: list[dict[str, object]] = []
    report_lines = [
        "# Identity Master Suite Report",
        "",
        f"- Config: `{args.config}`",
        f"- Output root: `{output_root}`",
        "",
    ]

    sections: list[tuple[str, list[dict[str, object]] | None, str | None]] = []

    self_other_item = _optional_csv(output_root / "self_other_boundary_transfer_v5" / "item_summary.csv")
    if self_other_item is not None:
        self_other_metrics = [
            _metric_row("self_other_control_accuracy", self_other_item.loc[self_other_item["item_type"] == "control", "control_accuracy_mean"].tolist()),
            _metric_row("self_other_structure_score", self_other_item.loc[self_other_item["item_type"] == "descriptive", "structure_score"].tolist()),
            _metric_row("self_other_non_tie_rate", self_other_item.loc[self_other_item["item_type"] == "descriptive", "non_tie_rate"].tolist()),
            _metric_row("self_other_contradiction_rate", self_other_item.loc[self_other_item["item_type"] == "descriptive", "contradiction_rate"].tolist()),
        ]
        self_other_metrics.extend(
            _paired_contrast_rows(
                self_other_item,
                frame_col="identity_frame",
                baseline_frame="baseline_helpful",
                compare_frame="family_self",
                unit_cols=["seed", "model_id", "model_size_label", "item_type", "item_id", "domain"],
                metric_cols=["control_accuracy_mean", "structure_score", "non_tie_rate", "contradiction_rate"],
                prefix="self_other",
            )
        )
        sections.append(("Self/Other Boundary V5", self_other_metrics, None))
    else:
        sections.append(("Self/Other Boundary V5", None, "skipped: no completed boundary output in this suite"))

    self_pred_results = _optional_csv(output_root / "self_prediction_transfer_v3" / "results.csv")
    if self_pred_results is not None:
        self_pred_metrics = [
            _metric_row("self_prediction_self_accuracy", self_pred_results["self_accuracy_mean"].tolist()),
            _metric_row("self_prediction_other_accuracy", self_pred_results["other_accuracy_mean"].tolist()),
            _metric_row("self_prediction_gap_accuracy", self_pred_results["gap_direction_accuracy_mean"].tolist()),
            _metric_row("self_prediction_actual_gap_rate", self_pred_results["actual_gap_rate"].tolist()),
        ]
        self_pred_metrics.extend(
            _paired_contrast_rows(
                self_pred_results,
                frame_col="identity_frame",
                baseline_frame="baseline_helpful",
                compare_frame="family_self",
                unit_cols=["seed", "model_id", "model_size_label", "prompt_id", "prompt_family"],
                metric_cols=["self_accuracy_mean", "other_accuracy_mean", "gap_direction_accuracy_mean", "actual_gap_rate"],
                prefix="self_prediction",
            )
        )
        sections.append(("Self Prediction Transfer V3", self_pred_metrics, None))
    else:
        sections.append(("Self Prediction Transfer V3", None, "skipped: no completed self-prediction output in this suite"))

    commitment_results = _optional_csv(output_root / "commitment_persistence_v2" / "results.csv")
    if commitment_results is not None:
        commitment_metrics = [
            _metric_row("commitment_adherence", commitment_results["adherence_rate"].tolist()),
            _metric_row("commitment_reveal_accuracy", commitment_results["reveal_accuracy"].tolist()),
            _metric_row("commitment_reveal_pair_accuracy", commitment_results["reveal_pair_accuracy"].tolist()),
            _metric_row("commitment_label_a_rate", commitment_results["label_a_rate"].tolist()),
        ]
        commitment_metrics.extend(
            _paired_contrast_rows(
                commitment_results,
                frame_col="identity_frame",
                baseline_frame="baseline_helpful",
                compare_frame="family_self",
                unit_cols=["seed", "model_id", "model_size_label", "condition", "assigned_commitment"],
                metric_cols=["adherence_rate", "reveal_accuracy", "reveal_pair_accuracy", "label_a_rate"],
                prefix="commitment",
            )
        )
        sections.append(("Commitment Persistence V2", commitment_metrics, None))
    else:
        sections.append(("Commitment Persistence V2", None, "skipped: no completed commitment output in this suite"))

    nearfoil_results = _optional_csv(output_root / "self_recognition_nearfoil_v2" / "results.csv")
    if nearfoil_results is not None:
        nearfoil_metrics = [
            _metric_row("nearfoil_ownership_accuracy", nearfoil_results["choose_self_baseline"].tolist()),
            _metric_row("nearfoil_pair_valid_rate", nearfoil_results["pair_valid"].tolist()),
            _metric_row("nearfoil_same_frame_resample", nearfoil_results.loc[nearfoil_results["difficulty_name"] == "same_frame_resample", "choose_self_baseline"].tolist()),
        ]
        nearfoil_metrics.extend(
            _paired_contrast_rows(
                nearfoil_results,
                frame_col="identity_frame",
                baseline_frame="baseline_helpful",
                compare_frame="family_self",
                unit_cols=["seed", "model_id", "model_size_label", "axis_name", "prompt_id", "difficulty_name"],
                metric_cols=["choose_self_baseline", "pair_valid", "style_distance", "semantic_overlap"],
                prefix="nearfoil",
            )
        )
        sections.append(("Self Recognition Near-Foil V2", nearfoil_metrics, None))
    else:
        sections.append(("Self Recognition Near-Foil V2", None, "skipped: no completed near-foil output in this suite"))

    for title, metrics, note in sections:
        report_lines.extend([f"## {title}", ""])
        if metrics is None:
            report_lines.append(f"- `{note}`")
            report_lines.append("")
            continue
        for metric in metrics:
            rows.append(metric)
            report_lines.append(
                f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
                f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` natural units"
            )
        report_lines.append("")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    stats_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
