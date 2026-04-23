from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from identity_battery_common import add_src_to_path, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a compact report for the family-self mediation confirm.")
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


def _frame_family_rows(
    df: pd.DataFrame,
    *,
    frame_col: str,
    family_col: str,
    metric_cols: list[str],
    prefix: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (frame_name, family_name), sub in df.groupby([frame_col, family_col], as_index=False):
        for metric in metric_cols:
            values = sub[metric].dropna().astype(float).tolist()
            rows.append(_metric_row(f"{prefix}_{frame_name}_{family_name}_{metric}", values))
    return rows


def main() -> None:
    repo_root = add_src_to_path()
    args = parse_args()
    config = load_yaml_config(args.config)
    output_root = Path(config["output_root"])
    report_path = output_root / "family_self_mediation_confirm_report.md"
    stats_path = output_root / "family_self_mediation_confirm_stats.json"

    self_pred_results = pd.read_csv(output_root / "self_prediction_transfer_v3" / "results.csv")
    nearfoil_results = pd.read_csv(output_root / "self_recognition_nearfoil_v2" / "results.csv")
    nearfoil_results["choose_self_baseline"] = pd.to_numeric(nearfoil_results["choose_self_baseline"], errors="coerce")

    rows: list[dict[str, object]] = []
    report_lines = [
        "# Family-Self Mediation Confirm Report",
        "",
        f"- Config: `{args.config}`",
        f"- Output root: `{output_root}`",
        "",
        "## Self Prediction Transfer V3",
        "",
    ]

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
            unit_cols=["seed", "model_id", "prompt_id", "prompt_family"],
            metric_cols=["self_accuracy_mean", "other_accuracy_mean", "gap_direction_accuracy_mean", "actual_gap_rate", "discriminative_win"],
            prefix="self_prediction",
        )
    )
    self_pred_metrics.extend(
        _paired_contrast_rows(
            self_pred_results,
            frame_col="identity_frame",
            baseline_frame="stable_style_policy",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "prompt_id", "prompt_family"],
            metric_cols=["self_accuracy_mean", "other_accuracy_mean", "gap_direction_accuracy_mean", "actual_gap_rate", "discriminative_win"],
            prefix="self_prediction",
        )
    )
    self_pred_metrics.extend(
        _frame_family_rows(
            self_pred_results,
            frame_col="identity_frame",
            family_col="prompt_family",
            metric_cols=["self_accuracy_mean", "gap_direction_accuracy_mean", "actual_gap_rate"],
            prefix="self_prediction_family",
        )
    )
    for metric in self_pred_metrics:
        rows.append(metric)
        report_lines.append(
            f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
            f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` natural units"
        )

    report_lines.extend(["", "## Self Recognition Near-Foil V2", ""])

    valid_nearfoil = nearfoil_results[nearfoil_results["choose_self_baseline"].notna()].copy()
    nearfoil_metrics = [
        _metric_row("nearfoil_ownership_accuracy", valid_nearfoil["choose_self_baseline"].tolist()),
        _metric_row("nearfoil_pair_valid_rate", nearfoil_results["pair_valid"].tolist()),
        _metric_row(
            "nearfoil_same_frame_resample",
            valid_nearfoil.loc[valid_nearfoil["difficulty_name"] == "same_frame_resample", "choose_self_baseline"].tolist(),
        ),
    ]
    nearfoil_metrics.extend(
        _paired_contrast_rows(
            valid_nearfoil,
            frame_col="identity_frame",
            baseline_frame="baseline_helpful",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "axis_name", "prompt_id", "difficulty_name"],
            metric_cols=["choose_self_baseline"],
            prefix="nearfoil",
        )
    )
    nearfoil_metrics.extend(
        _paired_contrast_rows(
            valid_nearfoil,
            frame_col="identity_frame",
            baseline_frame="stable_style_policy",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "axis_name", "prompt_id", "difficulty_name"],
            metric_cols=["choose_self_baseline"],
            prefix="nearfoil",
        )
    )
    nearfoil_metrics.extend(
        _paired_contrast_rows(
            nearfoil_results,
            frame_col="identity_frame",
            baseline_frame="baseline_helpful",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "axis_name", "prompt_id", "difficulty_name"],
            metric_cols=["pair_valid", "style_distance", "semantic_overlap"],
            prefix="nearfoil",
        )
    )
    nearfoil_metrics.extend(
        _paired_contrast_rows(
            nearfoil_results,
            frame_col="identity_frame",
            baseline_frame="stable_style_policy",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "axis_name", "prompt_id", "difficulty_name"],
            metric_cols=["pair_valid", "style_distance", "semantic_overlap"],
            prefix="nearfoil",
        )
    )
    nearfoil_metrics.extend(
        _frame_family_rows(
            valid_nearfoil,
            frame_col="identity_frame",
            family_col="prompt_family",
            metric_cols=["choose_self_baseline"],
            prefix="nearfoil_family",
        )
    )
    for metric in nearfoil_metrics:
        rows.append(metric)
        report_lines.append(
            f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
            f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` natural units"
        )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    stats_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
