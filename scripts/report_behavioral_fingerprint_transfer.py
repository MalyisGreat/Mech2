from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from identity_battery_common import add_src_to_path, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a compact report for behavioral fingerprint transfer.")
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
    add_src_to_path()
    args = parse_args()
    config = load_yaml_config(args.config)
    output_root = Path(config["output_root"])
    report_path = output_root / "behavioral_fingerprint_transfer_report.md"
    stats_path = output_root / "behavioral_fingerprint_transfer_stats.json"

    results = pd.read_csv(output_root / "behavioral_fingerprint_transfer" / "results.csv")
    feature_results = pd.read_csv(output_root / "behavioral_fingerprint_transfer" / "feature_results.csv")

    rows: list[dict[str, object]] = []
    report_lines = [
        "# Behavioral Fingerprint Transfer Report",
        "",
        f"- Config: `{args.config}`",
        f"- Output root: `{output_root}`",
        "",
    ]

    headline_metrics = [
        _metric_row("behavioral_fingerprint_self_profile_accuracy", results["self_profile_accuracy_mean"].tolist()),
        _metric_row("behavioral_fingerprint_matched_decoy_accuracy", results["matched_decoy_accuracy_mean"].tolist()),
        _metric_row("behavioral_fingerprint_scrambled_accuracy", results["scrambled_profile_accuracy_mean"].tolist()),
        _metric_row("behavioral_fingerprint_self_minus_decoy_accuracy", results["self_minus_decoy_accuracy"].tolist()),
        _metric_row("behavioral_fingerprint_self_minus_scrambled_accuracy", results["self_minus_scrambled_accuracy"].tolist()),
        _metric_row("behavioral_fingerprint_triadic_choose_self", results["triadic_choose_self"].tolist()),
        _metric_row("behavioral_fingerprint_triadic_nearest_accuracy", results["triadic_nearest_accuracy"].tolist()),
        _metric_row("behavioral_fingerprint_self_margin_vs_decoy", results["self_margin_vs_decoy"].tolist()),
        _metric_row("behavioral_fingerprint_self_margin_vs_scrambled", results["self_margin_vs_scrambled"].tolist()),
    ]
    headline_metrics.extend(
        _paired_contrast_rows(
            results,
            frame_col="identity_frame",
            baseline_frame="baseline_helpful",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "prompt_id", "prompt_family"],
            metric_cols=[
                "self_profile_accuracy_mean",
                "matched_decoy_accuracy_mean",
                "self_minus_decoy_accuracy",
                "self_minus_scrambled_accuracy",
                "triadic_nearest_accuracy",
                "self_margin_vs_decoy",
                "self_margin_vs_scrambled",
            ],
            prefix="behavioral_fingerprint",
        )
    )
    headline_metrics.extend(
        _paired_contrast_rows(
            results,
            frame_col="identity_frame",
            baseline_frame="stable_style_policy",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "prompt_id", "prompt_family"],
            metric_cols=[
                "self_profile_accuracy_mean",
                "matched_decoy_accuracy_mean",
                "self_minus_decoy_accuracy",
                "self_minus_scrambled_accuracy",
                "triadic_nearest_accuracy",
                "self_margin_vs_decoy",
                "self_margin_vs_scrambled",
            ],
            prefix="behavioral_fingerprint",
        )
    )
    headline_metrics.extend(
        _frame_family_rows(
            results,
            frame_col="identity_frame",
            family_col="prompt_family",
            metric_cols=["self_minus_decoy_accuracy", "triadic_nearest_accuracy", "self_margin_vs_decoy"],
            prefix="behavioral_fingerprint_family",
        )
    )

    report_lines.extend(["## Unit-Level Results", ""])
    for metric in headline_metrics:
        rows.append(metric)
        report_lines.append(
            f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
            f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` natural units"
        )

    report_lines.extend(["", "## Feature-Level Results", ""])
    for (frame_name, profile_source), sub in feature_results.groupby(["identity_frame", "profile_source"], as_index=False):
        accuracy_row = _metric_row(
            f"behavioral_fingerprint_feature_{frame_name}_{profile_source}_accuracy",
            sub["correct"].tolist(),
        )
        valid_row = _metric_row(
            f"behavioral_fingerprint_feature_{frame_name}_{profile_source}_valid_choice",
            sub["valid_choice"].tolist(),
        )
        rows.extend([accuracy_row, valid_row])
        for metric in (accuracy_row, valid_row):
            report_lines.append(
                f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
                f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` feature rows"
            )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    stats_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
