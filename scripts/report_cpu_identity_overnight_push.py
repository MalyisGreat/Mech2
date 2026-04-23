from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from identity_battery_common import add_src_to_path, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a compact report for the overnight CPU identity push suite.")
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


def _paired_diff(
    df: pd.DataFrame,
    *,
    frame_col: str,
    baseline_frame: str,
    compare_frame: str,
    unit_cols: list[str],
    metric_col: str,
    name: str,
) -> dict[str, object] | None:
    subset = df[df[frame_col].isin([baseline_frame, compare_frame])].copy()
    if subset.empty:
        return None
    wide = subset.pivot_table(index=unit_cols, columns=frame_col, values=metric_col, aggfunc="mean")
    if compare_frame not in wide.columns or baseline_frame not in wide.columns:
        return None
    diffs = (wide[compare_frame] - wide[baseline_frame]).dropna().astype(float).tolist()
    if not diffs:
        return None
    return _metric_row(name, diffs)


def _optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    add_src_to_path()
    args = parse_args()
    config = load_yaml_config(args.config)
    output_root = Path(config["output_root"])
    report_path = output_root / "cpu_identity_overnight_push_report.md"
    stats_path = output_root / "cpu_identity_overnight_push_stats.json"

    rows: list[dict[str, object]] = []
    report_lines = [
        "# CPU Identity Overnight Push Report",
        "",
        f"- Config: `{args.config}`",
        f"- Output root: `{output_root}`",
        "",
    ]

    template_runs = config.get("template_runs", [])
    report_lines.extend(["## Prompt Template Screening", ""])
    for template_row in template_runs:
        run_root = output_root / str(template_row["run_root"]) / "self_prediction_transfer_v3" / "results.csv"
        df = _optional_csv(run_root)
        if df is None:
            report_lines.append(f"- `{template_row['name']}`: missing output")
            continue
        metrics = [
            _metric_row(
                f"template_{template_row['name']}_gap_accuracy",
                df["gap_direction_accuracy_mean"].tolist(),
            ),
            _metric_row(
                f"template_{template_row['name']}_valid_choice_rate",
                df["valid_choice_rate"].tolist(),
            ),
        ]
        delta = _paired_diff(
            df,
            frame_col="identity_frame",
            baseline_frame="baseline_helpful",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "prompt_id", "prompt_family"],
            metric_col="gap_direction_accuracy_mean",
            name=f"template_{template_row['name']}_family_self_minus_baseline_gap_accuracy",
        )
        if delta is not None:
            metrics.append(delta)
        for metric in metrics:
            rows.append(metric)
            report_lines.append(
                f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
                f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` units"
            )
    report_lines.append("")

    kinship_pairs = _optional_csv(output_root / "kinship_ladder_dissociation" / "kinship_ladder_dissociation" / "summary_by_pair.csv")
    kinship_summary = _optional_csv(output_root / "kinship_ladder_dissociation" / "kinship_ladder_dissociation" / "summary_by_foil.csv")
    report_lines.extend(["## Kinship Ladder Dissociation", ""])
    if kinship_pairs is None or kinship_summary is None:
        report_lines.append("- missing output")
    else:
        metrics = [
            _metric_row("kinship_choose_host", kinship_pairs["pair_choose_host_rate"].tolist()),
            _metric_row("kinship_swap_consistency", kinship_pairs["swap_consistency"].tolist()),
        ]
        delta = _paired_diff(
            kinship_pairs,
            frame_col="identity_frame",
            baseline_frame="baseline_helpful",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "foil_kind", "prompt_id", "prompt_family"],
            metric_col="pair_choose_host_rate",
            name="kinship_family_self_minus_baseline_choose_host",
        )
        if delta is not None:
            metrics.append(delta)
        for metric in metrics:
            rows.append(metric)
            report_lines.append(
                f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
                f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` pairs"
            )
        for row in kinship_summary.itertuples(index=False):
            report_lines.append(
                f"- `{row.identity_frame} / {row.foil_kind}`: choose-host `{float(row.choose_host_rate_mean):.4f}`, "
                f"swap-consistency `{float(row.swap_consistency_mean):.4f}`, n=`{int(row.n_pairs)}`"
            )
    report_lines.append("")

    fingerprint_results = _optional_csv(output_root / "behavioral_fingerprint_nuisance" / "behavioral_fingerprint_transfer" / "results.csv")
    report_lines.extend(["## Behavioral Fingerprint Stability", ""])
    if fingerprint_results is None:
        report_lines.append("- missing output")
    else:
        metrics = [
            _metric_row("fingerprint_self_minus_decoy_accuracy", fingerprint_results["self_minus_decoy_accuracy"].tolist()),
            _metric_row("fingerprint_self_margin_vs_scrambled", fingerprint_results["self_margin_vs_scrambled"].tolist()),
        ]
        delta = _paired_diff(
            fingerprint_results,
            frame_col="identity_frame",
            baseline_frame="baseline_helpful",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "prompt_id", "prompt_family"],
            metric_col="self_minus_decoy_accuracy",
            name="fingerprint_family_self_minus_baseline_self_minus_decoy",
        )
        if delta is not None:
            metrics.append(delta)
        for metric in metrics:
            rows.append(metric)
            report_lines.append(
                f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
                f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` units"
            )
    report_lines.append("")

    commitment_results = _optional_csv(output_root / "commitment_persistence_adversarial" / "commitment_persistence_v2" / "results.csv")
    report_lines.extend(["## Commitment Persistence", ""])
    if commitment_results is None:
        report_lines.append("- missing output")
    else:
        metrics = [
            _metric_row("commitment_adherence", commitment_results["adherence_rate"].tolist()),
            _metric_row("commitment_reveal_pair_accuracy", commitment_results["reveal_pair_accuracy"].tolist()),
            _metric_row("commitment_post_counter_adherence", commitment_results["post_counter_adherence_rate"].tolist()),
        ]
        delta = _paired_diff(
            commitment_results,
            frame_col="identity_frame",
            baseline_frame="baseline_helpful",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "condition", "assigned_commitment"],
            metric_col="reveal_pair_accuracy",
            name="commitment_family_self_minus_baseline_reveal_pair_accuracy",
        )
        if delta is not None:
            metrics.append(delta)
        for metric in metrics:
            rows.append(metric)
            report_lines.append(
                f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
                f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` dialogues"
            )
    report_lines.append("")

    source_pairs = _optional_csv(output_root / "source_monitoring_attribution" / "source_monitoring_attribution" / "summary_by_pair.csv")
    source_summary = _optional_csv(output_root / "source_monitoring_attribution" / "source_monitoring_attribution" / "summary_by_pair_type.csv")
    report_lines.extend(["## Source Monitoring Attribution", ""])
    if source_pairs is None or source_summary is None:
        report_lines.append("- missing output")
    else:
        metrics = [
            _metric_row("source_monitoring_choose_self", source_pairs["pair_choose_self_rate"].tolist()),
            _metric_row("source_monitoring_swap_consistency", source_pairs["swap_consistency"].tolist()),
        ]
        delta = _paired_diff(
            source_pairs,
            frame_col="identity_frame",
            baseline_frame="baseline_helpful",
            compare_frame="family_self",
            unit_cols=["seed", "model_id", "pair_type", "variant", "prompt_id", "prompt_family"],
            metric_col="pair_choose_self_rate",
            name="source_monitoring_family_self_minus_baseline_choose_self",
        )
        if delta is not None:
            metrics.append(delta)
        for metric in metrics:
            rows.append(metric)
            report_lines.append(
                f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
                f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` pairs"
            )
        for row in source_summary.itertuples(index=False):
            report_lines.append(
                f"- `{row.identity_frame} / {row.pair_type} / {row.variant}`: choose-self `{float(row.choose_self_rate_mean):.4f}`, "
                f"swap-consistency `{float(row.swap_consistency_mean):.4f}`, n=`{int(row.n_pairs)}`"
            )
    report_lines.append("")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    stats_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
