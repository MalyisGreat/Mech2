from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from identity_battery_common import add_src_to_path, load_yaml_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a report for checkpoint age recognition v2.")
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


def _save_distance_curve(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    plot_df = df.copy().sort_values("comparison_step")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(plot_df["comparison_step"], plot_df["choose_anchor_centered_mean"], marker="o", linewidth=2.0)
    ax.set_xlabel("Comparison checkpoint step")
    ax.set_ylabel("Choose-current rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    add_src_to_path()
    config = load_yaml_config(args.config)

    output_root = Path(config["output_root"])
    run_dir = output_root / "checkpoint_age_recognition_v2"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(run_dir / "results.csv")
    by_pair = pd.read_csv(run_dir / "summary_by_pair.csv")
    by_comparison = pd.read_csv(run_dir / "summary_by_comparison.csv")
    generation_quality = pd.read_csv(run_dir / "generation_quality_summary.csv")
    prompt_div = pd.read_csv(run_dir / "prompt_divergence_summary.csv")

    report_lines = [
        "# Checkpoint Age Recognition V2 Report",
        "",
        f"- Config: `{args.config}`",
        f"- Output root: `{output_root}`",
        f"- Model: `{config['model_id']}`",
        f"- Anchor revision: `{config['anchor_revision']}`",
        f"- Comparison revisions: `{', '.join(str(rev) for rev in config['comparison_revisions'])}`",
        f"- Evaluator revisions: `{', '.join(str(rev) for rev in config.get('evaluator_revisions', [config['anchor_revision']]))}`",
        "",
    ]

    stats_rows: list[dict[str, object]] = []
    metrics = [
        _metric_row("checkpoint_age_centered_choice_rate", results["choose_anchor_centered"].astype(float).tolist()),
        _metric_row("checkpoint_age_raw_choice_rate", results["choose_anchor_raw"].astype(float).tolist()),
        _metric_row("checkpoint_age_centered_margin_logprob", results["centered_margin_logprob"].astype(float).tolist()),
        _metric_row("checkpoint_age_raw_margin_logprob", results["raw_margin_logprob"].astype(float).tolist()),
        _metric_row("prompt_screen_mean_anchor_js", prompt_div["mean_anchor_js"].astype(float).tolist()),
    ]
    for metric in metrics:
        stats_rows.append(metric)
        report_lines.append(
            f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
            f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` units"
        )

    report_lines.extend(["", "## By Comparison Revision", ""])
    for row in by_comparison.itertuples(index=False):
        report_lines.append(
            f"- `{row.comparison_revision}`: choose-current `{float(row.choose_anchor_centered_mean):.4f}`, "
            f"centered margin `{float(row.centered_margin_logprob_mean):.4f}`, "
            f"n=`{int(row.n)}`"
        )

    report_lines.extend(["", "## By Evaluator-Comparison Pair", ""])
    for row in by_pair.itertuples(index=False):
        report_lines.append(
            f"- evaluator `{row.evaluator_revision}` vs `{row.comparison_revision}`: "
            f"choose-current `{float(row.choose_anchor_centered_mean):.4f}`, "
            f"centered margin `{float(row.centered_margin_logprob_mean):.4f}`, "
            f"n=`{int(row.n)}`"
        )

    report_lines.extend(["", "## Generation Quality", ""])
    for row in generation_quality.itertuples(index=False):
        report_lines.append(
            f"- `{row.source_revision}`: valid-rate `{float(row.generation_valid_rate):.4f}`, "
            f"unique-ratio `{float(row.unique_token_ratio_mean):.4f}`, "
            f"top-token-rate `{float(row.top_token_rate_mean):.4f}`, "
            f"top-bigram-rate `{float(row.top_bigram_rate_mean):.4f}`, n=`{int(row.n)}`"
        )

    curve_path = figures_dir / "checkpoint_age_recognition_curve.png"
    comparison_plot = by_comparison.copy()
    comparison_plot["comparison_step"] = comparison_plot["comparison_revision"].astype(str).str.extract(r"(\d+)").astype(float)
    _save_distance_curve(comparison_plot, curve_path, "Checkpoint age recognition v2")

    report_lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- Choice-rate curve: `{curve_path}`",
        ]
    )

    (output_root / "checkpoint_age_recognition_v2_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_root / "checkpoint_age_recognition_v2_stats.json").write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
