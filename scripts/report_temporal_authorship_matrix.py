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
    parser = argparse.ArgumentParser(description="Write a report for the temporal authorship matrix experiment.")
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


def _save_heatmap(matrix_df: pd.DataFrame, path: Path, title: str) -> None:
    if matrix_df.empty:
        return
    data = matrix_df.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(matrix_df.columns)))
    ax.set_xticklabels(list(matrix_df.columns), rotation=45, ha="right")
    ax.set_yticks(range(len(matrix_df.index)))
    ax.set_yticklabels(list(matrix_df.index))
    ax.set_xlabel("Source checkpoint output")
    ax.set_ylabel("Evaluator checkpoint")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean avg logprob")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_self_pref_curve(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    plot_df["evaluator_step"] = plot_df["evaluator_revision"].astype(str).str.extract(r"(\d+)").astype(float)
    plot_df = plot_df.sort_values("evaluator_step")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(plot_df["evaluator_step"], plot_df["self_preference_rate"], marker="o", linewidth=2.0)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Self-preference rate")
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
    run_dir = output_root / "temporal_authorship_matrix"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(run_dir / "results.csv")
    self_pref = pd.read_csv(run_dir / "self_preference_summary.csv")
    pair_summary = pd.read_csv(run_dir / "summary_by_pair.csv")
    preference_matrix = pd.read_csv(run_dir / "authorship_preference_matrix.csv", index_col=0)
    generation_quality = pd.read_csv(run_dir / "generation_quality_summary.csv")

    stats_rows: list[dict[str, object]] = []
    report_lines = [
        "# Temporal Authorship Matrix Report",
        "",
        f"- Config: `{args.config}`",
        f"- Output root: `{output_root}`",
        f"- Model: `{config['model_id']}`",
        f"- Revisions: `{', '.join(str(rev) for rev in config['revisions'])}`",
        "",
    ]

    prompt_winners = results.groupby(["evaluator_revision", "prompt_id"], as_index=False).first()
    overall_self_pref = _metric_row(
        "temporal_authorship_self_preference_rate",
        (prompt_winners["winner_source_revision"].astype(str) == prompt_winners["evaluator_revision"].astype(str)).astype(float).tolist(),
    )
    diagonal_rows = pair_summary[pair_summary["evaluator_revision"] == pair_summary["source_revision"]]
    off_rows = pair_summary[pair_summary["evaluator_revision"] != pair_summary["source_revision"]]
    diagonal_metric = _metric_row(
        "temporal_authorship_diagonal_avg_logprob",
        diagonal_rows["avg_logprob_mean"].astype(float).tolist(),
    )
    off_metric = _metric_row(
        "temporal_authorship_off_diagonal_avg_logprob",
        off_rows["avg_logprob_mean"].astype(float).tolist(),
    )
    margin_metric = _metric_row(
        "temporal_authorship_diagonal_margin_logprob",
        self_pref["mean_diagonal_margin_logprob"].dropna().astype(float).tolist(),
    )

    age_eval = prompt_winners[prompt_winners["evaluator_revision"] == "step143000"].copy()
    age_metric = _metric_row(
        "checkpoint_age_recognition_final_self_preference_rate",
        (age_eval["winner_source_revision"].astype(str) == "step143000").astype(float).tolist(),
    )

    for metric in [overall_self_pref, diagonal_metric, off_metric, margin_metric, age_metric]:
        stats_rows.append(metric)
        report_lines.append(
            f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
            f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` units"
        )

    report_lines.extend(["", "## By Evaluator", ""])
    for row in self_pref.itertuples(index=False):
        report_lines.append(
            f"- `{row.evaluator_revision}`: self-preference `{float(row.self_preference_rate):.4f}`, "
            f"mean diagonal margin `{float(row.mean_diagonal_margin_logprob):.4f}`, prompts `n={int(row.prompt_count)}`"
        )

    report_lines.extend(["", "## Generation Quality", ""])
    for row in generation_quality.itertuples(index=False):
        report_lines.append(
            f"- `{row.source_revision}`: valid-rate `{float(row.generation_valid_rate):.4f}`, "
            f"unique-ratio `{float(row.unique_token_ratio_mean):.4f}`, "
            f"top-token-rate `{float(row.top_token_rate_mean):.4f}`, "
            f"top-bigram-rate `{float(row.top_bigram_rate_mean):.4f}`, n=`{int(row.n)}`"
        )

    heatmap_path = figures_dir / "temporal_authorship_matrix_heatmap.png"
    curve_path = figures_dir / "temporal_authorship_self_preference_curve.png"
    _save_heatmap(preference_matrix, heatmap_path, "Temporal authorship preference matrix")
    _save_self_pref_curve(self_pref, curve_path, "Temporal authorship self-preference")

    report_lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- Heatmap: `{heatmap_path}`",
            f"- Self-preference curve: `{curve_path}`",
        ]
    )

    (output_root / "temporal_authorship_matrix_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_root / "temporal_authorship_matrix_stats.json").write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
