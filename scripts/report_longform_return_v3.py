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
    parser = argparse.ArgumentParser(description="Write a report for long-form return v3.")
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


def _save_chunk_curve(chunk_df: pd.DataFrame, path: Path, title: str) -> None:
    if chunk_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    plot_df = (
        chunk_df.groupby("chunk_index", as_index=False)["chunk_return_to_baseline_index"]
        .mean(numeric_only=True)
        .sort_values("chunk_index")
    )
    ax.plot(plot_df["chunk_index"], plot_df["chunk_return_to_baseline_index"], marker="o", linewidth=2.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Return chunk")
    ax.set_ylabel("Return-to-baseline index")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    add_src_to_path()
    config = load_yaml_config(args.config)

    output_root = Path(config["output_root"])
    run_dir = output_root / "longform_return_v3"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(run_dir / "results.csv")
    chunks = pd.read_csv(run_dir / "chunk_curves.csv")
    summary = pd.read_csv(run_dir / "summary.csv")

    stats_rows = [
        _metric_row(
            "longform_return_chunk1_style_preference",
            results["chunk_1_style_preference"].dropna().astype(float).tolist(),
        ),
        _metric_row(
            "longform_return_final_style_preference",
            results["final_chunk_style_preference"].dropna().astype(float).tolist(),
        ),
        _metric_row(
            "longform_return_chunk1_axis_index",
            results["chunk_1_return_to_baseline_index"].dropna().astype(float).tolist(),
        ),
        _metric_row(
            "longform_return_final_axis_index",
            results["final_chunk_return_to_baseline_index"].dropna().astype(float).tolist(),
        ),
        _metric_row(
            "longform_return_half_life_chunk",
            results["return_half_life_chunk"].dropna().astype(float).tolist(),
        ),
        _metric_row(
            "longform_forced_shift_magnitude",
            results["forced_shift_magnitude"].dropna().astype(float).tolist(),
        ),
    ]

    report_lines = [
        "# Long-Form Return V3 Report",
        "",
        f"- Config: `{args.config}`",
        f"- Output root: `{output_root}`",
        f"- Models: `{', '.join(str(x) for x in config['model_ids'])}`",
        f"- Frames: `{', '.join(str(x) for x in config['identity_frames'])}`",
        f"- Items: `{len(config['item_ids'])}`",
        f"- Seeds: `{', '.join(str(x) for x in config.get('seeds', [config.get('seed', 7)]))}`",
        "",
    ]

    for metric in stats_rows:
        report_lines.append(
            f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
            f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` units"
        )

    report_lines.extend(["", "## By Model / Frame / Axis", ""])
    for row in summary.itertuples(index=False):
        report_lines.append(
            f"- `{row.model_size_label} / {row.identity_frame} / {row.axis_name}`: "
            f"chunk1-style `{float(row.chunk_1_style_preference_mean):.4f}`, "
            f"final-style `{float(row.final_chunk_style_preference_mean):.4f}`, "
            f"chunk1-axis `{float(row.chunk_1_return_to_baseline_index_mean):.4f}`, "
            f"final-axis `{float(row.final_chunk_return_to_baseline_index_mean):.4f}`, "
            f"half-life `{float(row.return_half_life_chunk_mean):.4f}`, "
            f"forced-shift `{float(row.forced_shift_magnitude_mean):.4f}`, n=`{int(row.n)}`"
        )

    overall_curve_path = figures_dir / "longform_return_v3_chunk_curve.png"
    _save_chunk_curve(chunks, overall_curve_path, "Long-form return v3: overall chunk curve")
    report_lines.extend(["", "## Figures", "", f"- Overall chunk curve: `{overall_curve_path}`"])

    (output_root / "longform_return_v3_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_root / "longform_return_v3_stats.json").write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
