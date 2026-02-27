from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an experiment run directory.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory containing metrics_full.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    df = pd.read_csv(run_dir / "metrics_full.csv")

    model_summary = (
        df.groupby(["concept_name", "model_id"], as_index=False)
        .agg(
            peak_drift_mean=("peak_drift", "mean"),
            peak_drift_relative_mean=("peak_drift_relative", "mean"),
            end_drift_mean=("end_drift", "mean"),
            end_drift_relative_mean=("end_drift_relative", "mean"),
            recovery_fraction_mean=("recovery_fraction", "mean"),
            overshoot_index_mean=("overshoot_index", "mean"),
            crossed_baseline_rate=("crossed_baseline", "mean"),
            next_token_kl_mean=("next_token_kl", "mean"),
            end_cosine_alignment_mean=("end_cosine_alignment", "mean"),
            rows=("prompt_index", "count"),
        )
        .sort_values("model_id")
    )
    model_summary["end_over_peak"] = model_summary["end_drift_mean"] / model_summary["peak_drift_mean"]
    model_summary.to_csv(run_dir / "model_summary.csv", index=False)

    alpha_summary = (
        df.groupby(["concept_name", "model_id", "alpha"], as_index=False)
        .agg(
            peak_drift_mean=("peak_drift", "mean"),
            peak_drift_relative_mean=("peak_drift_relative", "mean"),
            recovery_fraction_mean=("recovery_fraction", "mean"),
            next_token_kl_mean=("next_token_kl", "mean"),
            rows=("prompt_index", "count"),
        )
        .sort_values(["concept_name", "model_id", "alpha"])
    )
    alpha_summary.to_csv(run_dir / "alpha_summary.csv", index=False)

    layer_summary = (
        df.groupby(["concept_name", "model_id", "layer_index", "vector_method"], as_index=False)
        .agg(
            peak_drift_mean=("peak_drift", "mean"),
            peak_drift_relative_mean=("peak_drift_relative", "mean"),
            recovery_fraction_mean=("recovery_fraction", "mean"),
            crossed_baseline_rate=("crossed_baseline", "mean"),
            rows=("prompt_index", "count"),
        )
        .sort_values(["concept_name", "model_id", "layer_index", "vector_method"])
    )
    layer_summary.to_csv(run_dir / "layer_summary_recomputed.csv", index=False)

    with (run_dir / "summary_generated.md").open("w", encoding="utf-8") as f:
        f.write("# Generated Summary\n\n")
        f.write(f"- Rows: `{len(df)}`\n")
        f.write(f"- Models: `{df['model_id'].nunique()}`\n\n")
        f.write("## Model Summary\n\n")
        for _, row in model_summary.iterrows():
            f.write(
                "- "
                f"{row['concept_name']} | {row['model_id']}: "
                f"peak={row['peak_drift_mean']:.4f}, "
                f"peak_rel={row['peak_drift_relative_mean']:.6f}, "
                f"end={row['end_drift_mean']:.4f}, "
                f"end_rel={row['end_drift_relative_mean']:.6f}, "
                f"recovery={row['recovery_fraction_mean']:.4f}, "
                f"end/peak={row['end_over_peak']:.4f}, "
                f"crossed={row['crossed_baseline_rate']:.4f}\n"
            )

    print(f"[summary] wrote {run_dir / 'model_summary.csv'}")
    print(f"[summary] wrote {run_dir / 'alpha_summary.csv'}")
    print(f"[summary] wrote {run_dir / 'layer_summary_recomputed.csv'}")
    print(f"[summary] wrote {run_dir / 'summary_generated.md'}")


if __name__ == "__main__":
    main()
