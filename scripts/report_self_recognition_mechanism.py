from __future__ import annotations

import argparse
import json
from math import exp, lgamma, log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from identity_battery_common import add_src_to_path, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize self-recognition mechanism results.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def exact_binomial_p_greater_or_equal(k: int, n: int, p: float) -> float:
    if n <= 0:
        return float("nan")
    log_terms: list[float] = []
    for i in range(k, n + 1):
        log_term = (
            lgamma(n + 1)
            - lgamma(i + 1)
            - lgamma(n - i + 1)
            + i * log(p)
            + (n - i) * log(1.0 - p)
        )
        log_terms.append(log_term)
    max_log = max(log_terms)
    total = sum(exp(term - max_log) for term in log_terms)
    return float(min(1.0, exp(max_log) * total))


def bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 123) -> tuple[float, float]:
    add_src_to_path()
    from identity_stability.identity_analysis import bootstrap_mean_ci

    return bootstrap_mean_ci(values, iters=iters, seed=seed)


def holm_adjust(p_values: list[float]) -> list[float]:
    add_src_to_path()
    from identity_stability.identity_analysis import holm_adjust as identity_holm_adjust

    return identity_holm_adjust(p_values)


def _save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summarize_accuracy(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    chance = 1.0 / 3.0
    rows: list[dict[str, object]] = []
    for keys, sub in df.groupby(group_cols, as_index=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {name: value for name, value in zip(group_cols, keys)}
        values = sub["chose_self_baseline"].astype(float).tolist()
        hits = int(sum(values))
        trials = int(len(values))
        mean_accuracy = float(sum(values) / len(values))
        ci_low, ci_high = bootstrap_ci(values)
        row.update(
            {
                "accuracy_mean": mean_accuracy,
                "accuracy_ci95_low": ci_low,
                "accuracy_ci95_high": ci_high,
                "hits": hits,
                "trials": trials,
                "seed_count": int(sub["seed"].nunique()) if "seed" in sub.columns else 1,
                "p_value_vs_chance": exact_binomial_p_greater_or_equal(hits, trials, chance),
                "baseline_vs_contrary_axis_gap_mean": float(sub["baseline_vs_contrary_axis_gap"].mean()),
                "baseline_vs_contrary_style_distance_mean": float(sub["baseline_vs_contrary_style_distance"].mean()),
                "baseline_vs_alt_axis_gap_mean": float(sub["baseline_vs_alt_axis_gap"].mean()),
                "baseline_vs_alt_style_distance_mean": float(sub["baseline_vs_alt_style_distance"].mean()),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_vs_chance_holm"] = holm_adjust(out["p_value_vs_chance"].tolist())
    return out


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    root = Path(config["output_root"]) / "self_recognition_from_foils"
    results_path = root / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    df = pd.read_csv(results_path)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    chance = 1.0 / 3.0

    summary_by_strength = _summarize_accuracy(
        df,
        ["model_size_label", "identity_frame", "strength_magnitude"],
    ).sort_values(
        ["identity_frame", "model_size_label", "strength_magnitude"],
        ascending=[True, True, True],
    )
    summary_by_strength.to_csv(root / "mechanism_summary_by_strength.csv", index=False)

    summary_by_axis_strength = _summarize_accuracy(
        df,
        ["model_size_label", "identity_frame", "axis_name", "strength_magnitude"],
    ).sort_values(
        ["model_size_label", "identity_frame", "axis_name", "strength_magnitude"],
        ascending=[True, True, True, True],
    )
    summary_by_axis_strength.to_csv(root / "mechanism_summary_by_axis_strength.csv", index=False)

    summary_by_prompt_source = _summarize_accuracy(
        df,
        ["model_size_label", "identity_frame", "prompt_source", "strength_magnitude"],
    ).sort_values(
        ["model_size_label", "identity_frame", "prompt_source", "strength_magnitude"],
        ascending=[True, True, True, True],
    )
    summary_by_prompt_source.to_csv(root / "mechanism_summary_by_prompt_source.csv", index=False)

    target_mask = (df["model_size_label"] == "1b") & (df["identity_frame"] == "family_self")
    target_df = df[target_mask].copy()
    if target_df.empty:
        raise RuntimeError("Mechanism report expected 1b/family_self rows but found none.")

    target_strength = summary_by_strength[
        (summary_by_strength["model_size_label"] == "1b")
        & (summary_by_strength["identity_frame"] == "family_self")
    ].copy()

    target_axis = summary_by_axis_strength[
        (summary_by_axis_strength["model_size_label"] == "1b")
        & (summary_by_axis_strength["identity_frame"] == "family_self")
    ].copy()

    target_prompt_source = summary_by_prompt_source[
        (summary_by_prompt_source["model_size_label"] == "1b")
        & (summary_by_prompt_source["identity_frame"] == "family_self")
    ].copy()

    gap_bins = pd.qcut(
        target_df["baseline_vs_contrary_style_distance"],
        q=min(4, target_df["baseline_vs_contrary_style_distance"].nunique()),
        duplicates="drop",
    )
    gap_summary = (
        target_df.assign(style_gap_quartile=gap_bins.astype(str))
        .groupby("style_gap_quartile", as_index=False)
        .agg(
            accuracy_mean=("chose_self_baseline", "mean"),
            baseline_vs_contrary_style_distance_mean=("baseline_vs_contrary_style_distance", "mean"),
            baseline_vs_contrary_axis_gap_mean=("baseline_vs_contrary_axis_gap", "mean"),
            n=("prompt", "count"),
        )
    )
    gap_summary.to_csv(root / "mechanism_target_style_gap_quartiles.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    family_strength = summary_by_strength[summary_by_strength["identity_frame"] == "family_self"]
    for model_size_label, sub in family_strength.groupby("model_size_label"):
        ax.plot(
            sub["strength_magnitude"],
            sub["accuracy_mean"],
            marker="o",
            label=model_size_label,
        )
    ax.axhline(chance, color="black", linestyle="--", linewidth=1.0, label="chance")
    ax.set_xlabel("Contrary steer strength magnitude")
    ax.set_ylabel("Choose-self accuracy")
    ax.set_title("Family-self accuracy vs contrary strength")
    ax.legend()
    _save_plot(fig, fig_dir / "family_self_accuracy_vs_strength.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for axis_name, sub in target_axis.groupby("axis_name"):
        ax.plot(
            sub["strength_magnitude"],
            sub["accuracy_mean"],
            marker="o",
            label=axis_name,
        )
    ax.axhline(chance, color="black", linestyle="--", linewidth=1.0, label="chance")
    ax.set_xlabel("Contrary steer strength magnitude")
    ax.set_ylabel("Choose-self accuracy")
    ax.set_title("1b / family_self by axis")
    ax.legend()
    _save_plot(fig, fig_dir / "target_axis_accuracy_vs_strength.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    source_pivot = target_prompt_source.pivot(
        index="strength_magnitude",
        columns="prompt_source",
        values="accuracy_mean",
    ).fillna(0.0)
    source_pivot.plot(kind="bar", ax=ax)
    ax.axhline(chance, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Contrary steer strength magnitude")
    ax.set_ylabel("Choose-self accuracy")
    ax.set_title("1b / family_self by prompt source")
    _save_plot(fig, fig_dir / "target_prompt_source_accuracy.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        gap_summary["baseline_vs_contrary_style_distance_mean"],
        gap_summary["accuracy_mean"],
        marker="o",
    )
    ax.axhline(chance, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean baseline-vs-contrary style distance")
    ax.set_ylabel("Choose-self accuracy")
    ax.set_title("1b / family_self accuracy vs foil distance")
    _save_plot(fig, fig_dir / "target_style_gap_accuracy.png")

    low_strength = target_strength.sort_values("strength_magnitude").iloc[0]
    high_strength = target_strength.sort_values("strength_magnitude").iloc[-1]
    neighbor_family = summary_by_strength[
        (summary_by_strength["identity_frame"] == "family_self")
        & (summary_by_strength["strength_magnitude"] == 1.0)
    ].sort_values("model_size_label")
    strongest_axis = target_axis.sort_values("accuracy_mean", ascending=False).iloc[0]
    weakest_axis = target_axis.sort_values("accuracy_mean", ascending=True).iloc[0]

    source_lines: list[str] = []
    for row in target_prompt_source.sort_values(["prompt_source", "strength_magnitude"]).itertuples():
        source_lines.append(
            f"- `{row.prompt_source}` at strength `{row.strength_magnitude:.2f}`: accuracy `{row.accuracy_mean:.4f}` "
            f"over `{int(row.trials)}` trials."
        )

    stats = {
        "target_low_strength_accuracy": float(low_strength["accuracy_mean"]),
        "target_high_strength_accuracy": float(high_strength["accuracy_mean"]),
        "target_strength_delta": float(high_strength["accuracy_mean"] - low_strength["accuracy_mean"]),
        "target_best_axis": str(strongest_axis["axis_name"]),
        "target_worst_axis": str(weakest_axis["axis_name"]),
    }
    with (root / "mechanism_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with (root / "mechanism_report.md").open("w", encoding="utf-8") as f:
        f.write("# Self-Recognition Mechanism Report\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Results: `{results_path}`\n")
        f.write(f"- Models: `{config['model_ids']}`\n")
        f.write(f"- Frames: `{config['identity_frames']}`\n")
        f.write(f"- Strength magnitudes: `{config['self_recognition_strengths']}`\n")
        f.write(
            f"- Prompt sources: `self_prediction_bank` plus contrastive prompts = `{bool(config.get('self_recognition_include_axis_seed_prompts', False))}`\n\n"
        )

        f.write("## Target Cell: 1b / family_self\n\n")
        for row in target_strength.sort_values("strength_magnitude").itertuples():
            f.write(
                f"- Strength `{row.strength_magnitude:.2f}`: accuracy `{row.accuracy_mean:.4f}` "
                f"with 95% CI `[{row.accuracy_ci95_low:.4f}, {row.accuracy_ci95_high:.4f}]`, "
                f"`{int(row.hits)}/{int(row.trials)}` hits, Holm-adjusted p `{row.p_value_vs_chance_holm:.6f}`.\n"
            )
        f.write("\n")

        f.write("## Family-Self Neighbor Comparison At Strength 1.0\n\n")
        for row in neighbor_family.itertuples():
            f.write(
                f"- `{row.model_size_label}`: accuracy `{row.accuracy_mean:.4f}` "
                f"with Holm-adjusted p `{row.p_value_vs_chance_holm:.6f}`.\n"
            )
        f.write("\n")

        f.write("## Prompt-Source Generalization For 1b / family_self\n\n")
        for line in source_lines:
            f.write(f"{line}\n")
        f.write("\n")

        f.write("## Axis Structure For 1b / family_self\n\n")
        f.write(
            f"- Strongest axis/strength cell: `{strongest_axis['axis_name']}` at strength `{float(strongest_axis['strength_magnitude']):.2f}` "
            f"with accuracy `{float(strongest_axis['accuracy_mean']):.4f}`.\n"
        )
        f.write(
            f"- Weakest axis/strength cell: `{weakest_axis['axis_name']}` at strength `{float(weakest_axis['strength_magnitude']):.2f}` "
            f"with accuracy `{float(weakest_axis['accuracy_mean']):.4f}`.\n\n"
        )

        f.write("## Interpretation\n\n")
        if float(high_strength["accuracy_mean"]) > float(low_strength["accuracy_mean"]):
            f.write(
                f"- The target effect strengthens as the contrary foil is pushed farther from baseline: accuracy rises from `{float(low_strength['accuracy_mean']):.4f}` at the weakest steer to `{float(high_strength['accuracy_mean']):.4f}` at the strongest steer.\n"
            )
        else:
            f.write(
                f"- The target effect does not simply grow with stronger contrary steering: accuracy changes from `{float(low_strength['accuracy_mean']):.4f}` at the weakest steer to `{float(high_strength['accuracy_mean']):.4f}` at the strongest steer.\n"
            )
        f.write(
            "- The key question is whether `1b / family_self` remains distinct from neighboring family-self cells and whether it survives the expanded prompt pool rather than only the original prompt bank.\n"
        )
        f.write(
            "- If the effect remains local to `1b / family_self`, the right interpretation is still a narrow, frame-sensitive answer-ownership pocket rather than a general self-model.\n"
        )


if __name__ == "__main__":
    main()
