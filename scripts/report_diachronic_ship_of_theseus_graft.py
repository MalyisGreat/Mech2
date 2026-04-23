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


LAYER_ORDER = {
    "early": 0,
    "early_middle": 1,
    "middle": 2,
    "late_middle": 3,
    "late": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a report for the diachronic Ship-of-Theseus identity graft run.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _bootstrap_mean_ci(values: list[float]) -> tuple[float, float]:
    add_src_to_path()
    from identity_stability.identity_analysis import bootstrap_mean_ci

    return bootstrap_mean_ci(values, iters=2000, seed=123)


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if np.isfinite(value)]


def _metric_row(name: str, values: list[float]) -> dict[str, object]:
    finite = _finite(values)
    mean = float(np.mean(finite)) if finite else float("nan")
    ci_low, ci_high = _bootstrap_mean_ci(finite) if finite else (float("nan"), float("nan"))
    return {
        "metric": name,
        "n": len(finite),
        "mean": mean,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def _prompt_level_values(df: pd.DataFrame, group_cols: list[str], value_col: str) -> list[float]:
    if df.empty or value_col not in df.columns:
        return []
    grouped = (
        df.groupby(group_cols, as_index=False)[value_col]
        .mean(numeric_only=True)
        [value_col]
        .dropna()
        .astype(float)
        .tolist()
    )
    return grouped


def _paired_prompt_diffs(
    df: pd.DataFrame,
    *,
    baseline_value: str,
    compare_value: str,
    frame_col: str,
    group_cols: list[str],
    metric_col: str,
) -> list[float]:
    subset = df[df[frame_col].isin([baseline_value, compare_value])].copy()
    if subset.empty or metric_col not in subset.columns:
        return []
    wide = subset.pivot_table(index=group_cols, columns=frame_col, values=metric_col, aggfunc="mean")
    if baseline_value not in wide.columns or compare_value not in wide.columns:
        return []
    return (wide[compare_value] - wide[baseline_value]).dropna().astype(float).tolist()


def _save_heatmap(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    pivot = (
        df.pivot_table(index="layer_label", columns="blend_lambda", values="donor_identity_fraction", aggfunc="mean")
        .reindex(sorted(df["layer_label"].dropna().unique(), key=lambda x: LAYER_ORDER.get(str(x), 999)))
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{float(col):.2f}" for col in pivot.columns], rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("Blend lambda")
    ax.set_ylabel("Layer bucket")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Donor identity fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_prefix_suffix_curves(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    plot_df["layer_order"] = plot_df["layer_label"].map(LAYER_ORDER)
    plot_df = plot_df.dropna(subset=["layer_order"])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, mode_sub in plot_df.groupby("graft_mode", as_index=False):
        grouped = (
            mode_sub.groupby(["layer_order", "layer_label"], as_index=False)["donor_identity_fraction"]
            .mean(numeric_only=True)
            .sort_values("layer_order")
        )
        ax.plot(
            grouped["layer_order"],
            grouped["donor_identity_fraction"],
            marker="o",
            linewidth=2.0,
            label=str(mode),
        )
    ax.set_xticks(list(LAYER_ORDER.values()))
    ax.set_xticklabels(list(LAYER_ORDER.keys()), rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Donor identity fraction")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_self_report_scatter(df: pd.DataFrame, path: Path, title: str) -> None:
    scatter_df = df.dropna(subset=["donor_identity_fraction", "verbal_donor_claim"]).copy()
    if scatter_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        scatter_df["donor_identity_fraction"],
        scatter_df["verbal_donor_claim"],
        alpha=0.35,
        s=16,
        c=scatter_df["blend_lambda"],
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Causal donor identity fraction")
    ax.set_ylabel("Verbal donor-claim score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_control_bars(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    grouped = (
        df.groupby("control_kind", as_index=False)["donor_identity_fraction"]
        .mean(numeric_only=True)
        .sort_values("donor_identity_fraction", ascending=False)
    )
    if grouped.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(grouped["control_kind"], grouped["donor_identity_fraction"], color="#4C78A8")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Donor identity fraction")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    add_src_to_path()
    config = load_yaml_config(args.config)

    output_root = Path(config["output_root"])
    run_dir = output_root / "diachronic_ship_of_theseus_graft"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(run_dir / "results.csv")
    selected_prompts = pd.read_csv(run_dir / "selected_prompts.csv")
    clean_divergence = pd.read_csv(run_dir / "clean_prompt_divergence.csv")
    results["layer_label"] = results["layer_label"].fillna("none")
    results["graft_mode"] = results["graft_mode"].fillna("none")
    results["token_position_label"] = results["token_position_label"].fillna("unknown")

    natural_unit_cols = [
        "pair_name",
        "control_kind",
        "graft_mode",
        "token_position_label",
        "layer_label",
        "blend_lambda",
        "prompt_id",
    ]
    prompt_level = (
        results.groupby(natural_unit_cols, as_index=False)[
            [
                "donor_identity_fraction",
                "text_donor_identity_fraction",
                "semantic_donor_identity_fraction",
                "recovery_fraction",
                "persistence",
                "cad",
                "next_token_kl",
                "activation_norm_deviation",
                "verbal_donor_claim",
            ]
        ]
        .mean(numeric_only=True)
    )

    stats_rows: list[dict[str, object]] = []
    report_lines = [
        "# Diachronic Ship-of-Theseus Identity Graft Report",
        "",
        f"- Config: `{args.config}`",
        f"- Output root: `{output_root}`",
        f"- Model: `{config['model_id']}`",
        f"- Selected prompts: `{len(selected_prompts)}` from `{len(clean_divergence)}` screened prompts",
        f"- Pair directions: `{', '.join(str(pair['pair_name']) for pair in config['pairs'])}`",
        "",
        "## Prompt Selection",
        "",
    ]

    selection_metrics = [
        _metric_row("clean_prompt_js_selected", selected_prompts["clean_js"].astype(float).tolist()),
        _metric_row("clean_prompt_js_all_candidates", clean_divergence["clean_js"].astype(float).tolist()),
    ]
    for metric in selection_metrics:
        stats_rows.append(metric)
        report_lines.append(
            f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
            f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` prompts"
        )

    report_lines.extend(["", "## Headline Results", ""])
    headline_specs = [
        (
            "later_host_young_donor_primary_single_layer_last_prompt_lambda1_dif",
            (
                (prompt_level["pair_name"] == "later_host_young_donor")
                & (prompt_level["control_kind"] == "primary")
                & (prompt_level["graft_mode"] == "single_layer")
                & (prompt_level["token_position_label"] == "last_prompt_token")
                & np.isclose(prompt_level["blend_lambda"], 1.0)
            ),
        ),
        (
            "young_host_later_donor_primary_single_layer_last_prompt_lambda1_dif",
            (
                (prompt_level["pair_name"] == "young_host_later_donor")
                & (prompt_level["control_kind"] == "primary")
                & (prompt_level["graft_mode"] == "single_layer")
                & (prompt_level["token_position_label"] == "last_prompt_token")
                & np.isclose(prompt_level["blend_lambda"], 1.0)
            ),
        ),
        (
            "later_host_young_donor_primary_prefix_last_prompt_lambda1_dif",
            (
                (prompt_level["pair_name"] == "later_host_young_donor")
                & (prompt_level["control_kind"] == "primary")
                & (prompt_level["graft_mode"] == "prefix")
                & (prompt_level["token_position_label"] == "last_prompt_token")
                & np.isclose(prompt_level["blend_lambda"], 1.0)
            ),
        ),
        (
            "later_host_young_donor_primary_suffix_last_prompt_lambda1_dif",
            (
                (prompt_level["pair_name"] == "later_host_young_donor")
                & (prompt_level["control_kind"] == "primary")
                & (prompt_level["graft_mode"] == "suffix")
                & (prompt_level["token_position_label"] == "last_prompt_token")
                & np.isclose(prompt_level["blend_lambda"], 1.0)
            ),
        ),
        (
            "name_only_last_prompt_dif",
            (
                (prompt_level["control_kind"] == "name_only")
                & (prompt_level["token_position_label"] == "last_prompt_token")
            ),
        ),
    ]
    for metric_name, mask in headline_specs:
        metric = _metric_row(metric_name, prompt_level.loc[mask, "donor_identity_fraction"].astype(float).tolist())
        stats_rows.append(metric)
        report_lines.append(
            f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
            f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` prompt-level cells"
        )

    report_lines.extend(["", "## Control Checks", ""])
    control_filter = (
        (prompt_level["pair_name"] == "later_host_young_donor")
        & (prompt_level["graft_mode"] == "single_layer")
        & (prompt_level["token_position_label"] == "last_prompt_token")
        & np.isclose(prompt_level["blend_lambda"], 1.0)
    )
    for control_name in ["primary", "adjacent", "very_early", "random_same_norm", "shuffled_prompt", "name_only"]:
        metric = _metric_row(
            f"control_{control_name}_donor_identity_fraction",
            prompt_level.loc[control_filter & (prompt_level["control_kind"] == control_name), "donor_identity_fraction"]
            .astype(float)
            .tolist(),
        )
        stats_rows.append(metric)
        report_lines.append(
            f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
            f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` prompt-level cells"
        )

    report_lines.extend(["", "## Secondary Metrics", ""])
    secondary_specs = [
        ("primary_cad", "cad"),
        ("primary_recovery_fraction", "recovery_fraction"),
        ("primary_persistence", "persistence"),
        ("primary_next_token_kl", "next_token_kl"),
        ("primary_activation_norm_deviation", "activation_norm_deviation"),
        ("primary_text_donor_identity_fraction", "text_donor_identity_fraction"),
        ("primary_semantic_donor_identity_fraction", "semantic_donor_identity_fraction"),
    ]
    primary_secondary_mask = (
        (prompt_level["control_kind"] == "primary")
        & (prompt_level["graft_mode"] == "single_layer")
        & (prompt_level["token_position_label"] == "last_prompt_token")
        & np.isclose(prompt_level["blend_lambda"], 1.0)
    )
    for metric_name, col in secondary_specs:
        metric = _metric_row(metric_name, prompt_level.loc[primary_secondary_mask, col].astype(float).tolist())
        stats_rows.append(metric)
        report_lines.append(
            f"- `{metric['metric']}`: mean `{metric['mean']:.4f}` "
            f"[{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]` over `n={metric['n']}` prompt-level cells"
        )

    verbal_subset = prompt_level.dropna(subset=["verbal_donor_claim"]).copy()
    verbal_corr = float(
        verbal_subset["donor_identity_fraction"].corr(verbal_subset["verbal_donor_claim"])
    ) if len(verbal_subset) >= 2 else float("nan")
    verbal_metric = _metric_row(
        "self_report_minus_causal_abs_error",
        (
            verbal_subset["verbal_donor_claim"] - verbal_subset["donor_identity_fraction"]
        ).abs().astype(float).tolist(),
    )
    stats_rows.append({"metric": "self_report_causal_correlation", "n": int(len(verbal_subset)), "mean": verbal_corr, "ci95_low": float("nan"), "ci95_high": float("nan")})
    stats_rows.append(verbal_metric)
    report_lines.append(f"- `self_report_causal_correlation`: `{verbal_corr:.4f}` over `n={len(verbal_subset)}` prompt-level cells")
    report_lines.append(
        f"- `{verbal_metric['metric']}`: mean `{verbal_metric['mean']:.4f}` "
        f"[{verbal_metric['ci95_low']:.4f}, {verbal_metric['ci95_high']:.4f}]` over `n={verbal_metric['n']}` prompt-level cells"
    )

    heatmap_last = prompt_level[
        (prompt_level["pair_name"] == "later_host_young_donor")
        & (prompt_level["control_kind"] == "primary")
        & (prompt_level["graft_mode"] == "single_layer")
        & (prompt_level["token_position_label"] == "last_prompt_token")
    ]
    heatmap_first = prompt_level[
        (prompt_level["pair_name"] == "later_host_young_donor")
        & (prompt_level["control_kind"] == "primary")
        & (prompt_level["graft_mode"] == "single_layer")
        & (prompt_level["token_position_label"] == "first_prompt_token")
    ]
    prefix_suffix = prompt_level[
        (prompt_level["pair_name"] == "later_host_young_donor")
        & (prompt_level["control_kind"] == "primary")
        & (prompt_level["token_position_label"] == "last_prompt_token")
        & np.isclose(prompt_level["blend_lambda"], 1.0)
        & (prompt_level["graft_mode"].isin(["prefix", "suffix"]))
    ]
    control_bars = prompt_level[
        (prompt_level["pair_name"] == "later_host_young_donor")
        & (prompt_level["graft_mode"] == "single_layer")
        & (prompt_level["token_position_label"] == "last_prompt_token")
        & np.isclose(prompt_level["blend_lambda"], 1.0)
    ]
    scatter_df = prompt_level[
        (prompt_level["control_kind"] == "primary")
        & (prompt_level["graft_mode"].isin(["single_layer", "prefix", "suffix"]))
    ]

    heatmap_last_path = figures_dir / "donor_identity_heatmap_last_prompt.png"
    heatmap_first_path = figures_dir / "donor_identity_heatmap_first_prompt.png"
    prefix_suffix_path = figures_dir / "prefix_suffix_curves_last_prompt.png"
    control_bars_path = figures_dir / "control_comparison_last_prompt.png"
    scatter_path = figures_dir / "self_report_vs_causal_identity.png"

    _save_heatmap(heatmap_last, heatmap_last_path, "Donor Identity Fraction: later host, last prompt token")
    _save_heatmap(heatmap_first, heatmap_first_path, "Donor Identity Fraction: later host, first prompt token")
    _save_prefix_suffix_curves(prefix_suffix, prefix_suffix_path, "Prefix vs suffix replacement (later host, lambda=1.0)")
    _save_control_bars(control_bars, control_bars_path, "Control comparison (later host, single-layer, lambda=1.0)")
    _save_self_report_scatter(scatter_df, scatter_path, "Self-report vs causal donor identity")

    report_lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- Heatmap, last prompt token: `{heatmap_last_path}`",
            f"- Heatmap, first prompt token: `{heatmap_first_path}`",
            f"- Prefix vs suffix curves: `{prefix_suffix_path}`",
            f"- Control comparison: `{control_bars_path}`",
            f"- Self-report vs causal identity: `{scatter_path}`",
        ]
    )

    (output_root / "diachronic_ship_of_theseus_graft_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_root / "diachronic_ship_of_theseus_graft_stats.json").write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
