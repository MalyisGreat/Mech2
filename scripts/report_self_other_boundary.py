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
    parser = argparse.ArgumentParser(description="Summarize self/other boundary results.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Apply config.smoke overrides if present.")
    return parser.parse_args()


def _apply_mode_overrides(config: dict, *, smoke: bool) -> dict:
    resolved = dict(config)
    if smoke and isinstance(config.get("smoke"), dict):
        resolved.update(dict(config["smoke"]))
    return resolved


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


def _safe_holm_adjust(p_values: list[float]) -> list[float]:
    finite_pairs = [(idx, val) for idx, val in enumerate(p_values) if np.isfinite(val)]
    adjusted = [float("nan")] * len(p_values)
    if not finite_pairs:
        return adjusted
    finite_adjusted = holm_adjust([float(val) for _, val in finite_pairs])
    for (idx, _), value in zip(finite_pairs, finite_adjusted):
        adjusted[idx] = value
    return adjusted


def _cluster_stats(
    sub: pd.DataFrame,
    *,
    cluster_cols: list[str],
    value_col: str,
    chance: float,
) -> dict[str, float | int]:
    add_src_to_path()
    from identity_stability.identity_analysis import cluster_bootstrap_mean_ci, cluster_mean_values

    cluster_values = cluster_mean_values(sub, cluster_cols=cluster_cols, value_col=value_col)
    cluster_values = cluster_values[np.isfinite(cluster_values)]
    if cluster_values.size == 0:
        return {
            "cluster_mean": float("nan"),
            "cluster_ci95_low": float("nan"),
            "cluster_ci95_high": float("nan"),
            "cluster_count": 0,
            "cluster_hits_above_chance": 0,
            "cluster_non_tie_count": 0,
            "cluster_tie_count": 0,
            "cluster_sign_p_value_vs_chance": float("nan"),
        }
    ci_low, ci_high = cluster_bootstrap_mean_ci(
        sub,
        cluster_cols=cluster_cols,
        value_col=value_col,
        iters=2000,
        seed=123,
    )
    non_ties = cluster_values[np.abs(cluster_values - chance) > 1e-12]
    hits = int((non_ties > chance).sum())
    trials = int(non_ties.size)
    ties = int(cluster_values.size - trials)
    sign_p = exact_binomial_p_greater_or_equal(hits, trials, 0.5) if trials else float("nan")
    return {
        "cluster_mean": float(cluster_values.mean()),
        "cluster_ci95_low": float(ci_low),
        "cluster_ci95_high": float(ci_high),
        "cluster_count": int(cluster_values.size),
        "cluster_hits_above_chance": hits,
        "cluster_non_tie_count": trials,
        "cluster_tie_count": ties,
        "cluster_sign_p_value_vs_chance": float(sign_p),
    }


def _save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summarize_boundary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    chance = 1.0 / 3.0
    rows: list[dict[str, object]] = []
    for keys, sub in df.groupby(group_cols, as_index=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {name: value for name, value in zip(group_cols, keys)}
        no_steer_values = sub["boundary_order_match_no_steer"].dropna().astype(float).tolist()
        steer_values = sub["boundary_order_match_steer"].dropna().astype(float).tolist()
        no_steer_mean = float(sum(no_steer_values) / len(no_steer_values)) if no_steer_values else float("nan")
        steer_mean = float(sum(steer_values) / len(steer_values)) if steer_values else float("nan")
        no_steer_ci = bootstrap_ci(no_steer_values) if no_steer_values else (float("nan"), float("nan"))
        steer_ci = bootstrap_ci(steer_values) if steer_values else (float("nan"), float("nan"))
        available_group_cols = set(group_cols)
        if {"model_size_label", "identity_frame", "axis_name"}.issubset(available_group_cols):
            no_steer_cluster_cols = ["prompt_index"]
        elif {"model_size_label", "identity_frame"}.issubset(available_group_cols):
            no_steer_cluster_cols = ["axis_name", "prompt_index"]
        else:
            no_steer_cluster_cols = ["model_size_label", "identity_frame", "axis_name", "prompt_index"]
        no_steer_cluster = _cluster_stats(
            sub,
            cluster_cols=no_steer_cluster_cols,
            value_col="boundary_order_match_no_steer",
            chance=chance,
        )
        steer_cluster = _cluster_stats(
            sub,
            cluster_cols=no_steer_cluster_cols,
            value_col="boundary_order_match_steer",
            chance=chance,
        )

        row.update(
            {
                "boundary_match_no_steer_mean": no_steer_mean,
                "boundary_match_no_steer_ci95_low": no_steer_ci[0],
                "boundary_match_no_steer_ci95_high": no_steer_ci[1],
                "boundary_match_no_steer_cluster_mean": no_steer_cluster["cluster_mean"],
                "boundary_match_no_steer_cluster_ci95_low": no_steer_cluster["cluster_ci95_low"],
                "boundary_match_no_steer_cluster_ci95_high": no_steer_cluster["cluster_ci95_high"],
                "boundary_match_no_steer_cluster_count": no_steer_cluster["cluster_count"],
                "boundary_match_no_steer_cluster_non_tie_count": no_steer_cluster["cluster_non_tie_count"],
                "boundary_match_no_steer_cluster_tie_count": no_steer_cluster["cluster_tie_count"],
                "boundary_match_no_steer_cluster_hits_above_chance": no_steer_cluster["cluster_hits_above_chance"],
                "boundary_match_no_steer_cluster_sign_p_value_vs_chance": no_steer_cluster["cluster_sign_p_value_vs_chance"],
                "boundary_match_steer_mean": steer_mean,
                "boundary_match_steer_ci95_low": steer_ci[0],
                "boundary_match_steer_ci95_high": steer_ci[1],
                "boundary_match_steer_cluster_mean": steer_cluster["cluster_mean"],
                "boundary_match_steer_cluster_ci95_low": steer_cluster["cluster_ci95_low"],
                "boundary_match_steer_cluster_ci95_high": steer_cluster["cluster_ci95_high"],
                "boundary_match_steer_cluster_count": steer_cluster["cluster_count"],
                "boundary_match_steer_cluster_non_tie_count": steer_cluster["cluster_non_tie_count"],
                "boundary_match_steer_cluster_tie_count": steer_cluster["cluster_tie_count"],
                "boundary_match_steer_cluster_hits_above_chance": steer_cluster["cluster_hits_above_chance"],
                "boundary_match_steer_cluster_sign_p_value_vs_chance": steer_cluster["cluster_sign_p_value_vs_chance"],
                "boundary_transfer_delta": float(steer_mean - no_steer_mean)
                if pd.notna(no_steer_mean) and pd.notna(steer_mean)
                else float("nan"),
                "self_prediction_advantage_no_steer_mean": float(sub["self_prediction_advantage_no_steer"].mean()),
                "self_prediction_advantage_steer_mean": float(sub["self_prediction_advantage_steer"].mean()),
                "other_prediction_advantage_mean": float(sub["other_prediction_advantage"].mean()),
                "self_moved_toward_other_actual_mean": float(sub["self_moved_toward_other_actual"].mean()),
                "boundary_margin_delta_mean": float(sub["boundary_margin_delta"].mean()),
                "predicted_boundary_confidence_mean": float(sub["predicted_boundary_confidence"].mean()),
                "seed_count": int(sub["seed"].nunique()) if "seed" in sub.columns else 1,
                "n": int(len(sub)),
                "valid_n_no_steer": int(len(no_steer_values)),
                "valid_n_steer": int(len(steer_values)),
                "p_value_no_steer_vs_chance": exact_binomial_p_greater_or_equal(int(sum(no_steer_values)), len(no_steer_values), chance)
                if no_steer_values
                else float("nan"),
                "p_value_steer_vs_chance": exact_binomial_p_greater_or_equal(int(sum(steer_values)), len(steer_values), chance)
                if steer_values
                else float("nan"),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_no_steer_holm"] = _safe_holm_adjust(out["p_value_no_steer_vs_chance"].tolist())
        out["p_value_steer_holm"] = _safe_holm_adjust(out["p_value_steer_vs_chance"].tolist())
        out["boundary_match_no_steer_cluster_sign_p_value_vs_chance_holm"] = _safe_holm_adjust(
            out["boundary_match_no_steer_cluster_sign_p_value_vs_chance"].tolist()
        )
        out["boundary_match_steer_cluster_sign_p_value_vs_chance_holm"] = _safe_holm_adjust(
            out["boundary_match_steer_cluster_sign_p_value_vs_chance"].tolist()
        )
    return out


def main() -> None:
    args = parse_args()
    config = _apply_mode_overrides(load_yaml_config(args.config), smoke=args.smoke)
    root = Path(config["output_root"]) / "self_other_boundary"
    results_path = root / "results.csv"
    partial_mode = False
    if not results_path.exists():
        partial_path = root / "results.partial.csv"
        if not partial_path.exists():
            raise FileNotFoundError(f"Missing results file: {results_path}")
        results_path = partial_path
        partial_mode = True

    df = pd.read_csv(results_path)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_by_model_frame = _summarize_boundary(df, ["model_size_label", "identity_frame"]).sort_values(
        ["model_size_label", "identity_frame"]
    )
    summary_by_model_frame.to_csv(root / "confirm_summary_by_model_frame.csv", index=False)

    summary_by_cell = _summarize_boundary(df, ["model_size_label", "identity_frame", "axis_name"]).sort_values(
        ["model_size_label", "identity_frame", "axis_name"]
    )
    summary_by_cell.to_csv(root / "confirm_summary_by_cell.csv", index=False)

    overall = _summarize_boundary(df, ["model_family"])
    overall_row = overall.iloc[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{row.model_size_label}\n{row.identity_frame}" for row in summary_by_model_frame.itertuples()]
    xs = range(len(labels))
    width = 0.38
    ax.bar([x - width / 2 for x in xs], summary_by_model_frame["boundary_match_no_steer_mean"], width=width, label="no_steer")
    ax.bar([x + width / 2 for x in xs], summary_by_model_frame["boundary_match_steer_mean"], width=width, label="steered_self")
    ax.axhline(1.0 / 3.0, color="black", linestyle="--", linewidth=1.0, label="chance")
    ax.set_xticks(list(xs), labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Boundary match rate")
    ax.set_title("Self/Other Boundary Match")
    ax.legend()
    _save_plot(fig, fig_dir / "boundary_match_by_model_frame.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, summary_by_model_frame["boundary_transfer_delta"])
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Steered minus no-steer boundary match")
    ax.set_title("Boundary Transfer Delta")
    _save_plot(fig, fig_dir / "boundary_transfer_delta_by_model_frame.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([x - width / 2 for x in xs], summary_by_model_frame["self_prediction_advantage_no_steer_mean"], width=width, label="no_steer")
    ax.bar([x + width / 2 for x in xs], summary_by_model_frame["self_prediction_advantage_steer_mean"], width=width, label="steered_self")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Self-prediction advantage")
    ax.set_title("Self Prediction Advantage vs Steering")
    ax.legend()
    _save_plot(fig, fig_dir / "self_prediction_advantage_by_model_frame.png")

    family_rows = summary_by_model_frame[summary_by_model_frame["identity_frame"] == "family_self"].copy()
    target_mask = (
        (summary_by_model_frame["model_size_label"] == "1b")
        & (summary_by_model_frame["identity_frame"] == "family_self")
    )
    target_row = summary_by_model_frame[target_mask].iloc[0] if target_mask.any() else None
    best_no_steer = summary_by_model_frame.sort_values("boundary_match_no_steer_mean", ascending=False).iloc[0]
    strongest_transfer_drop = summary_by_model_frame.sort_values("boundary_transfer_delta").iloc[0]

    stats = {
        "overall_boundary_match_no_steer_mean": float(overall_row["boundary_match_no_steer_mean"]),
        "overall_boundary_match_steer_mean": float(overall_row["boundary_match_steer_mean"]),
        "overall_boundary_transfer_delta": float(overall_row["boundary_transfer_delta"]),
        "best_no_steer_model_frame": {
            "model_size_label": str(best_no_steer["model_size_label"]),
            "identity_frame": str(best_no_steer["identity_frame"]),
            "boundary_match_no_steer_mean": float(best_no_steer["boundary_match_no_steer_mean"]),
        },
        "strongest_transfer_drop_model_frame": {
            "model_size_label": str(strongest_transfer_drop["model_size_label"]),
            "identity_frame": str(strongest_transfer_drop["identity_frame"]),
            "boundary_transfer_delta": float(strongest_transfer_drop["boundary_transfer_delta"]),
        },
    }
    if target_row is not None:
        stats["target_1b_family_self"] = {
            "boundary_match_no_steer_mean": float(target_row["boundary_match_no_steer_mean"]),
            "boundary_match_no_steer_cluster_mean": float(target_row["boundary_match_no_steer_cluster_mean"]),
            "boundary_match_steer_mean": float(target_row["boundary_match_steer_mean"]),
            "boundary_match_steer_cluster_mean": float(target_row["boundary_match_steer_cluster_mean"]),
            "boundary_transfer_delta": float(target_row["boundary_transfer_delta"]),
            "p_value_no_steer_vs_chance": float(target_row["p_value_no_steer_vs_chance"]),
            "p_value_no_steer_holm": float(target_row["p_value_no_steer_holm"]),
            "boundary_match_no_steer_cluster_sign_p_value_vs_chance": float(
                target_row["boundary_match_no_steer_cluster_sign_p_value_vs_chance"]
            ),
            "boundary_match_no_steer_cluster_sign_p_value_vs_chance_holm": float(
                target_row["boundary_match_no_steer_cluster_sign_p_value_vs_chance_holm"]
            ),
        }
    with (root / "confirm_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    family_lines: list[str] = []
    for row in family_rows.itertuples():
        family_lines.append(
            f"- `{row.model_size_label}`: no-steer `{row.boundary_match_no_steer_mean:.4f}`, "
            f"steered `{row.boundary_match_steer_mean:.4f}`, delta `{row.boundary_transfer_delta:.4f}`."
        )

    target_lines: list[str] = []
    if target_row is not None:
        target_lines.extend(
            [
                f"- `1b / family_self` no-steer boundary match: `{target_row['boundary_match_no_steer_mean']:.4f}` "
                f"with 95% CI `[{target_row['boundary_match_no_steer_ci95_low']:.4f}, {target_row['boundary_match_no_steer_ci95_high']:.4f}]`.",
                f"- `1b / family_self` clustered no-steer boundary match: `{target_row['boundary_match_no_steer_cluster_mean']:.4f}` "
                f"with clustered 95% CI `[{target_row['boundary_match_no_steer_cluster_ci95_low']:.4f}, {target_row['boundary_match_no_steer_cluster_ci95_high']:.4f}]`.",
                f"- `1b / family_self` steered boundary match: `{target_row['boundary_match_steer_mean']:.4f}`.",
                f"- `1b / family_self` clustered sign-test p-value vs chance: `{target_row['boundary_match_no_steer_cluster_sign_p_value_vs_chance']:.6f}`.",
                f"- `1b / family_self` Holm-adjusted clustered sign-test p-value vs chance: "
                f"`{target_row['boundary_match_no_steer_cluster_sign_p_value_vs_chance_holm']:.6f}`.",
            ]
        )

    with (root / "confirm_report.md").open("w", encoding="utf-8") as f:
        f.write("# Self/Other Boundary Confirm Report\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Smoke mode: `{args.smoke}`\n")
        f.write(f"- Results: `{results_path}`\n")
        f.write(f"- Partial mode: `{partial_mode}`\n")
        f.write(f"- Models: `{config['model_ids']}`\n")
        f.write(f"- Frames: `{config['identity_frames']}`\n")
        f.write(f"- Seeds: `{config.get('seeds', [config.get('seed', 7)])}`\n")
        f.write(f"- Other-frame map: `{config.get('self_other_other_frame_map', {})}`\n\n")
        f.write("## Main Result\n\n")
        f.write(
            f"- Overall no-steer boundary match: `{overall_row['boundary_match_no_steer_mean']:.4f}` against `0.3333` chance.\n"
        )
        f.write(f"- Overall steered boundary match: `{overall_row['boundary_match_steer_mean']:.4f}`.\n")
        f.write(f"- Overall transfer delta (steered minus no-steer): `{overall_row['boundary_transfer_delta']:.4f}`.\n")
        f.write(
            f"- Overall self-moved-toward-other rate under steering: `{overall_row['self_moved_toward_other_actual_mean']:.4f}`.\n\n"
        )
        f.write("## Best Cell\n\n")
        f.write(
            f"- Strongest no-steer cell: `{best_no_steer['model_size_label']} / {best_no_steer['identity_frame']}` "
            f"at `{best_no_steer['boundary_match_no_steer_mean']:.4f}`.\n"
        )
        if target_lines:
            for line in target_lines:
                f.write(f"{line}\n")
        f.write("\n## Family-Self Pattern\n\n")
        for line in family_lines:
            f.write(f"{line}\n")
        f.write("\n## Transfer / Collapse Pattern\n\n")
        f.write(
            f"- Largest boundary drop under steering: `{strongest_transfer_drop['model_size_label']} / {strongest_transfer_drop['identity_frame']}` "
            f"with delta `{strongest_transfer_drop['boundary_transfer_delta']:.4f}`.\n"
        )
        f.write(
            "- Negative deltas mean the model's predicted self/other ordering became less aligned with the actual ordering after contrary steering.\n"
        )
        f.write("\n## Interpretation\n\n")
        f.write(
            "- If this report shows above-chance no-steer boundary match concentrated in a narrow model/frame pocket and weak elsewhere, that supports local self/other boundary knowledge rather than a broad self-model.\n"
        )
        f.write(
            "- If steering pushes self outputs toward the other-frame distribution while boundary match falls, that supports fragile self/other separation under pressure rather than robust identity coherence.\n"
        )


if __name__ == "__main__":
    main()
