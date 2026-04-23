from __future__ import annotations

import argparse
import json
from math import exp, lgamma, log
from pathlib import Path

import numpy as np
import pandas as pd

from identity_battery_common import add_src_to_path, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize confirmatory self-recognition results.")
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
            "cluster_accuracy_mean": float("nan"),
            "cluster_accuracy_ci95_low": float("nan"),
            "cluster_accuracy_ci95_high": float("nan"),
            "cluster_count": 0,
            "cluster_hits_above_chance": 0,
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
        "cluster_accuracy_mean": float(cluster_values.mean()),
        "cluster_accuracy_ci95_low": float(ci_low),
        "cluster_accuracy_ci95_high": float(ci_high),
        "cluster_count": int(cluster_values.size),
        "cluster_hits_above_chance": hits,
        "cluster_non_tie_count": trials,
        "cluster_tie_count": ties,
        "cluster_sign_p_value_vs_chance": float(sign_p),
    }


def _extract_candidate_label(candidates_json: object, candidate_type: str) -> str:
    try:
        for candidate in json.loads(str(candidates_json)):
            if str(candidate.get("type")) == candidate_type:
                return str(candidate.get("label", ""))
    except Exception:
        return ""
    return ""


def _ensure_label_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "self_baseline_label" not in work.columns and "candidates_json" in work.columns:
        work["self_baseline_label"] = work["candidates_json"].apply(
            lambda value: _extract_candidate_label(value, "self_baseline")
        )
    if "contrary_steer_label" not in work.columns and "candidates_json" in work.columns:
        work["contrary_steer_label"] = work["candidates_json"].apply(
            lambda value: _extract_candidate_label(value, "contrary_steer")
        )
    if "alt_frame_label" not in work.columns and "candidates_json" in work.columns:
        work["alt_frame_label"] = work["candidates_json"].apply(
            lambda value: _extract_candidate_label(value, "alt_frame")
        )
    if "choice_permutation_id" not in work.columns:
        work["choice_permutation_id"] = "legacy_shuffle"
    return work


def _normalized_entropy(counts: list[int]) -> float:
    total = int(sum(counts))
    if total <= 0:
        return float("nan")
    probs = [count / total for count in counts if count > 0]
    if not probs:
        return float("nan")
    return float(-sum(prob * log(prob) for prob in probs) / log(len(counts)))


def _label_bias_audit(df: pd.DataFrame, *, group_cols: list[str], value_col: str) -> pd.DataFrame:
    required = {"selected_label", "self_baseline_label", value_col}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    labels = ["1", "2", "3"]
    rows: list[dict[str, object]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(group_cols, keys))
        n = int(len(sub))
        row["n"] = n
        row["accuracy_mean"] = float(pd.to_numeric(sub[value_col], errors="coerce").mean())

        selected = sub["selected_label"].astype(str)
        selected_counts = [int((selected == label).sum()) for label in labels]
        for label, count in zip(labels, selected_counts):
            row[f"selected_label_{label}_rate"] = float(count / n) if n else float("nan")
        row["selected_label_max_rate"] = float(max(selected_counts) / n) if n else float("nan")
        row["selected_label_entropy_norm"] = _normalized_entropy(selected_counts)

        self_label_accuracies: list[float] = []
        for label in labels:
            label_sub = sub[sub["self_baseline_label"].astype(str) == label]
            label_accuracy = float(pd.to_numeric(label_sub[value_col], errors="coerce").mean()) if len(label_sub) else float("nan")
            row[f"self_label_{label}_accuracy"] = label_accuracy
            if np.isfinite(label_accuracy):
                self_label_accuracies.append(label_accuracy)
        row["self_label_accuracy_spread"] = (
            float(max(self_label_accuracies) - min(self_label_accuracies))
            if self_label_accuracies
            else float("nan")
        )

        if "selection_confidence" in sub.columns:
            confidence = pd.to_numeric(sub["selection_confidence"], errors="coerce")
            row["selection_confidence_mean"] = float(confidence.mean())
            row["selection_confidence_median"] = float(confidence.median())
            row["selection_confidence_p10"] = float(confidence.quantile(0.10))
            row["selection_confidence_p90"] = float(confidence.quantile(0.90))
            row["low_confidence_rate_lt_0_5"] = float((confidence < 0.5).mean())
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    add_src_to_path()
    args = parse_args()
    config = load_yaml_config(args.config)
    root = Path(config["output_root"]) / "self_recognition_from_foils"
    results_path = root / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    df = _ensure_label_audit_columns(pd.read_csv(results_path))
    chance = 1.0 / 3.0
    value_col = "chose_self_baseline"
    by_cell_cluster_cols = ["prompt_index", "prompt_source", "strength_magnitude"]
    by_model_frame_cluster_cols = ["axis_name", "prompt_index", "prompt_source", "strength_magnitude"]
    overall_cluster_cols = ["model_size_label", "identity_frame", "axis_name", "prompt_index", "prompt_source", "strength_magnitude"]

    by_cell_rows: list[dict[str, object]] = []
    for keys, sub in df.groupby(["model_size_label", "identity_frame", "axis_name"], as_index=False):
        model_size_label, identity_frame, axis_name = keys
        values = sub[value_col].astype(float).tolist()
        mean_accuracy = float(sum(values) / len(values))
        ci_low, ci_high = bootstrap_ci(values)
        hits = int(sum(values))
        trials = int(len(values))
        p_value = exact_binomial_p_greater_or_equal(hits, trials, chance)
        cluster = _cluster_stats(
            sub,
            cluster_cols=by_cell_cluster_cols,
            value_col=value_col,
            chance=chance,
        )
        by_cell_rows.append(
            {
                "model_size_label": model_size_label,
                "identity_frame": identity_frame,
                "axis_name": axis_name,
                "accuracy_mean": mean_accuracy,
                "accuracy_ci95_low": ci_low,
                "accuracy_ci95_high": ci_high,
                "hits": hits,
                "trials": trials,
                "seed_count": int(sub["seed"].nunique()) if "seed" in sub.columns else 1,
                "p_value_vs_chance": p_value,
                "row_p_value_vs_chance": p_value,
                **cluster,
                "baseline_vs_contrary_axis_gap_mean": float(sub["baseline_vs_contrary_axis_gap"].mean()),
                "baseline_vs_alt_axis_gap_mean": float(sub["baseline_vs_alt_axis_gap"].mean()),
            }
        )

    by_cell = pd.DataFrame(by_cell_rows).sort_values(
        ["accuracy_mean", "model_size_label", "identity_frame", "axis_name"],
        ascending=[False, True, True, True],
    )
    by_cell["p_value_vs_chance_holm"] = _safe_holm_adjust(by_cell["p_value_vs_chance"].tolist())
    by_cell["cluster_sign_p_value_vs_chance_holm"] = _safe_holm_adjust(
        by_cell["cluster_sign_p_value_vs_chance"].tolist()
    )
    by_cell.to_csv(root / "confirm_summary_by_cell.csv", index=False)

    by_model_frame_rows: list[dict[str, object]] = []
    for keys, sub in df.groupby(["model_size_label", "identity_frame"], as_index=False):
        model_size_label, identity_frame = keys
        values = sub[value_col].astype(float).tolist()
        mean_accuracy = float(sum(values) / len(values))
        ci_low, ci_high = bootstrap_ci(values)
        hits = int(sum(values))
        trials = int(len(values))
        p_value = exact_binomial_p_greater_or_equal(hits, trials, chance)
        cluster = _cluster_stats(
            sub,
            cluster_cols=by_model_frame_cluster_cols,
            value_col=value_col,
            chance=chance,
        )
        by_model_frame_rows.append(
            {
                "model_size_label": model_size_label,
                "identity_frame": identity_frame,
                "accuracy_mean": mean_accuracy,
                "accuracy_ci95_low": ci_low,
                "accuracy_ci95_high": ci_high,
                "hits": hits,
                "trials": trials,
                "seed_count": int(sub["seed"].nunique()) if "seed" in sub.columns else 1,
                "p_value_vs_chance": p_value,
                "row_p_value_vs_chance": p_value,
                **cluster,
            }
        )

    by_model_frame = pd.DataFrame(by_model_frame_rows).sort_values(
        ["accuracy_mean", "model_size_label", "identity_frame"],
        ascending=[False, True, True],
    )
    by_model_frame["p_value_vs_chance_holm"] = _safe_holm_adjust(by_model_frame["p_value_vs_chance"].tolist())
    by_model_frame["cluster_sign_p_value_vs_chance_holm"] = _safe_holm_adjust(
        by_model_frame["cluster_sign_p_value_vs_chance"].tolist()
    )
    by_model_frame.to_csv(root / "confirm_summary_by_model_frame.csv", index=False)

    if {"model_size_label", "identity_frame", "selected_label"}.issubset(df.columns):
        label_counts = (
            df.groupby(["model_size_label", "identity_frame", "selected_label"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        label_counts.to_csv(root / "confirm_selected_label_counts.csv", index=False)
    if {"model_size_label", "identity_frame", "self_baseline_label"}.issubset(df.columns):
        accuracy_by_self_label = (
            df.groupby(["model_size_label", "identity_frame", "self_baseline_label"], dropna=False)
            .agg(
                accuracy_mean=(value_col, "mean"),
                n=(value_col, "count"),
            )
            .reset_index()
        )
        accuracy_by_self_label.to_csv(root / "confirm_accuracy_by_self_label.csv", index=False)

    label_bias_by_model_frame = _label_bias_audit(
        df,
        group_cols=["model_size_label", "identity_frame"],
        value_col=value_col,
    )
    if not label_bias_by_model_frame.empty:
        label_bias_by_model_frame = label_bias_by_model_frame.sort_values(
            ["selected_label_max_rate", "self_label_accuracy_spread", "model_size_label", "identity_frame"],
            ascending=[False, False, True, True],
        )
        label_bias_by_model_frame.to_csv(root / "confirm_label_bias_audit_by_model_frame.csv", index=False)

    label_bias_by_cell = _label_bias_audit(
        df,
        group_cols=["model_size_label", "identity_frame", "axis_name"],
        value_col=value_col,
    )
    if not label_bias_by_cell.empty:
        label_bias_by_cell = label_bias_by_cell.sort_values(
            ["selected_label_max_rate", "self_label_accuracy_spread", "model_size_label", "identity_frame", "axis_name"],
            ascending=[False, False, True, True, True],
        )
        label_bias_by_cell.to_csv(root / "confirm_label_bias_audit_by_cell.csv", index=False)

    overall_values = df[value_col].astype(float).tolist()
    overall_mean = float(sum(overall_values) / len(overall_values))
    overall_low, overall_high = bootstrap_ci(overall_values)
    overall_hits = int(sum(overall_values))
    overall_trials = int(len(overall_values))
    overall_p = exact_binomial_p_greater_or_equal(overall_hits, overall_trials, chance)
    overall_cluster = _cluster_stats(
        df,
        cluster_cols=overall_cluster_cols,
        value_col=value_col,
        chance=chance,
    )

    top_row = by_model_frame.iloc[0] if not by_model_frame.empty else None
    target_row = by_model_frame[
        (by_model_frame["model_size_label"] == "1b") & (by_model_frame["identity_frame"] == "family_self")
    ]
    target = target_row.iloc[0] if not target_row.empty else None

    stats = {
        "overall_accuracy_mean": overall_mean,
        "overall_accuracy_ci95_low": overall_low,
        "overall_accuracy_ci95_high": overall_high,
        "overall_hits": overall_hits,
        "overall_trials": overall_trials,
        "overall_p_value_vs_chance": overall_p,
        **{f"overall_{key}": value for key, value in overall_cluster.items()},
    }
    with (root / "confirm_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with (root / "confirm_report.md").open("w", encoding="utf-8") as f:
        f.write("# Self-Recognition Confirmatory Report\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Results: `{results_path}`\n")
        f.write(f"- Seeds: `{sorted(df['seed'].unique().tolist()) if 'seed' in df.columns else ['single-seed']}`\n")
        f.write(f"- Overall row accuracy: `{overall_mean:.4f}` with row-bootstrap 95% CI `[{overall_low:.4f}, {overall_high:.4f}]`\n")
        f.write(
            f"- Overall clustered accuracy: `{float(overall_cluster['cluster_accuracy_mean']):.4f}` "
            f"with cluster-bootstrap 95% CI `[{float(overall_cluster['cluster_accuracy_ci95_low']):.4f}, "
            f"{float(overall_cluster['cluster_accuracy_ci95_high']):.4f}]` over "
            f"`{int(overall_cluster['cluster_count'])}` prompt/axis/condition clusters.\n"
        )
        f.write(f"- Chance level: `{chance:.4f}`\n")
        f.write(f"- Overall row-level one-sided exact binomial p-value vs chance: `{overall_p:.6f}`\n")
        f.write(
            f"- Overall cluster sign-test p-value vs chance: "
            f"`{float(overall_cluster['cluster_sign_p_value_vs_chance']):.6f}` "
            f"using `{int(overall_cluster['cluster_hits_above_chance'])}/"
            f"{int(overall_cluster['cluster_non_tie_count'])}` non-tie clusters above chance "
            f"and `{int(overall_cluster['cluster_tie_count'])}` exact chance ties.\n\n"
        )

        if target is not None:
            f.write("## Target Cell\n\n")
            f.write(
                f"- `1b / family_self`: accuracy `{float(target['accuracy_mean']):.4f}` "
                f"with row-bootstrap 95% CI `[{float(target['accuracy_ci95_low']):.4f}, {float(target['accuracy_ci95_high']):.4f}]`, "
                f"`{int(target['hits'])}/{int(target['trials'])}` hits, "
                f"row p-value `{float(target['p_value_vs_chance']):.6f}`, "
                f"row Holm-adjusted `{float(target['p_value_vs_chance_holm']):.6f}`.\n"
            )
            f.write(
                f"- Clustered target estimate: `{float(target['cluster_accuracy_mean']):.4f}` "
                f"with cluster-bootstrap 95% CI "
                f"`[{float(target['cluster_accuracy_ci95_low']):.4f}, {float(target['cluster_accuracy_ci95_high']):.4f}]`, "
                f"`{int(target['cluster_hits_above_chance'])}/{int(target['cluster_non_tie_count'])}` non-tie clusters above chance, "
                f"`{int(target['cluster_tie_count'])}` exact chance ties, "
                f"cluster sign-test p-value `{float(target['cluster_sign_p_value_vs_chance']):.6f}`, "
                f"Holm-adjusted `{float(target['cluster_sign_p_value_vs_chance_holm']):.6f}`.\n\n"
            )

        if top_row is not None:
            f.write("## Strongest Cell\n\n")
            f.write(
                f"- `{top_row['model_size_label']} / {top_row['identity_frame']}` is currently highest at "
                f"`{float(top_row['accuracy_mean']):.4f}` over `{int(top_row['trials'])}` trials.\n\n"
            )

        if not label_bias_by_model_frame.empty:
            audit_top = label_bias_by_model_frame.iloc[0]
            f.write("## Label-Bias Audit\n\n")
            f.write(
                "- Additional audit tables were written to "
                "`confirm_label_bias_audit_by_model_frame.csv` and "
                "`confirm_label_bias_audit_by_cell.csv`.\n"
            )
            f.write(
                f"- Strongest selected-label skew: `{audit_top['model_size_label']} / {audit_top['identity_frame']}` "
                f"selected one digit label at rate `{float(audit_top['selected_label_max_rate']):.4f}` "
                f"with normalized label entropy `{float(audit_top['selected_label_entropy_norm']):.4f}`.\n"
            )
            f.write(
                f"- In that same cell, accuracy spread across self-answer label positions was "
                f"`{float(audit_top['self_label_accuracy_spread']):.4f}`. Large spreads indicate answer-position bias, "
                "even when candidate placement is balanced by construction.\n\n"
            )

        f.write("## Interpretation\n\n")
        f.write(
            "- Primary inference is now clustered by prompt/axis/strength cells. Row-level binomial tests are retained only as diagnostics because repeated seeds and choice-label permutations are not independent observations.\n"
        )
        if target is not None and float(target["cluster_accuracy_mean"]) > chance and float(target["cluster_sign_p_value_vs_chance_holm"]) < 0.05:
            f.write(
                "- The earlier `1b / family_self` self-recognition bump survives the clustered confirmatory test and remains significant after Holm correction across model/frame cells, so it now looks like a real local effect rather than pure small-sample noise.\n"
            )
        else:
            f.write(
                "- The `1b / family_self` self-recognition bump is the strongest model/frame cell, but it does not clear the multiplicity-corrected clustered confirmatory threshold across the full balanced grid. The safer interpretation is a local answer-ownership pocket, not robust self-model coherence.\n"
            )
        f.write(
            "- Even if one cell remains above chance, the paper should still treat answer ownership as unstable unless the effect generalizes across nearby models, frames, and axes.\n"
        )


if __name__ == "__main__":
    main()
