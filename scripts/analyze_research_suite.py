from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


MODEL_PARAMS = {
    "EleutherAI/pythia-70m": 70e6,
    "EleutherAI/pythia-160m": 160e6,
    "EleutherAI/pythia-410m": 410e6,
    "EleutherAI/pythia-1b": 1e9,
    "EleutherAI/pythia-1.4b": 1.4e9,
    "EleutherAI/pythia-2.8b": 2.8e9,
    "Qwen/Qwen2.5-0.5B-Instruct": 0.5e9,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5e9,
    "Qwen/Qwen2.5-3B-Instruct": 3e9,
    "Qwen/Qwen2.5-7B-Instruct": 7e9,
    "Qwen/Qwen2.5-14B-Instruct": 14e9,
    "Qwen/Qwen2.5-32B-Instruct": 32e9,
    "Qwen/Qwen3-0.6B": 0.6e9,
    "Qwen/Qwen3-1.7B": 1.7e9,
    "Qwen/Qwen3-4B": 4e9,
    "Qwen/Qwen3-8B": 8e9,
    "Qwen/Qwen3-14B": 14e9,
    "Qwen/Qwen3-32B": 32e9,
    "Qwen/Qwen3.5-35B-A3B": 35e9,
    "gpt2": 124e6,
    "gpt2-medium": 355e6,
    "gpt2-large": 774e6,
    "gpt2-xl": 1.5e9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze outputs from run_research_suite.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to suite_manifest.csv",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=200,
        help="Bootstrap resamples per condition for prompt-level noise bands.",
    )
    return parser.parse_args()


def _ci95(mean: pd.Series, std: pd.Series, n: pd.Series) -> tuple[pd.Series, pd.Series]:
    sem = std / np.sqrt(np.maximum(n, 1))
    low = mean - 1.96 * sem
    high = mean + 1.96 * sem
    return low, high


def _bootstrap_ci_mean(
    values: np.ndarray,
    iters: int,
    seed: int = 12345,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(values[0]), float(values[0])
    means = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _fit_power_law(params: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(params) & np.isfinite(values) & (params > 0.0) & (values > 0.0)
    if int(mask.sum()) < 2:
        return np.nan, np.nan, np.nan
    x = np.log(params[mask])
    y = np.log(values[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    coef = float(np.exp(intercept))
    return coef, float(slope), r2


def _infer_model_params(model_id: str) -> float:
    if model_id in MODEL_PARAMS:
        return float(MODEL_PARAMS[model_id])

    mid = model_id.strip()
    lc = mid.lower()
    gpt2_alias = {
        "openai-community/gpt2": 124e6,
        "openai-community/gpt2-medium": 355e6,
        "openai-community/gpt2-large": 774e6,
        "openai-community/gpt2-xl": 1.5e9,
    }
    if lc in gpt2_alias:
        return float(gpt2_alias[lc])

    # Falls back to the first size token in the model id (for example 32B, 410M).
    match = re.search(r"(\d+(?:\.\d+)?)([bBmM])", mid)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        return float(value * (1e9 if unit == "b" else 1e6))

    return np.nan


def _load_manifest_df(manifest_csv: Path) -> pd.DataFrame:
    man = pd.read_csv(manifest_csv)
    rows = []
    for _, row in man.iterrows():
        run_dir = Path(row["run_dir"])
        metrics_path = run_dir / "metrics_full.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        df["suite_concept"] = row["concept_name"]
        df["suite_seed"] = int(row["seed"])
        df["run_dir"] = str(run_dir)
        rows.append(df)
    if not rows:
        raise ValueError("No metrics_full.csv files found from manifest.")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    manifest_csv = args.manifest
    suite_dir = manifest_csv.parent

    df = _load_manifest_df(manifest_csv)
    for col in [
        "recovery_slope",
        "drift_auc",
        "drift_auc_relative",
        "cad",
        "cad_relative",
        "persistence",
        "degradation",
        "prompt_style",
    ]:
        if col not in df.columns:
            if col == "prompt_style":
                df[col] = "unknown"
            else:
                df[col] = np.nan
    df["param_count"] = df["model_id"].map(_infer_model_params)
    df["log_params"] = np.log10(df["param_count"])

    raw_path = suite_dir / "suite_metrics_full.csv"
    df.to_csv(raw_path, index=False)

    agg = (
        df.groupby(
            ["suite_concept", "model_id", "vector_method", "alpha", "layer_index"],
            as_index=False,
        )
        .agg(
            recovery_mean=("recovery_fraction", "mean"),
            recovery_std=("recovery_fraction", "std"),
            recovery_slope_mean=("recovery_slope", "mean"),
            drift_auc_mean=("drift_auc", "mean"),
            drift_auc_rel_mean=("drift_auc_relative", "mean"),
            cad_mean=("cad", "mean"),
            persistence_mean=("persistence", "mean"),
            degradation_mean=("degradation", "mean"),
            peak_rel_mean=("peak_drift_relative", "mean"),
            peak_rel_std=("peak_drift_relative", "std"),
            next_token_kl_mean=("next_token_kl", "mean"),
            next_token_kl_std=("next_token_kl", "std"),
            crossed_rate=("crossed_baseline", "mean"),
            n=("recovery_fraction", "count"),
        )
    )
    low_r, high_r = _ci95(agg["recovery_mean"], agg["recovery_std"].fillna(0), agg["n"])
    low_p, high_p = _ci95(agg["peak_rel_mean"], agg["peak_rel_std"].fillna(0), agg["n"])
    agg["recovery_ci95_low"] = low_r
    agg["recovery_ci95_high"] = high_r
    agg["peak_rel_ci95_low"] = low_p
    agg["peak_rel_ci95_high"] = high_p
    low_kl, high_kl = _ci95(
        agg["next_token_kl_mean"],
        agg["next_token_kl_std"].fillna(0),
        agg["n"],
    )
    agg["next_token_kl_ci95_low"] = low_kl
    agg["next_token_kl_ci95_high"] = high_kl
    agg.to_csv(suite_dir / "suite_stratified_summary.csv", index=False)

    by_model = (
        df.groupby(["suite_concept", "model_id", "vector_method"], as_index=False)
        .agg(
            recovery_mean=("recovery_fraction", "mean"),
            recovery_std=("recovery_fraction", "std"),
            recovery_slope_mean=("recovery_slope", "mean"),
            drift_auc_mean=("drift_auc", "mean"),
            drift_auc_rel_mean=("drift_auc_relative", "mean"),
            cad_mean=("cad", "mean"),
            persistence_mean=("persistence", "mean"),
            degradation_mean=("degradation", "mean"),
            peak_rel_mean=("peak_drift_relative", "mean"),
            peak_rel_std=("peak_drift_relative", "std"),
            next_token_kl_mean=("next_token_kl", "mean"),
            next_token_kl_std=("next_token_kl", "std"),
            crossed_rate=("crossed_baseline", "mean"),
            n=("recovery_fraction", "count"),
        )
    )
    low_r2, high_r2 = _ci95(by_model["recovery_mean"], by_model["recovery_std"].fillna(0), by_model["n"])
    low_p2, high_p2 = _ci95(by_model["peak_rel_mean"], by_model["peak_rel_std"].fillna(0), by_model["n"])
    by_model["recovery_ci95_low"] = low_r2
    by_model["recovery_ci95_high"] = high_r2
    by_model["peak_rel_ci95_low"] = low_p2
    by_model["peak_rel_ci95_high"] = high_p2
    low_kl2, high_kl2 = _ci95(
        by_model["next_token_kl_mean"],
        by_model["next_token_kl_std"].fillna(0),
        by_model["n"],
    )
    by_model["next_token_kl_ci95_low"] = low_kl2
    by_model["next_token_kl_ci95_high"] = high_kl2
    by_model.to_csv(suite_dir / "suite_model_summary.csv", index=False)

    # Compare concept vector methods to random orthogonal control.
    target = by_model[by_model["vector_method"].isin(["mean_diff", "linear_probe"])]
    control = by_model[by_model["vector_method"] == "random_orthogonal"][
        ["suite_concept", "model_id", "recovery_mean", "peak_rel_mean", "next_token_kl_mean"]
    ].rename(
        columns={
            "recovery_mean": "recovery_mean_control",
            "peak_rel_mean": "peak_rel_mean_control",
            "next_token_kl_mean": "next_token_kl_mean_control",
        }
    )
    effects = target.merge(control, on=["suite_concept", "model_id"], how="left")
    effects["delta_recovery_vs_control"] = (
        effects["recovery_mean"] - effects["recovery_mean_control"]
    )
    effects["delta_peak_rel_vs_control"] = (
        effects["peak_rel_mean"] - effects["peak_rel_mean_control"]
    )
    effects["delta_next_token_kl_vs_control"] = (
        effects["next_token_kl_mean"] - effects["next_token_kl_mean_control"]
    )
    effects.to_csv(suite_dir / "suite_effect_vs_control.csv", index=False)

    style_summary = (
        df.groupby(
            ["suite_concept", "model_id", "vector_method", "prompt_style"],
            as_index=False,
        )
        .agg(
            recovery_mean=("recovery_fraction", "mean"),
            peak_rel_mean=("peak_drift_relative", "mean"),
            cad_mean=("cad", "mean"),
            persistence_mean=("persistence", "mean"),
            degradation_mean=("degradation", "mean"),
            n=("recovery_fraction", "count"),
        )
        .sort_values(["suite_concept", "model_id", "vector_method", "prompt_style"])
    )
    style_summary.to_csv(suite_dir / "suite_prompt_style_summary.csv", index=False)

    # Prompt bootstrap noise band per condition.
    boot_rows: list[dict[str, float | int | str]] = []
    group_cols = ["suite_concept", "model_id", "vector_method", "alpha", "layer_index"]
    for keys, sub in df.groupby(group_cols):
        concept, model_id, method, alpha, layer_index = keys
        rec = sub["recovery_fraction"].to_numpy(dtype=np.float64)
        peak_rel = sub["peak_drift_relative"].to_numpy(dtype=np.float64)
        kl = sub["next_token_kl"].to_numpy(dtype=np.float64)
        rec_low, rec_high = _bootstrap_ci_mean(rec, iters=args.bootstrap_iters, seed=11)
        peak_low, peak_high = _bootstrap_ci_mean(peak_rel, iters=args.bootstrap_iters, seed=13)
        kl_low, kl_high = _bootstrap_ci_mean(kl, iters=args.bootstrap_iters, seed=17)
        boot_rows.append(
            {
                "suite_concept": concept,
                "model_id": model_id,
                "vector_method": method,
                "alpha": float(alpha),
                "layer_index": int(layer_index),
                "n": int(len(sub)),
                "recovery_mean": float(np.mean(rec)),
                "recovery_boot_ci95_low": rec_low,
                "recovery_boot_ci95_high": rec_high,
                "peak_rel_mean": float(np.mean(peak_rel)),
                "peak_rel_boot_ci95_low": peak_low,
                "peak_rel_boot_ci95_high": peak_high,
                "next_token_kl_mean": float(np.mean(kl)),
                "next_token_kl_boot_ci95_low": kl_low,
                "next_token_kl_boot_ci95_high": kl_high,
            }
        )
    pd.DataFrame(boot_rows).to_csv(suite_dir / "suite_bootstrap_bands.csv", index=False)

    seed_consistency_rows: list[dict[str, float | str]] = []
    seed_group = (
        df.groupby(
            ["suite_concept", "vector_method", "model_id", "alpha", "layer_index", "suite_seed"],
            as_index=False,
        )
        .agg(
            recovery_mean=("recovery_fraction", "mean"),
            peak_rel_mean=("peak_drift_relative", "mean"),
            next_token_kl_mean=("next_token_kl", "mean"),
        )
    )

    for (concept, method), sub in seed_group.groupby(["suite_concept", "vector_method"]):
        pivot_r = sub.pivot_table(
            index=["model_id", "alpha", "layer_index"],
            columns="suite_seed",
            values="recovery_mean",
        )
        pivot_p = sub.pivot_table(
            index=["model_id", "alpha", "layer_index"],
            columns="suite_seed",
            values="peak_rel_mean",
        )
        pivot_kl = sub.pivot_table(
            index=["model_id", "alpha", "layer_index"],
            columns="suite_seed",
            values="next_token_kl_mean",
        )

        for metric_name, pivot_df in [
            ("recovery_mean", pivot_r),
            ("peak_rel_mean", pivot_p),
            ("next_token_kl_mean", pivot_kl),
        ]:
            if pivot_df.shape[1] < 2:
                continue
            cols = list(pivot_df.columns)
            x = pivot_df[cols[0]].to_numpy()
            y = pivot_df[cols[1]].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 2:
                corr = np.nan
            else:
                corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            seed_consistency_rows.append(
                {
                    "suite_concept": concept,
                    "vector_method": method,
                    "metric": metric_name,
                    "seed_a": int(cols[0]),
                    "seed_b": int(cols[1]),
                    "pearson_correlation": corr,
                    "n_points": int(mask.sum()),
                }
            )

    pd.DataFrame(seed_consistency_rows).to_csv(
        suite_dir / "suite_seed_consistency.csv",
        index=False,
    )

    # Scale trend fits by concept and method.
    trend_rows: list[dict[str, float | str]] = []
    for (concept, method), sub in by_model.groupby(["suite_concept", "vector_method"]):
        tmp = sub.copy()
        tmp["log_params"] = np.log10(tmp["model_id"].map(_infer_model_params))
        if tmp["log_params"].isna().any() or len(tmp) < 2:
            continue
        x = tmp["log_params"].to_numpy()
        y_recovery = tmp["recovery_mean"].to_numpy()
        y_peak_rel = tmp["peak_rel_mean"].to_numpy()
        slope_recovery, intercept_recovery = np.polyfit(x, y_recovery, 1)
        slope_peak_rel, intercept_peak_rel = np.polyfit(x, y_peak_rel, 1)
        trend_rows.append(
            {
                "suite_concept": concept,
                "vector_method": method,
                "recovery_slope_vs_log_params": float(slope_recovery),
                "recovery_intercept": float(intercept_recovery),
                "peak_rel_slope_vs_log_params": float(slope_peak_rel),
                "peak_rel_intercept": float(intercept_peak_rel),
                "n_models": int(len(tmp)),
            }
        )
    pd.DataFrame(trend_rows).to_csv(suite_dir / "suite_scale_trends.csv", index=False)

    law_rows: list[dict[str, float | str]] = []
    metric_specs = [
        ("cad_mean", "cad"),
        ("degradation_mean", "degradation"),
        ("persistence_mean", "persistence"),
    ]
    for (concept, method), sub in by_model.groupby(["suite_concept", "vector_method"]):
        params = sub["model_id"].map(_infer_model_params).to_numpy(dtype=np.float64)
        for col_name, metric_name in metric_specs:
            values = sub[col_name].to_numpy(dtype=np.float64)
            coef, exponent, r2 = _fit_power_law(params=params, values=values)
            law_rows.append(
                {
                    "suite_concept": concept,
                    "vector_method": method,
                    "metric": metric_name,
                    "coefficient": coef,
                    "exponent": exponent,
                    "r2": r2,
                    "n_models": int(sub["model_id"].nunique()),
                }
            )
    pd.DataFrame(law_rows).to_csv(suite_dir / "suite_scaling_laws.csv", index=False)

    with (suite_dir / "suite_report.md").open("w", encoding="utf-8") as f:
        f.write("# Suite Report\n\n")
        f.write(f"- Manifest: `{manifest_csv}`\n")
        f.write(f"- Total rows: `{len(df)}`\n")
        f.write(f"- Concepts: `{df['suite_concept'].nunique()}`\n")
        f.write(f"- Seeds: `{df['suite_seed'].nunique()}`\n")
        f.write(f"- Models: `{df['model_id'].nunique()}`\n\n")
        f.write("## Key outputs\n\n")
        f.write("- `suite_metrics_full.csv`\n")
        f.write("- `suite_stratified_summary.csv`\n")
        f.write("- `suite_model_summary.csv`\n")
        f.write("- `suite_effect_vs_control.csv`\n")
        f.write("- `suite_prompt_style_summary.csv`\n")
        f.write("- `suite_scale_trends.csv`\n")
        f.write("- `suite_scaling_laws.csv`\n")
        f.write("- `suite_seed_consistency.csv`\n")
        f.write("- `suite_bootstrap_bands.csv`\n")

    print(f"[suite-analysis] wrote {suite_dir / 'suite_metrics_full.csv'}")
    print(f"[suite-analysis] wrote {suite_dir / 'suite_stratified_summary.csv'}")
    print(f"[suite-analysis] wrote {suite_dir / 'suite_model_summary.csv'}")
    print(f"[suite-analysis] wrote {suite_dir / 'suite_effect_vs_control.csv'}")
    print(f"[suite-analysis] wrote {suite_dir / 'suite_prompt_style_summary.csv'}")
    print(f"[suite-analysis] wrote {suite_dir / 'suite_scale_trends.csv'}")
    print(f"[suite-analysis] wrote {suite_dir / 'suite_scaling_laws.csv'}")
    print(f"[suite-analysis] wrote {suite_dir / 'suite_seed_consistency.csv'}")
    print(f"[suite-analysis] wrote {suite_dir / 'suite_bootstrap_bands.csv'}")
    print(f"[suite-analysis] wrote {suite_dir / 'suite_report.md'}")


if __name__ == "__main__":
    main()
