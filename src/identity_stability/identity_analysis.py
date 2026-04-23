from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def bootstrap_mean_ci(values: list[float] | np.ndarray, iters: int = 1000, seed: int = 123) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = np.empty(iters, dtype=np.float64)
    for idx in range(iters):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[idx] = float(sample.mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def cluster_mean_values(
    df: pd.DataFrame,
    cluster_cols: list[str],
    value_col: str,
) -> np.ndarray:
    available_cols = [col for col in cluster_cols if col in df.columns]
    if not available_cols:
        return df[value_col].astype(float).to_numpy(dtype=np.float64)
    return (
        df.groupby(available_cols, dropna=False)[value_col]
        .mean()
        .astype(float)
        .to_numpy(dtype=np.float64)
    )


def cluster_bootstrap_mean_ci(
    df: pd.DataFrame,
    cluster_cols: list[str],
    value_col: str,
    iters: int = 1000,
    seed: int = 123,
) -> tuple[float, float]:
    return bootstrap_mean_ci(
        cluster_mean_values(df, cluster_cols=cluster_cols, value_col=value_col),
        iters=iters,
        seed=seed,
    )


def add_bootstrap_ci(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    seed: int = 123,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        low, high = bootstrap_mean_ci(sub[value_col].tolist(), seed=seed)
        row[f"{value_col}_mean"] = float(sub[value_col].mean())
        row[f"{value_col}_ci95_low"] = low
        row[f"{value_col}_ci95_high"] = high
        row["n"] = int(len(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * m
    running = 0.0
    for rank, (idx, p_val) in enumerate(indexed, start=1):
        adj = min(1.0, (m - rank + 1) * p_val)
        running = max(running, adj)
        adjusted[idx] = running
    return adjusted


def fit_continuous_model(
    df: pd.DataFrame,
    formula: str,
    group_col: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"formula": formula, "group_col": group_col, "mode": "unfit"}
    try:
        model = smf.mixedlm(formula, data=df, groups=df[group_col])
        fit = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        result["mode"] = "mixedlm"
        result["params"] = {str(k): float(v) for k, v in fit.params.items()}
        result["pvalues"] = {str(k): float(v) for k, v in fit.pvalues.items()}
        return result
    except Exception as exc:
        result["mixedlm_error"] = str(exc)

    fit = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[group_col]},
    )
    result["mode"] = "clustered_ols"
    result["params"] = {str(k): float(v) for k, v in fit.params.items()}
    result["pvalues"] = {str(k): float(v) for k, v in fit.pvalues.items()}
    return result


def fit_binary_model(
    df: pd.DataFrame,
    formula: str,
    group_col: str,
) -> dict[str, Any]:
    fit = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[group_col]},
    )
    return {
        "formula": formula,
        "group_col": group_col,
        "mode": "clustered_glm_binomial",
        "params": {str(k): float(v) for k, v in fit.params.items()},
        "pvalues": {str(k): float(v) for k, v in fit.pvalues.items()},
    }


def rank_robustness(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for column in [
        "seed_replication",
        "prompt_family_consistency",
        "ood_stability",
        "control_gap",
        "ci_width_penalty",
    ]:
        if column not in work.columns:
            work[column] = 0.0
    work["robustness_score"] = (
        0.30 * work["seed_replication"]
        + 0.25 * work["prompt_family_consistency"]
        + 0.25 * work["ood_stability"]
        + 0.15 * work["control_gap"]
        - 0.05 * work["ci_width_penalty"]
    )
    return work.sort_values("robustness_score", ascending=False).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
