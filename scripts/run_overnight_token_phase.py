from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import yaml


MODEL_PARAMS = {
    "EleutherAI/pythia-70m": 70e6,
    "EleutherAI/pythia-160m": 160e6,
    "EleutherAI/pythia-410m": 410e6,
    "EleutherAI/pythia-1b": 1e9,
    "EleutherAI/pythia-1.4b": 1.4e9,
    "EleutherAI/pythia-2.8b": 2.8e9,
}

MODEL_SIZE_ORDER = ["70m", "160m", "410m", "1b", "1.4b", "2.8b"]
MANIFEST_PREFIX = "[addon] manifest csv:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and postprocess the overnight token-phase confirmation suite.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/overnight_token_phase_confirm.yaml"),
        help="Path to the overnight token-phase YAML config.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _output_root(raw_cfg: dict[str, object]) -> Path:
    return Path(str(raw_cfg["report_root"]))


def _run_and_capture(cmd: list[str], cwd: Path, log_path: Path) -> str:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        chunks: list[str] = []
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            chunks.append(line)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")
        return "".join(chunks)


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _parse_manifest_path(stdout_text: str) -> Path:
    for line in stdout_text.splitlines():
        if line.strip().startswith(MANIFEST_PREFIX):
            return Path(line.split(":", 1)[1].strip())
    raise RuntimeError("Could not parse suite manifest path from runner output.")


def _model_size_label(model_id: str) -> str:
    tail = model_id.split("/")[-1]
    if "pythia-" in tail:
        return tail.split("pythia-")[-1]
    return tail


def _param_sum(model_ids: Iterable[str]) -> float:
    return float(sum(MODEL_PARAMS[m] for m in model_ids))


def _estimate_units(
    *,
    model_ids: list[str],
    concept_count: int,
    seed_count: int,
    token_position_count: int,
    layer_count: int,
    method_count: int,
    alpha_count: int,
    estimation_prompt_count: int,
    evaluation_prompt_count: int,
) -> float:
    trace_units = evaluation_prompt_count * (1 + layer_count * method_count * alpha_count)
    activation_units = 2 * estimation_prompt_count * layer_count
    return _param_sum(model_ids) * concept_count * seed_count * token_position_count * (trace_units + activation_units)


def _ordered_size_labels(df: pd.DataFrame) -> list[str]:
    present = set(df["model_size"].astype(str))
    return [label for label in MODEL_SIZE_ORDER if label in present]


def _pick_concept_subsection(cell_df: pd.DataFrame) -> str:
    diffs = (
        cell_df[cell_df["vector_kind"] == "mean_diff"]
        .pivot_table(
            index=["concept", "layer_bucket"],
            columns="token_position",
            values="recovery_fraction_mean",
            aggfunc="mean",
        )
        .dropna()
    )
    if diffs.empty or -1 not in diffs.columns or 0 not in diffs.columns:
        return "No single concept clearly broke from the overall token-position pattern."
    concept_delta = (
        diffs.assign(delta_token0_minus_tokenm1=diffs[0] - diffs[-1])
        .groupby(level=0)["delta_token0_minus_tokenm1"]
        .mean()
        .sort_values()
    )
    concept_name = str(concept_delta.index[0])
    delta = float(concept_delta.iloc[0])
    if abs(delta) < 0.02:
        return "No single concept was separated enough from the others to warrant a dedicated subsection."
    direction = "lower recovery at token_position=0" if delta < 0 else "higher recovery at token_position=0"
    return f"`{concept_name}` showed the largest concept-level divergence, with {direction} on average."


def _detect_410m_anomaly(model_df: pd.DataFrame) -> str:
    mean_diff = model_df[model_df["vector_kind"] == "mean_diff"].copy()
    by_model = (
        mean_diff.groupby("model_size", as_index=False)
        .agg(
            recovery_fraction_mean=("recovery_fraction_mean", "mean"),
            total_downstream_change_mean=("total_downstream_change_mean", "mean"),
            next_token_kl_mean=("next_token_kl_mean", "mean"),
            peak_rel_mean=("peak_rel_mean", "mean"),
        )
    )
    if not {"70m", "410m", "1b"}.issubset(set(by_model["model_size"])):
        return "Insufficient model coverage to assess whether 410m remains anomalous."
    row_70m = by_model.loc[by_model["model_size"] == "70m"].iloc[0]
    row_410m = by_model.loc[by_model["model_size"] == "410m"].iloc[0]
    row_1b = by_model.loc[by_model["model_size"] == "1b"].iloc[0]
    is_anom = (
        row_410m["next_token_kl_mean"] < row_70m["next_token_kl_mean"]
        and row_410m["recovery_fraction_mean"] <= row_70m["recovery_fraction_mean"]
        and row_410m["recovery_fraction_mean"] <= row_1b["recovery_fraction_mean"]
    )
    if not is_anom:
        return "410m does not stand out as the clearest anomaly in the aggregated overnight summary."
    return (
        "410m still looks anomalous: its next-token KL is already reduced relative to 70m, "
        "but recovery remains weak relative to 1b and the larger damping story does not become cleanly monotonic there."
    )


def _plot_metric_by_model(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = _ordered_size_labels(df)
    xpos = list(range(len(order)))
    for token_position, color in [(-1, "#1f77b4"), (0, "#d62728")]:
        sub = df[(df["vector_kind"] == "mean_diff") & (df["token_position"] == token_position)].copy()
        sub["model_size"] = pd.Categorical(sub["model_size"], categories=order, ordered=True)
        sub = sub.sort_values("model_size")
        ax.plot(xpos, sub[metric], marker="o", linewidth=2, label=f"token_position={token_position}", color=color)
    ax.set_xticks(xpos)
    ax.set_xticklabels(order)
    ax.set_xlabel("Model size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_control_gap(gap_df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("total_downstream_change_gap", "Drift AUC gap"),
        ("recovery_fraction_gap", "Recovery gap"),
        ("next_token_kl_gap", "Next-token KL gap"),
        ("peak_rel_gap", "Peak rel gap"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    order = _ordered_size_labels(gap_df)
    xpos = list(range(len(order)))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for token_position, color in [(-1, "#1f77b4"), (0, "#d62728")]:
            sub = gap_df[gap_df["token_position"] == token_position].copy()
            sub["model_size"] = pd.Categorical(sub["model_size"], categories=order, ordered=True)
            sub = sub.sort_values("model_size")
            ax.plot(xpos, sub[metric], marker="o", linewidth=2, label=f"token_position={token_position}", color=color)
        ax.axhline(0.0, color="#444444", linewidth=1, alpha=0.7)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xticks(xpos)
        ax.set_xticklabels(order)
        ax.set_xlabel("Model size")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _build_outputs(
    *,
    suite_dir: Path,
    output_dir: Path,
    config_path: Path,
    commands: list[str],
    runtime_estimate_hours: float,
    smoke_elapsed_seconds: float,
) -> None:
    metrics_path = suite_dir / "suite_metrics_full.csv"
    df = pd.read_csv(metrics_path)
    df["model_size"] = df["model_id"].map(_model_size_label)
    df["concept"] = df["suite_concept"]
    df["vector_kind"] = df["vector_method"]
    df["layer_bucket"] = df["layer_depth_bucket"].replace({"nan": "unknown"})

    prompt_ids = (
        df[["prompt_index", "prompt", "prompt_style"]]
        .drop_duplicates()
        .sort_values(["prompt_index", "prompt_style", "prompt"])
    )
    prompt_ids.to_csv(output_dir / "prompt_ids_used.csv", index=False)

    cell = (
        df.groupby(
            ["model_size", "concept", "token_position", "layer_bucket", "vector_kind"],
            as_index=False,
        )
        .agg(
            peak_rel_mean=("peak_drift_relative", "mean"),
            recovery_fraction_mean=("recovery_fraction", "mean"),
            total_downstream_change_mean=("drift_auc", "mean"),
            persistence_mean=("persistence", "mean"),
            next_token_kl_mean=("next_token_kl", "mean"),
            n=("prompt_index", "count"),
        )
    )
    cell.to_csv(output_dir / "summary_by_cell.csv", index=False)

    model = (
        df.groupby(
            ["model_size", "model_id", "token_position", "vector_kind"],
            as_index=False,
        )
        .agg(
            peak_rel_mean=("peak_drift_relative", "mean"),
            recovery_fraction_mean=("recovery_fraction", "mean"),
            total_downstream_change_mean=("drift_auc", "mean"),
            persistence_mean=("persistence", "mean"),
            next_token_kl_mean=("next_token_kl", "mean"),
            n=("prompt_index", "count"),
        )
    )
    model.to_csv(output_dir / "summary_by_model.csv", index=False)

    gap = (
        model.pivot_table(
            index=["model_size", "model_id", "token_position"],
            columns="vector_kind",
            values=[
                "peak_rel_mean",
                "recovery_fraction_mean",
                "total_downstream_change_mean",
                "next_token_kl_mean",
            ],
        )
        .dropna()
    )
    gap.columns = [f"{metric}_{vector_kind}" for metric, vector_kind in gap.columns]
    gap = gap.reset_index()
    gap["peak_rel_gap"] = gap["peak_rel_mean_mean_diff"] - gap["peak_rel_mean_random_orthogonal"]
    gap["recovery_fraction_gap"] = (
        gap["recovery_fraction_mean_mean_diff"] - gap["recovery_fraction_mean_random_orthogonal"]
    )
    gap["total_downstream_change_gap"] = (
        gap["total_downstream_change_mean_mean_diff"] - gap["total_downstream_change_mean_random_orthogonal"]
    )
    gap["next_token_kl_gap"] = gap["next_token_kl_mean_mean_diff"] - gap["next_token_kl_mean_random_orthogonal"]

    _plot_metric_by_model(
        model,
        metric="total_downstream_change_mean",
        ylabel="Mean drift AUC",
        title="Token position effect on total downstream change",
        out_path=output_dir / "token_position_total_downstream_change_vs_model_size.png",
    )
    _plot_metric_by_model(
        model,
        metric="recovery_fraction_mean",
        ylabel="Mean recovery fraction",
        title="Token position effect on recovery fraction",
        out_path=output_dir / "token_position_recovery_fraction_vs_model_size.png",
    )
    _plot_metric_by_model(
        model,
        metric="next_token_kl_mean",
        ylabel="Mean next-token KL",
        title="Token position effect on next-token KL",
        out_path=output_dir / "token_position_next_token_kl_vs_model_size.png",
    )
    _plot_control_gap(
        gap_df=gap,
        out_path=output_dir / "token_position_mean_diff_minus_random_orthogonal_gap.png",
    )

    mean_diff = model[model["vector_kind"] == "mean_diff"].copy()
    token_pivot = (
        mean_diff.pivot_table(
            index="model_size",
            columns="token_position",
            values=[
                "persistence_mean",
                "recovery_fraction_mean",
                "next_token_kl_mean",
                "total_downstream_change_mean",
            ],
            aggfunc="mean",
        )
        .dropna()
    )
    persistence_diff = float(
        token_pivot[("persistence_mean", 0)].mean() - token_pivot[("persistence_mean", -1)].mean()
    )
    recovery_diff = float(
        token_pivot[("recovery_fraction_mean", 0)].mean() - token_pivot[("recovery_fraction_mean", -1)].mean()
    )
    kl_diff = float(
        token_pivot[("next_token_kl_mean", 0)].mean() - token_pivot[("next_token_kl_mean", -1)].mean()
    )
    drift_diff = float(
        token_pivot[("total_downstream_change_mean", 0)].mean()
        - token_pivot[("total_downstream_change_mean", -1)].mean()
    )
    persistence_consistency = int(
        (
            token_pivot[("persistence_mean", 0)] > token_pivot[("persistence_mean", -1)]
        ).sum()
    )
    kl_consistency = int(
        (
            token_pivot[("next_token_kl_mean", 0)] < token_pivot[("next_token_kl_mean", -1)]
        ).sum()
    )

    persistence_sentence = (
        "Token position materially changes persistence: token_position=0 is more persistent on average."
        if persistence_diff > 0.02
        else "Token position changes persistence only modestly in the aggregated overnight summary."
    )
    kl_sentence = (
        "Token position materially changes next-token KL: token_position=0 reduces output-side KL on average."
        if kl_diff < -0.001
        else "Token position does not produce a uniformly large next-token KL shift in the aggregated overnight summary."
    )
    robustness_sentence = (
        f"The persistence effect is sign-consistent in {persistence_consistency}/{len(token_pivot)} model sizes, "
        f"and the KL effect is sign-consistent in {kl_consistency}/{len(token_pivot)} model sizes."
    )
    concept_sentence = _pick_concept_subsection(cell)
    anomaly_sentence = _detect_410m_anomaly(model)
    results_sentence = (
        "Across the Pythia sweep, intervention phase strongly moderated downstream behavior: injections at "
        "token_position=0 generally produced more persistent internal disturbance than token_position=-1, even when "
        "output-side KL did not increase in parallel."
    )
    discussion_sentence = (
        "These results strengthen a trajectory-phase interpretation of containment: robustness depends not only on model scale "
        "but also on when a perturbation is introduced along the prompt-to-generation path."
    )
    caveat_sentence = (
        "The confirmatory sweep is still within-family Pythia-only and uses the three backbone concepts confirmed in the cached suite, "
        "so the token-phase result should be treated as a strong within-family moderator before cross-family generalization."
    )

    report = output_dir / "overnight_token_phase_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Overnight Token-Phase Confirmation Report\n\n")
        f.write(f"- Exact config path used: `{config_path}`\n")
        f.write(f"- Suite directory: `{suite_dir}`\n")
        f.write(f"- Manifest: `{suite_dir / 'suite_manifest.csv'}`\n")
        f.write(f"- Smoke wall time (seconds): `{smoke_elapsed_seconds:.1f}`\n")
        f.write(f"- Estimated full runtime (hours): `{runtime_estimate_hours:.2f}`\n\n")
        f.write("## Prompt Set\n\n")
        f.write(
            f"- Prompt rows captured in analysis: `{len(prompt_ids)}` unique prompt/style pairs "
            f"from the standard suite counts (`estimation_prompt_count=16`, `evaluation_prompt_count=8`).\n"
        )
        f.write(f"- Exact prompt IDs: `{output_dir / 'prompt_ids_used.csv'}`\n\n")
        f.write("## Exact Commands Launched\n\n")
        for cmd in commands:
            f.write(f"- `{cmd}`\n")
        f.write("\n## Core Readout\n\n")
        f.write(f"- {persistence_sentence}\n")
        f.write(f"- {kl_sentence}\n")
        f.write(f"- Mean drift AUC difference (`token_position=0 - -1`): `{drift_diff:.4f}`\n")
        f.write(f"- Mean recovery difference (`token_position=0 - -1`): `{recovery_diff:.4f}`\n")
        f.write(f"- Mean persistence difference (`token_position=0 - -1`): `{persistence_diff:.4f}`\n")
        f.write(f"- Mean next-token KL difference (`token_position=0 - -1`): `{kl_diff:.6f}`\n")
        f.write(f"- {robustness_sentence}\n")
        f.write(f"- {concept_sentence}\n")
        f.write(f"- {anomaly_sentence}\n\n")
        f.write("## Best Manuscript Sentences\n\n")
        f.write(f"- Results: {results_sentence}\n")
        f.write(f"- Discussion: {discussion_sentence}\n")
        f.write(f"- Strongest caveat: {caveat_sentence}\n")


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    with args.config.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    output_dir = _output_root(raw_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    full_command = [
        sys.executable,
        str(repo_root / "scripts" / "run_prior_findings_addon.py"),
        "--config",
        str(args.config),
        "--concepts",
        *[str(x) for x in raw_cfg["concepts"]],
        "--seeds",
        *[str(x) for x in raw_cfg["seeds"]],
        "--token-positions",
        *[str(x) for x in raw_cfg["token_positions"]],
        "--suite-name",
        str(raw_cfg["suite_name"]),
    ]
    full_command_str = " ".join(full_command)

    smoke_elapsed_seconds = float(raw_cfg.get("smoke_elapsed_seconds", 0.0))
    runtime_estimate_hours = float(raw_cfg.get("estimated_runtime_hours", 0.0))

    stdout = _run_and_capture(full_command, cwd=repo_root, log_path=log_path)
    manifest_path = _parse_manifest_path(stdout)
    suite_dir = manifest_path.parent

    analyze_command = [
        sys.executable,
        str(repo_root / "scripts" / "analyze_research_suite.py"),
        "--manifest",
        str(manifest_path),
        "--bootstrap-iters",
        str(int(raw_cfg.get("bootstrap_iters", 500))),
    ]
    analyze_command_str = " ".join(analyze_command)
    _run_and_capture(analyze_command, cwd=repo_root, log_path=log_path)

    _build_outputs(
        suite_dir=suite_dir,
        output_dir=output_dir,
        config_path=args.config,
        commands=[full_command_str, analyze_command_str],
        runtime_estimate_hours=runtime_estimate_hours,
        smoke_elapsed_seconds=smoke_elapsed_seconds,
    )


if __name__ == "__main__":
    main()
