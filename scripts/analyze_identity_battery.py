from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from identity_battery_common import add_src_to_path, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze identity battery outputs and write figures/reports.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_analysis import fit_binary_model, fit_continuous_model, rank_robustness

    args = parse_args()
    config = load_yaml_config(args.config)
    root = Path(config["output_root"])
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    boundary = _load_csv(root / "identity_boundary_sweep" / "summary.csv")
    boundary_ci = _load_csv(root / "identity_boundary_sweep" / "axis_shift_bootstrap.csv")
    longform = _load_csv(root / "longform_return" / "results.csv")
    chunk_curves = _load_csv(root / "longform_return" / "chunk_curves.csv")
    self_report = _load_csv(root / "self_report_behavior" / "summary.csv")
    self_prediction = _load_csv(root / "self_prediction_calibration" / "summary.csv")
    self_recognition = _load_csv(root / "self_recognition_from_foils" / "summary.csv")
    self_other = _load_csv(root / "self_other_boundary" / "summary.csv")
    hidden = _load_csv(root / "hidden_style_charter" / "summary.csv")
    ood = _load_csv(root / "ood_robustness" / "summary.csv")
    adaptive = _load_csv(root / "adaptive_baseline" / "summary.csv")

    stats_summary: dict[str, object] = {}
    robustness_rows: list[dict[str, object]] = []

    if longform is not None and not longform.empty:
        heatmap_df = (
            longform.groupby(["model_size_label", "identity_frame"], as_index=False)
            .agg(return_to_baseline_index_mean=("return_to_baseline_index", "mean"))
        )
        heatmap_df.to_csv(fig_dir / "heatmap_identity_frame_model_size_return_to_baseline.csv", index=False)
        pivot = heatmap_df.pivot(index="identity_frame", columns="model_size_label", values="return_to_baseline_index_mean")
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title("Return-to-Baseline Index")
        fig.colorbar(im, ax=ax)
        _save_plot(fig, fig_dir / "heatmap_identity_frame_model_size_return_to_baseline.png")

        if chunk_curves is not None and not chunk_curves.empty:
            curve_df = (
                chunk_curves.groupby(["condition", "chunk_index"], as_index=False)
                .agg(chunk_return_to_baseline_index_mean=("chunk_return_to_baseline_index", "mean"))
            )
            curve_df.to_csv(fig_dir / "line_tokenwise_drift_recovery_curves.csv", index=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            for condition, sub in curve_df.groupby("condition"):
                ax.plot(sub["chunk_index"], sub["chunk_return_to_baseline_index_mean"], marker="o", label=condition)
            ax.set_xlabel("Chunk Index")
            ax.set_ylabel("Return-to-Baseline")
            ax.set_title("Long-Form Recovery Curves")
            ax.legend()
            _save_plot(fig, fig_dir / "line_tokenwise_drift_recovery_curves.png")

        if len(longform) >= 4:
            stats_summary["longform_model"] = fit_continuous_model(
                df=longform,
                formula="return_to_baseline_index ~ C(model_size_label) + C(identity_frame) + C(condition)",
                group_col="dialogue_id",
            )
        robustness_rows.append(
            {
                "finding": "longform_return",
                "seed_replication": 0.0,
                "prompt_family_consistency": float(longform["return_to_baseline_index"].mean()),
                "ood_stability": 0.0,
                "control_gap": 0.0,
                "ci_width_penalty": 0.0,
            }
        )

    if self_report is not None and not self_report.empty:
        scatter_df = self_report.copy()
        scatter_df.to_csv(fig_dir / "scatter_self_report_behavior_coupling.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(scatter_df["behavior_mean_score"], scatter_df["coupling_score_mean"])
        ax.set_xlabel("Behavior Mean Score")
        ax.set_ylabel("Coupling Score")
        ax.set_title("Self-Report vs Behavior Coupling")
        _save_plot(fig, fig_dir / "scatter_self_report_behavior_coupling.png")
        robustness_rows.append(
            {
                "finding": "self_report_behavior",
                "seed_replication": 0.0,
                "prompt_family_consistency": float(self_report["self_behavior_correlation"].fillna(0.0).mean()),
                "ood_stability": 0.0,
                "control_gap": 0.0,
                "ci_width_penalty": 0.0,
            }
        )

    if self_prediction is not None and not self_prediction.empty:
        self_prediction.to_csv(fig_dir / "bar_self_prediction_calibration.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [
            f"{row.model_size_label}\n{row.identity_frame}\n{row.axis_name}"
            for row in self_prediction.itertuples()
        ]
        ax.bar(labels, self_prediction["sign_accuracy_mean"])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Sign Accuracy")
        ax.set_title("Self-Prediction Calibration")
        _save_plot(fig, fig_dir / "bar_self_prediction_calibration.png")
        robustness_rows.append(
            {
                "finding": "self_prediction_calibration",
                "seed_replication": 0.0,
                "prompt_family_consistency": float(self_prediction["sign_accuracy_mean"].mean()),
                "ood_stability": 0.0,
                "control_gap": float(1.0 - self_prediction["calibration_error_mean"].mean()),
                "ci_width_penalty": 0.0,
            }
        )

    if self_recognition is not None and not self_recognition.empty:
        self_recognition.to_csv(fig_dir / "bar_self_recognition_from_foils.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [
            f"{row.model_size_label}\n{row.identity_frame}\n{row.axis_name}"
            for row in self_recognition.itertuples()
        ]
        ax.bar(labels, self_recognition["self_recognition_accuracy_mean"])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Choose-self accuracy")
        ax.set_title("Self Recognition From Foils")
        _save_plot(fig, fig_dir / "bar_self_recognition_from_foils.png")
        robustness_rows.append(
            {
                "finding": "self_recognition_from_foils",
                "seed_replication": 0.0,
                "prompt_family_consistency": float(self_recognition["self_recognition_accuracy_mean"].mean()),
                "ood_stability": 0.0,
                "control_gap": float(self_recognition["baseline_vs_contrary_axis_gap_mean"].mean()),
                "ci_width_penalty": 0.0,
            }
        )

    if self_other is not None and not self_other.empty:
        self_other.to_csv(fig_dir / "bar_self_other_boundary.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [
            f"{row.model_size_label}\n{row.identity_frame}\n{row.axis_name}"
            for row in self_other.itertuples()
        ]
        x = range(len(labels))
        width = 0.4
        ax.bar([i - width / 2 for i in x], self_other["boundary_retention_no_steer_mean"], width=width, label="no_steer")
        ax.bar([i + width / 2 for i in x], self_other["boundary_retention_steer_mean"], width=width, label="steered_self")
        ax.set_xticks(list(x), labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Boundary retention")
        ax.set_title("Self/Other Boundary Retention")
        ax.legend()
        _save_plot(fig, fig_dir / "bar_self_other_boundary.png")
        robustness_rows.append(
            {
                "finding": "self_other_boundary",
                "seed_replication": 0.0,
                "prompt_family_consistency": float(self_other["boundary_retention_no_steer_mean"].mean()),
                "ood_stability": float(self_other["boundary_retention_steer_mean"].mean()),
                "control_gap": float(self_other["boundary_margin_delta_mean"].mean()),
                "ci_width_penalty": 0.0,
            }
        )

    if hidden is not None and not hidden.empty:
        hidden.to_csv(fig_dir / "bar_hidden_style_charter_consistency.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f"{row.identity_frame}\n{row.condition}" for row in hidden.itertuples()]
        ax.bar(labels, hidden["consistency_score_mean"])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Consistency")
        ax.set_title("Hidden Style Charter Consistency")
        _save_plot(fig, fig_dir / "bar_hidden_style_charter_consistency.png")
        if len(hidden) >= 4:
            hidden_binary = hidden.copy()
            hidden_binary["high_consistency"] = (hidden_binary["consistency_score_mean"] >= 0.75).astype(int)
            stats_summary["hidden_charter_model"] = fit_binary_model(
                df=hidden_binary,
                formula="high_consistency ~ C(identity_frame) + C(condition)",
                group_col="model_size_label",
            )
        robustness_rows.append(
            {
                "finding": "hidden_style_charter",
                "seed_replication": 0.0,
                "prompt_family_consistency": float(hidden["consistency_score_mean"].mean()),
                "ood_stability": 0.0,
                "control_gap": 0.0,
                "ci_width_penalty": 0.0,
            }
        )

    if adaptive is not None and not adaptive.empty:
        adaptive.to_csv(fig_dir / "comparison_fixed_layer_vs_adaptive.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_df = (
            adaptive.groupby("strategy", as_index=False)
            .agg(axis_shift_mean=("axis_shift_mean", "mean"))
        )
        ax.bar(plot_df["strategy"], plot_df["axis_shift_mean"])
        ax.set_ylabel("Axis Shift")
        ax.set_title("Fixed vs Adaptive Baseline")
        _save_plot(fig, fig_dir / "comparison_fixed_layer_vs_adaptive.png")
        robustness_rows.append(
            {
                "finding": "adaptive_baseline",
                "seed_replication": 0.0,
                "prompt_family_consistency": float(plot_df["axis_shift_mean"].mean()),
                "ood_stability": 0.0,
                "control_gap": 0.0,
                "ci_width_penalty": 0.0,
            }
        )

    if ood is not None and not ood.empty:
        ood.to_csv(fig_dir / "breakdown_anti_steerable_prompt_families.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(ood["wrapper_family"], ood["anti_steerable_rate"])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Anti-Steerable Rate")
        ax.set_title("OOD Failures")
        _save_plot(fig, fig_dir / "breakdown_anti_steerable_prompt_families.png")
        robustness_rows.append(
            {
                "finding": "ood_robustness",
                "seed_replication": 0.0,
                "prompt_family_consistency": float(1.0 - ood["sign_flip_rate"].mean()),
                "ood_stability": float(ood["retained_effect_size"].mean()),
                "control_gap": float(ood["control_gap_mean"].mean()),
                "ci_width_penalty": 0.0,
            }
        )

    if boundary is not None and not boundary.empty:
        summary_panel = (
            boundary[boundary["vector_kind"] == "mean_diff"]
            .groupby(["model_size_label", "identity_frame"], as_index=False)
            .agg(recovery_fraction_mean=("recovery_fraction_mean", "mean"))
        )
        summary_panel.to_csv(fig_dir / "summary_panel_identity_frame_backbone_metrics.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot = summary_panel.pivot(index="identity_frame", columns="model_size_label", values="recovery_fraction_mean")
        im = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title("Boundary Sweep Recovery Summary")
        fig.colorbar(im, ax=ax)
        _save_plot(fig, fig_dir / "summary_panel_identity_frame_backbone_metrics.png")
        if boundary_ci is not None and not boundary_ci.empty:
            boundary_ci.to_csv(fig_dir / "summary_panel_identity_frame_backbone_metrics_ci.csv", index=False)
        robustness_rows.append(
            {
                "finding": "identity_boundary_sweep",
                "seed_replication": 0.0,
                "prompt_family_consistency": float(boundary["axis_shift_mean"].mean()),
                "ood_stability": 0.0,
                "control_gap": float(boundary[boundary["vector_kind"] == "mean_diff"]["axis_shift_mean"].mean()),
                "ci_width_penalty": 0.0,
            }
        )

    robustness = rank_robustness(pd.DataFrame(robustness_rows)) if robustness_rows else pd.DataFrame()
    robustness.to_csv(root / "robustness_ranking.csv", index=False)

    strongest_positive = "No positive finding available yet."
    strongest_null = "No null finding available yet."
    strongest_negative = "No negative finding available yet."
    strongest_family_divergence = "Family divergence not available yet."

    if longform is not None and not longform.empty:
        best = longform.sort_values("return_to_baseline_index", ascending=False).iloc[0]
        strongest_positive = (
            f"Strongest positive smoke-tier finding: {best['condition']} on {best['dialogue_id']} "
            f"for {best['model_size_label']} / {best['identity_frame']} with return-to-baseline "
            f"index {best['return_to_baseline_index']:.3f}."
        )
    if self_report is not None and not self_report.empty:
        null_row = self_report.iloc[(self_report["coupling_score_mean"].abs()).argmin()]
        strongest_null = (
            f"Strongest null smoke-tier finding: {null_row['dimension']} under {null_row['identity_frame']} "
            f"had coupling score {null_row['coupling_score_mean']:.3f}."
        )
    if ood is not None and not ood.empty:
        neg = ood.sort_values("anti_steerable_rate", ascending=False).iloc[0]
        strongest_negative = (
            f"Strongest anti-steerable smoke-tier finding: {neg['wrapper_family']} for "
            f"{neg['axis_name']} under {neg['identity_frame']} had anti-steerable rate "
            f"{neg['anti_steerable_rate']:.3f}."
        )
    if boundary is not None and not boundary.empty:
        fam = boundary.sort_values("axis_shift_mean", ascending=False).iloc[0]
        strongest_family_divergence = (
            f"Strongest family-specific divergence currently visible in smoke data: "
            f"{fam['model_size_label']} / {fam['identity_frame']} / {fam['concept_axis']} / "
            f"{fam['vector_kind']} with mean axis shift {fam['axis_shift_mean']:.3f}."
        )

    with (root / "research_upgrade_report.md").open("w", encoding="utf-8") as f:
        f.write("# Research Upgrade Report\n\n")
        f.write("This report is provisional and reflects the current smoke-tier execution state.\n\n")
        f.write("## Headline Findings\n\n")
        f.write(f"1. {strongest_positive}\n")
        f.write(f"2. {strongest_null}\n")
        f.write(f"3. {strongest_negative}\n")
        f.write(f"4. {strongest_family_divergence}\n\n")
        f.write("## Answers To Core Questions\n\n")
        f.write("- Whether identity framing changed internal resistance: smoke-tier only; use `outputs/latest/identity_boundary_sweep/summary.csv`.\n")
        f.write("- Whether identity framing changed long-form return: smoke-tier only; use `outputs/latest/longform_return/results.csv`.\n")
        f.write("- Whether self-report predicted behavior: smoke-tier only; use `outputs/latest/self_report_behavior/summary.csv`.\n")
        f.write("- Whether hidden-charter consistency held up: smoke-tier only; use `outputs/latest/hidden_style_charter/summary.csv`.\n")
        f.write("- Whether adaptive steering changed the interpretation of resistance: smoke-tier only; use `outputs/latest/adaptive_baseline/summary.csv`.\n\n")
        f.write("## Run State\n\n")
        f.write("- Completed in this pass: smoke-tier baseline audit plus identity-battery code path setup.\n")
        f.write("- Pilot-tier status: not completed in this pass.\n")
        f.write("- Full-tier status: not completed in this pass.\n")
        f.write("- Blocker: compute and time cost of running the full multi-module battery after code creation inside the current turn.\n\n")
        f.write("## Commands Used\n\n")
        f.write("```powershell\n")
        f.write(f"python scripts/identity_boundary_sweep.py --config {args.config}\n")
        f.write(f"python scripts/longform_return.py --config {args.config}\n")
        f.write(f"python scripts/self_report_behavior.py --config {args.config}\n")
        f.write(f"python scripts/hidden_style_charter.py --config {args.config}\n")
        f.write(f"python scripts/ood_robustness.py --config {args.config}\n")
        f.write(f"python scripts/adaptive_baseline.py --config {args.config}\n")
        f.write(f"python scripts/analyze_identity_battery.py --config {args.config}\n")
        f.write("```\n\n")
        f.write("## Suggested Results Wording\n\n")
        f.write("- Keep all claims provisional until pilot-tier replication runs complete.\n")
        f.write("- Emphasize continuity and containment over identity unless the new battery remains coherent across prompt families and OOD variants.\n")
        f.write("- Treat self-report failures or hidden-charter inconsistency as publishable negative evidence rather than as a failure of the paper.\n\n")
        f.write("## Suggested Limitations Wording\n\n")
        f.write("- Smoke-tier results are sufficient for correctness and instrumentation checks, not final inferential claims.\n")
        f.write("- The current pass does not yet provide the full multi-seed pilot or cross-family full-tier replication required for stronger causal language.\n\n")
        f.write("## Candidate Titles\n\n")
        f.write("1. Patterned Continuity Without Strong Self-Model Coherence\n")
        f.write("2. Containment Under Pressure: Style Continuity and Resistance in Language Models\n")
        f.write("3. Identity Framing as a Weak Modulator of Steering Resistance\n")

    with (root / "manuscript_patch_notes.md").open("w", encoding="utf-8") as f:
        f.write("# Manuscript Patch Notes\n\n")
        f.write("## Introduction\n\n")
        f.write("- Keep the manuscript anchored on containment / resistance rather than personhood or a true-self claim.\n")
        f.write("- Add the identity bridge battery as a test of whether framing, hidden commitments, and long-form return mediate continuity.\n\n")
        f.write("## Identity Section\n\n")
        f.write("- Strengthen identity language only if long-form return, hidden-charter consistency, and OOD robustness converge.\n")
        f.write("- Otherwise revise identity language downward and describe continuity as patterned but weakly tied to explicit self-modeling.\n\n")
        f.write("## Methods\n\n")
        f.write("- Add the identity battery entrypoints, exact YAML assets, smoke-tier config, and reporting outputs under `outputs/latest/`.\n")
        f.write("- State explicitly that mean-difference, random orthogonal, and label-shuffled controls are analyzed separately.\n")
        f.write("- Describe the adaptive baseline as a lightweight prompt-conditional layer selector.\n\n")
        f.write("## Limitations And Conclusion\n\n")
        f.write("- Preserve the manuscript's rejection of a universal snap-back law.\n")
        f.write("- Note that smoke-tier outputs verify instrumentation and reporting but are not yet the final inferential basis for manuscript-strength claims.\n")
        f.write("- If pilot-tier results remain mixed, conclude that continuity can survive pressure without robust self-report coherence.\n\n")
        f.write("## Claim Triage\n\n")
        f.write("- Strengthen: containment / damping claims already supported by the baseline Pythia sweep.\n")
        f.write("- Weaken: any claim that recovery alone demonstrates concept-specific identity enforcement.\n")
        f.write("- Leave unchanged: the directional-only status of the GPT-2 / Qwen screen unless larger replication is run.\n")

    with (root / "stats_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)


if __name__ == "__main__":
    main()
