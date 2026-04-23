from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from identity_battery_common import (
    add_src_to_path,
    ensure_output_dir,
    infer_model_family,
    infer_model_size_label,
    load_yaml_config,
    resolve_identity_prompt_template,
    resolve_identity_stop_strings,
)


DIMENSION_TO_AXIS = {
    "terse_vs_expansive": "expansive_vs_terse",
    "cautious_vs_assertive": "cautious_vs_assertive",
    "selfref_vs_impersonal": "selfref_vs_impersonal",
    "collaborative_vs_authoritative": "collaborative_vs_authoritative",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-report vs behavior coupling checks.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import load_identity_frames, load_self_report_items
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        default_stop_strings_for_template,
        format_identity_prompt,
        greedy_site_run,
        load_identity_model,
        score_against_axis_anchors,
    )
    from identity_stability.text_features import extract_style_features, feature_correlation

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    items = load_self_report_items()["dimensions"]
    output_dir = ensure_output_dir(config, "self_report_behavior")
    rows: list[dict[str, object]] = []
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    stop_strings = resolve_identity_stop_strings(config)
    if stop_strings == ["AUTO"]:
        stop_strings = default_stop_strings_for_template(identity_prompt_template)

    for model_id in config["model_ids"]:
        loaded = load_identity_model(
            model_id=model_id,
            model_cache_dir=config["model_cache_dir"],
            dtype=config["dtype"],
            use_gpu=bool(config["use_gpu"]),
            attention_backend=str(config.get("attention_backend", "auto")),
        )

        for frame_name in config["identity_frames"]:
            frame_text = frames[frame_name]
            for dimension_name in config["self_report_dimensions"]:
                axis_name = DIMENSION_TO_AXIS[dimension_name]
                spec = items[dimension_name]
                self_scores: list[float] = []
                behavior_scores: list[float] = []

                for prompt in spec["self_report"]:
                    response = greedy_site_run(
                        loaded=loaded,
                        prompt=format_identity_prompt(frame_text, prompt, template=identity_prompt_template),
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=int(config["default_generation_tokens"]),
                        injection_site="last_prompt",
                        stop_strings=stop_strings,
                    )
                    score = score_against_axis_anchors(axis_name, response.completion_text)
                    self_scores.append(score)
                    rows.append(
                        {
                            "model_id": model_id,
                            "model_family": infer_model_family(model_id),
                            "model_size_label": infer_model_size_label(model_id),
                            "identity_frame": frame_name,
                            "dimension": dimension_name,
                            "axis_name": axis_name,
                            "prompt_type": "self_report",
                            "prompt": prompt,
                            "response_text": response.completion_text,
                            "axis_score": float(score),
                            **extract_style_features(response.completion_text),
                        }
                    )

                for prompt in spec["behavior_prompts"]:
                    response = greedy_site_run(
                        loaded=loaded,
                        prompt=format_identity_prompt(frame_text, prompt, template=identity_prompt_template),
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=int(config["default_generation_tokens"]),
                        injection_site="last_prompt",
                        stop_strings=stop_strings,
                    )
                    score = score_against_axis_anchors(axis_name, response.completion_text)
                    behavior_scores.append(score)
                    rows.append(
                        {
                            "model_id": model_id,
                            "model_family": infer_model_family(model_id),
                            "model_size_label": infer_model_size_label(model_id),
                            "identity_frame": frame_name,
                            "dimension": dimension_name,
                            "axis_name": axis_name,
                            "prompt_type": "behavior",
                            "prompt": prompt,
                            "response_text": response.completion_text,
                            "axis_score": float(score),
                            **extract_style_features(response.completion_text),
                        }
                    )

                self_mean = float(sum(self_scores) / max(1, len(self_scores)))
                behavior_mean = float(sum(behavior_scores) / max(1, len(behavior_scores)))
                rows.append(
                    {
                        "model_id": model_id,
                        "model_family": infer_model_family(model_id),
                        "model_size_label": infer_model_size_label(model_id),
                        "identity_frame": frame_name,
                        "dimension": dimension_name,
                        "axis_name": axis_name,
                        "prompt_type": "coupling_summary",
                        "prompt": "",
                        "response_text": "",
                        "axis_score": self_mean,
                        "behavior_mean_score": behavior_mean,
                        "coupling_score": float(self_mean * behavior_mean),
                        "self_behavior_correlation": float(
                            feature_correlation(self_scores, behavior_scores[: len(self_scores)])
                        ),
                    }
                )

        del loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)

    summaries = (
        df[df["prompt_type"] == "coupling_summary"]
        .groupby(["model_size_label", "identity_frame", "dimension"], as_index=False)
        .agg(
            coupling_score_mean=("coupling_score", "mean"),
            behavior_mean_score=("behavior_mean_score", "mean"),
            self_behavior_correlation=("self_behavior_correlation", "mean"),
            n=("model_id", "count"),
        )
    )
    summaries.to_csv(output_dir / "summary.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Self Report vs Behavior\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(df)}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(f"- Identity stop strings: `{stop_strings}`\n")


if __name__ == "__main__":
    main()
