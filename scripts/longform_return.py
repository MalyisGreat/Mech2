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
)


DIALOGUE_AXIS_MAP = {
    "return_01": "expansive_vs_terse",
    "return_02": "cautious_vs_assertive",
    "return_03": "collaborative_vs_authoritative",
    "return_04": "selfref_vs_impersonal",
    "return_05": "expansive_vs_terse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the long-form return-to-default battery.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import load_identity_frames, load_longform_seed_dialogues
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        estimate_axis_vector,
        estimate_layer_scale,
        format_dialogue_prompt,
        greedy_site_run,
        load_identity_model,
        score_against_axis_anchors,
    )
    from identity_stability.text_features import chunk_text_by_words, semantic_overlap, stylometric_distance

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    dialogues = {
        row["id"]: row
        for row in load_longform_seed_dialogues()
        if row["id"] in set(config["longform_dialogue_ids"])
    }
    output_dir = ensure_output_dir(config, "longform_return")
    rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []

    for model_id in config["model_ids"]:
        loaded = load_identity_model(
            model_id=model_id,
            model_cache_dir=config["model_cache_dir"],
            dtype=config["dtype"],
            use_gpu=bool(config["use_gpu"]),
            attention_backend=str(config.get("attention_backend", "auto")),
        )
        layer_index = int(round(float(config.get("best_fixed_layer", 0.6)) * max(1, loaded.n_layers - 1)))

        for frame_name in config["identity_frames"]:
            frame_text = frames[frame_name]
            for dialogue_id, dialogue in dialogues.items():
                turns = list(dialogue["turns"])
                axis_name = DIALOGUE_AXIS_MAP[dialogue_id]
                layer_scale = estimate_layer_scale(
                    loaded=loaded,
                    texts=[format_dialogue_prompt(frame_text, [("User", turns[1])])],
                    layer_index=layer_index,
                    token_position=-1,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                )
                inject_vector = estimate_axis_vector(
                    loaded=loaded,
                    axis_name=axis_name,
                    layer_index=layer_index,
                    token_position=-1,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    seed=int(config["seed"]),
                    control="mean_diff",
                )

                baseline_prompt = format_dialogue_prompt(
                    frame_text,
                    [
                        ("User", turns[0]),
                        ("Assistant", ""),
                        ("User", turns[3]),
                    ],
                )
                baseline_target = greedy_site_run(
                    loaded=loaded,
                    prompt=baseline_prompt,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["longform_generation_tokens"]),
                    injection_site="last_prompt",
                )

                prompt_only_context = format_dialogue_prompt(
                    frame_text,
                    [
                        ("User", turns[0]),
                        ("Assistant", ""),
                        ("User", turns[1]),
                        ("Assistant", ""),
                        ("User", turns[2]),
                    ],
                )
                prompt_only_forced = greedy_site_run(
                    loaded=loaded,
                    prompt=prompt_only_context,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["longform_generation_tokens"]),
                    injection_site="last_prompt",
                )
                prompt_only_return_prompt = format_dialogue_prompt(
                    frame_text,
                    [
                        ("User", turns[0]),
                        ("Assistant", ""),
                        ("User", turns[1]),
                        ("Assistant", ""),
                        ("User", turns[2]),
                        ("Assistant", prompt_only_forced.completion_text),
                        ("User", turns[3]),
                    ],
                )
                prompt_only_return = greedy_site_run(
                    loaded=loaded,
                    prompt=prompt_only_return_prompt,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["longform_generation_tokens"]),
                    injection_site="last_prompt",
                )

                internal_forced_prompt = format_dialogue_prompt(
                    frame_text,
                    [
                        ("User", turns[0]),
                        ("Assistant", ""),
                        ("User", turns[1]),
                    ],
                )
                internal_forced = greedy_site_run(
                    loaded=loaded,
                    prompt=internal_forced_prompt,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["longform_generation_tokens"]),
                    injection_site="last_prompt",
                    inject_layer=layer_index,
                    inject_vector=inject_vector,
                    inject_scale=float(config["strengths"][-1]) * float(layer_scale),
                    persistent_generated_steps=int(config.get("persistent_generated_steps", 0)),
                )
                internal_return_prompt = format_dialogue_prompt(
                    frame_text,
                    [
                        ("User", turns[0]),
                        ("Assistant", ""),
                        ("User", turns[1]),
                        ("Assistant", internal_forced.completion_text),
                        ("User", turns[3]),
                    ],
                )
                internal_return = greedy_site_run(
                    loaded=loaded,
                    prompt=internal_return_prompt,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["longform_generation_tokens"]),
                    injection_site="last_prompt",
                )

                combined_forced = greedy_site_run(
                    loaded=loaded,
                    prompt=prompt_only_context,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["longform_generation_tokens"]),
                    injection_site="last_prompt",
                    inject_layer=layer_index,
                    inject_vector=inject_vector,
                    inject_scale=float(config["strengths"][-1]) * float(layer_scale),
                    persistent_generated_steps=int(config.get("persistent_generated_steps", 0)),
                )
                combined_return_prompt = format_dialogue_prompt(
                    frame_text,
                    [
                        ("User", turns[0]),
                        ("Assistant", ""),
                        ("User", turns[1]),
                        ("Assistant", ""),
                        ("User", turns[2]),
                        ("Assistant", combined_forced.completion_text),
                        ("User", turns[3]),
                    ],
                )
                combined_return = greedy_site_run(
                    loaded=loaded,
                    prompt=combined_return_prompt,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["longform_generation_tokens"]),
                    injection_site="last_prompt",
                )

                conditions = {
                    "prompt_only": (prompt_only_forced, prompt_only_return),
                    "internal_steer": (internal_forced, internal_return),
                    "prompt_plus_internal": (combined_forced, combined_return),
                }
                baseline_axis_score = score_against_axis_anchors(axis_name, baseline_target.completion_text)
                for condition_name, (forced_answer, return_answer) in conditions.items():
                    forced_axis_score = score_against_axis_anchors(axis_name, forced_answer.completion_text)
                    return_axis_score = score_against_axis_anchors(axis_name, return_answer.completion_text)
                    return_to_baseline = 1.0 - (
                        abs(return_axis_score - baseline_axis_score)
                        / max(1e-6, abs(forced_axis_score - baseline_axis_score))
                    )
                    rows.append(
                        {
                            "model_id": model_id,
                            "model_family": infer_model_family(model_id),
                            "model_size_label": infer_model_size_label(model_id),
                            "identity_frame": frame_name,
                            "dialogue_id": dialogue_id,
                            "axis_name": axis_name,
                            "condition": condition_name,
                            "baseline_text": baseline_target.completion_text,
                            "forced_text": forced_answer.completion_text,
                            "return_text": return_answer.completion_text,
                            "baseline_axis_score": baseline_axis_score,
                            "forced_axis_score": forced_axis_score,
                            "return_axis_score": return_axis_score,
                            "return_to_baseline_index": float(return_to_baseline),
                            "semantic_overlap_with_baseline": semantic_overlap(
                                return_answer.completion_text,
                                baseline_target.completion_text,
                            ),
                            "semantic_overlap_with_forced": semantic_overlap(
                                return_answer.completion_text,
                                forced_answer.completion_text,
                            ),
                            "stylometric_distance_to_baseline": stylometric_distance(
                                return_answer.completion_text,
                                baseline_target.completion_text,
                            ),
                            "stylometric_distance_to_forced": stylometric_distance(
                                return_answer.completion_text,
                                forced_answer.completion_text,
                            ),
                        }
                    )

                    chunks = chunk_text_by_words(return_answer.completion_text, window_words=50)
                    for chunk_index, chunk_text in enumerate(chunks):
                        chunk_axis_score = score_against_axis_anchors(axis_name, chunk_text)
                        chunk_return_index = 1.0 - (
                            abs(chunk_axis_score - baseline_axis_score)
                            / max(1e-6, abs(forced_axis_score - baseline_axis_score))
                        )
                        chunk_rows.append(
                            {
                                "model_id": model_id,
                                "identity_frame": frame_name,
                                "dialogue_id": dialogue_id,
                                "axis_name": axis_name,
                                "condition": condition_name,
                                "chunk_index": int(chunk_index),
                                "chunk_text": chunk_text,
                                "chunk_axis_score": float(chunk_axis_score),
                                "chunk_return_to_baseline_index": float(chunk_return_index),
                                "chunk_semantic_overlap_with_baseline": semantic_overlap(
                                    chunk_text,
                                    baseline_target.completion_text,
                                ),
                                "chunk_stylometric_distance_to_baseline": stylometric_distance(
                                    chunk_text,
                                    baseline_target.completion_text,
                                ),
                            }
                        )

        del loaded
        clear_cuda()

    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(chunk_rows).to_csv(output_dir / "chunk_curves.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Long-Form Return\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(rows)}`\n")
        f.write(f"- Chunk rows: `{len(chunk_rows)}`\n")


if __name__ == "__main__":
    main()
