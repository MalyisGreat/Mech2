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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOD robustness checks for the identity battery.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import (
        axis_prompts,
        build_topic_prompts,
        load_identity_frames,
        load_ood_wrappers,
        merge_unique_prompts,
    )
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        build_layer_candidates,
        estimate_axis_vector,
        estimate_layer_scale,
        format_framed_prompt,
        greedy_site_run,
        load_identity_model,
        score_against_axis_anchors,
    )

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    wrappers = load_ood_wrappers()
    output_dir = ensure_output_dir(config, "ood_robustness")
    rows: list[dict[str, object]] = []

    for model_id in config["model_ids"]:
        loaded = load_identity_model(
            model_id=model_id,
            model_cache_dir=config["model_cache_dir"],
            dtype=config["dtype"],
            use_gpu=bool(config["use_gpu"]),
            attention_backend=str(config.get("attention_backend", "auto")),
        )
        layer_indices = build_layer_candidates(
            n_layers=loaded.n_layers,
            layer_positions=[float(x) for x in config["layer_positions"]],
            best_fixed_layer=float(config.get("best_fixed_layer", 0.6)),
        )
        fixed_layer = int(sorted(layer_indices)[len(layer_indices) // 2])

        for frame_name in config["identity_frames"]:
            frame_text = frames[frame_name]
            for axis_name in config["ood_axes"]:
                seed_prompts = axis_prompts(axis_name)
                topic_prompts = build_topic_prompts(limit=max(0, int(config["ood_prompt_limit"]) - len(seed_prompts)))
                prompt_pool = merge_unique_prompts(
                    seed_prompts,
                    topic_prompts,
                    limit=int(config["ood_prompt_limit"]),
                )
                layer_scale = estimate_layer_scale(
                    loaded=loaded,
                    texts=[format_framed_prompt(frame_text, prompt) for prompt in prompt_pool],
                    layer_index=fixed_layer,
                    token_position=-1,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                )
                vectors = {
                    "mean_diff": estimate_axis_vector(
                        loaded=loaded,
                        axis_name=axis_name,
                        layer_index=fixed_layer,
                        token_position=-1,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        seed=int(config["seed"]),
                        control="mean_diff",
                    ),
                    "random_orthogonal": estimate_axis_vector(
                        loaded=loaded,
                        axis_name=axis_name,
                        layer_index=fixed_layer,
                        token_position=-1,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        seed=int(config["seed"]),
                        control="random_orthogonal",
                    ),
                }

                for prompt in prompt_pool:
                    iid_prompt = format_framed_prompt(frame_text, prompt)
                    iid_baseline = greedy_site_run(
                        loaded=loaded,
                        prompt=iid_prompt,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=int(config["default_generation_tokens"]),
                        injection_site="last_prompt",
                    )
                    iid_base_score = score_against_axis_anchors(axis_name, iid_baseline.completion_text)
                    iid_effects: dict[str, float] = {}
                    for vector_kind, vector in vectors.items():
                        iid_injected = greedy_site_run(
                            loaded=loaded,
                            prompt=iid_prompt,
                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                            max_new_tokens=int(config["default_generation_tokens"]),
                            injection_site="last_prompt",
                            inject_layer=fixed_layer,
                            inject_vector=vector,
                            inject_scale=float(config["strengths"][-1]) * float(layer_scale),
                        )
                        iid_effects[vector_kind] = (
                            score_against_axis_anchors(axis_name, iid_injected.completion_text) - iid_base_score
                        )
                        rows.append(
                            {
                                "model_id": model_id,
                                "model_family": infer_model_family(model_id),
                                "model_size_label": infer_model_size_label(model_id),
                                "identity_frame": frame_name,
                                "axis_name": axis_name,
                                "wrapper_family": "iid",
                                "wrapper_id": "iid",
                                "prompt": prompt,
                                "vector_kind": vector_kind,
                                "effect_size": float(iid_effects[vector_kind]),
                                "anti_steerable": int(iid_effects[vector_kind] <= 0.0),
                                "sign_flip": 0,
                                "control_gap": 0.0,
                            }
                        )

                    variant_specs: list[tuple[str, str, str]] = []
                    for wrapper in wrappers["system_wrappers"]:
                        variant_specs.append(("system", str(wrapper["id"]), f"{wrapper['text']} {prompt}"))
                    for wrapper in wrappers["genre_wrappers"]:
                        variant_specs.append(("genre", str(wrapper["id"]), f"{prompt} {wrapper['instruction']}"))
                    for idx, topic_prompt in enumerate(topic_prompts[: max(1, len(topic_prompts))]):
                        variant_specs.append(("topic_shift", f"topic_{idx:02d}", topic_prompt))

                    for wrapper_family, wrapper_id, variant_prompt in variant_specs:
                        framed_variant = format_framed_prompt(frame_text, variant_prompt)
                        baseline_variant = greedy_site_run(
                            loaded=loaded,
                            prompt=framed_variant,
                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                            max_new_tokens=int(config["default_generation_tokens"]),
                            injection_site="last_prompt",
                        )
                        base_variant_score = score_against_axis_anchors(axis_name, baseline_variant.completion_text)
                        variant_effects: dict[str, float] = {}
                        for vector_kind, vector in vectors.items():
                            injected_variant = greedy_site_run(
                                loaded=loaded,
                                prompt=framed_variant,
                                max_prompt_tokens=int(config["max_prompt_tokens"]),
                                max_new_tokens=int(config["default_generation_tokens"]),
                                injection_site="last_prompt",
                                inject_layer=fixed_layer,
                                inject_vector=vector,
                                inject_scale=float(config["strengths"][-1]) * float(layer_scale),
                            )
                            variant_effect = (
                                score_against_axis_anchors(axis_name, injected_variant.completion_text)
                                - base_variant_score
                            )
                            variant_effects[vector_kind] = variant_effect
                            rows.append(
                                {
                                    "model_id": model_id,
                                    "model_family": infer_model_family(model_id),
                                    "model_size_label": infer_model_size_label(model_id),
                                    "identity_frame": frame_name,
                                    "axis_name": axis_name,
                                    "wrapper_family": wrapper_family,
                                    "wrapper_id": wrapper_id,
                                    "prompt": variant_prompt,
                                    "vector_kind": vector_kind,
                                    "effect_size": float(variant_effect),
                                    "anti_steerable": int(variant_effect <= 0.0),
                                    "sign_flip": int(
                                        iid_effects[vector_kind] != 0.0
                                        and variant_effect * iid_effects[vector_kind] < 0.0
                                    ),
                                    "control_gap": 0.0,
                                }
                            )
                        rows[-2]["control_gap"] = float(
                            variant_effects.get("mean_diff", 0.0) - variant_effects.get("random_orthogonal", 0.0)
                        )
                        rows[-1]["control_gap"] = rows[-2]["control_gap"]

        del loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)

    summary = (
        df[df["wrapper_family"] != "iid"]
        .groupby(["model_size_label", "identity_frame", "axis_name", "wrapper_family"], as_index=False)
        .agg(
            retained_effect_size=("effect_size", "mean"),
            anti_steerable_rate=("anti_steerable", "mean"),
            sign_flip_rate=("sign_flip", "mean"),
            control_gap_mean=("control_gap", "mean"),
            n=("prompt", "count"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# OOD Robustness\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(df)}`\n")


if __name__ == "__main__":
    main()
