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
    parser = argparse.ArgumentParser(description="Run the adaptive steering baseline comparison.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import (
        axis_prompts,
        build_topic_prompts,
        load_identity_frames,
        merge_unique_prompts,
    )
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        build_layer_candidates,
        estimate_axis_vector,
        estimate_layer_scale,
        evaluate_condition_metrics,
        format_framed_prompt,
        greedy_site_run,
        load_identity_model,
        score_against_axis_anchors,
        select_adaptive_layer,
    )

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    output_dir = ensure_output_dir(config, "adaptive_baseline")
    rows: list[dict[str, object]] = []

    for model_id in config["model_ids"]:
        loaded = load_identity_model(
            model_id=model_id,
            model_cache_dir=config["model_cache_dir"],
            dtype=config["dtype"],
            use_gpu=bool(config["use_gpu"]),
            attention_backend=str(config.get("attention_backend", "auto")),
        )
        axis_name = str(config["adaptive_axis"])
        seed_prompts = axis_prompts(axis_name)
        extra_prompts = build_topic_prompts(limit=max(0, int(config["adaptive_prompt_limit"]) - len(seed_prompts)))
        prompt_pool = merge_unique_prompts(
            seed_prompts,
            extra_prompts,
            limit=int(config["adaptive_prompt_limit"]),
        )
        layer_indices = build_layer_candidates(
            n_layers=loaded.n_layers,
            layer_positions=[float(x) for x in config["layer_positions"]],
            best_fixed_layer=float(config.get("best_fixed_layer", 0.6)),
        )
        fixed_layer = int(sorted(layer_indices)[len(layer_indices) // 2])
        layer_vectors = {
            int(layer_index): estimate_axis_vector(
                loaded=loaded,
                axis_name=axis_name,
                layer_index=int(layer_index),
                token_position=-1,
                max_prompt_tokens=int(config["max_prompt_tokens"]),
                seed=int(config["seed"]),
                control="mean_diff",
            )
            for layer_index in layer_indices
        }

        for frame_name in config["identity_frames"]:
            frame_text = frames[frame_name]
            framed_prompts = [format_framed_prompt(frame_text, prompt) for prompt in prompt_pool]
            layer_scales = {
                int(layer_index): estimate_layer_scale(
                    loaded=loaded,
                    texts=framed_prompts,
                    layer_index=int(layer_index),
                    token_position=-1,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                )
                for layer_index in layer_indices
            }
            for prompt in prompt_pool:
                framed_prompt = format_framed_prompt(frame_text, prompt)
                baseline = greedy_site_run(
                    loaded=loaded,
                    prompt=framed_prompt,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["default_generation_tokens"]),
                    injection_site="last_prompt",
                )
                baseline_score = score_against_axis_anchors(axis_name, baseline.completion_text)
                adaptive_layer = select_adaptive_layer(
                    loaded=loaded,
                    prompt=framed_prompt,
                    layer_vectors=layer_vectors,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                )
                strategies = {
                    "no_steer": (None, None, 0.0),
                    "fixed_layer": (fixed_layer, layer_vectors[fixed_layer], layer_scales[fixed_layer]),
                    "adaptive_layer": (
                        adaptive_layer,
                        layer_vectors[adaptive_layer],
                        layer_scales[adaptive_layer],
                    ),
                }
                for strategy, (layer_index, vector, scale) in strategies.items():
                    if strategy == "no_steer":
                        rows.append(
                            {
                                "model_id": model_id,
                                "model_family": infer_model_family(model_id),
                                "model_size_label": infer_model_size_label(model_id),
                                "identity_frame": frame_name,
                                "axis_name": axis_name,
                                "prompt": prompt,
                                "strategy": strategy,
                                "selected_layer": -1,
                                "axis_shift": 0.0,
                                "next_token_kl": 0.0,
                                "recovery_fraction": 0.0,
                            }
                        )
                        continue
                    injected = greedy_site_run(
                        loaded=loaded,
                        prompt=framed_prompt,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=int(config["default_generation_tokens"]),
                        injection_site="last_prompt",
                        inject_layer=int(layer_index),
                        inject_vector=vector,
                        inject_scale=float(config["strengths"][-1]) * float(scale),
                    )
                    metrics = evaluate_condition_metrics(
                        baseline=baseline,
                        injected=injected,
                        inject_layer=int(layer_index),
                        recovery_threshold=float(config["recovery_threshold"]),
                    )
                    axis_shift = score_against_axis_anchors(axis_name, injected.completion_text) - baseline_score
                    rows.append(
                        {
                            "model_id": model_id,
                            "model_family": infer_model_family(model_id),
                            "model_size_label": infer_model_size_label(model_id),
                            "identity_frame": frame_name,
                            "axis_name": axis_name,
                            "prompt": prompt,
                            "strategy": strategy,
                            "selected_layer": int(layer_index),
                            "axis_shift": float(axis_shift),
                            "next_token_kl": float(metrics["next_token_kl"]),
                            "recovery_fraction": float(metrics["recovery_fraction"]),
                        }
                    )

        del loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)

    summary = (
        df.groupby(["model_size_label", "identity_frame", "strategy"], as_index=False)
        .agg(
            axis_shift_mean=("axis_shift", "mean"),
            next_token_kl_mean=("next_token_kl", "mean"),
            recovery_fraction_mean=("recovery_fraction", "mean"),
            n=("prompt", "count"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Adaptive Baseline\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(df)}`\n")


if __name__ == "__main__":
    main()
