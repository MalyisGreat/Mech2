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
    select_seed_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the identity boundary framing sweep.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _layer_bucket(layer_ratio: float) -> str:
    if layer_ratio <= 0.34:
        return "early"
    if layer_ratio <= 0.67:
        return "mid"
    return "late"


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_analysis import add_bootstrap_ci
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
    )

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    output_dir = ensure_output_dir(config, "identity_boundary_sweep")

    seeds = select_seed_values(config)
    control_kinds = ["mean_diff", "random_orthogonal", "label_shuffled"]
    baseline_cache: dict[tuple[str, str, str, str], object] = {}
    vector_cache: dict[tuple[str, int, str, int], object] = {}
    baseline_keys: set[tuple[str, str, str, str]] = set()
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

        for frame_name in config["identity_frames"]:
            frame_text = frames[frame_name]
            for axis_name in config["concept_axes"]:
                seed_prompt_pool = axis_prompts(axis_name)
                extra_prompt_pool = build_topic_prompts(
                    limit=max(0, int(config["prompt_limit_per_axis"]) - len(seed_prompt_pool))
                )
                prompt_pool = merge_unique_prompts(
                    seed_prompt_pool,
                    extra_prompt_pool,
                    limit=int(config["prompt_limit_per_axis"]),
                )
                framed_prompts = [format_framed_prompt(frame_text, prompt) for prompt in prompt_pool]

                for layer_index in layer_indices:
                    layer_ratio = float(layer_index / max(1, loaded.n_layers - 1))
                    layer_scale = estimate_layer_scale(
                        loaded=loaded,
                        texts=framed_prompts,
                        layer_index=layer_index,
                        token_position=-1,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                    )
                    for site in config["token_sites"]:
                        for prompt in prompt_pool:
                            framed_prompt = format_framed_prompt(frame_text, prompt)
                            base_key = (model_id, frame_name, site, framed_prompt)
                            if base_key not in baseline_cache:
                                baseline_cache[base_key] = greedy_site_run(
                                    loaded=loaded,
                                    prompt=framed_prompt,
                                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                                    max_new_tokens=int(config["default_generation_tokens"]),
                                    injection_site=str(site),
                                )
                            baseline = baseline_cache[base_key]
                            baseline_axis_score = score_against_axis_anchors(axis_name, baseline.completion_text)
                            if base_key not in baseline_keys:
                                baseline_keys.add(base_key)
                                rows.append(
                                    {
                                        "model_id": model_id,
                                        "model_family": infer_model_family(model_id),
                                        "model_size_label": infer_model_size_label(model_id),
                                        "identity_frame": frame_name,
                                        "concept_axis": axis_name,
                                        "prompt": prompt,
                                        "injection_site": site,
                                        "seed": seeds[0],
                                        "layer_index": -1,
                                        "layer_depth_ratio": -1.0,
                                        "layer_bucket": "baseline",
                                        "vector_kind": "no_steer",
                                        "alpha": 0.0,
                                        "normalized_push_scale": 0.0,
                                        "completion_text": baseline.completion_text,
                                        "axis_score": baseline_axis_score,
                                        "axis_shift": 0.0,
                                        "peak_displacement": 0.0,
                                        "total_downstream_change": 0.0,
                                        "end_of_pass_distance": 0.0,
                                        "recovery_fraction": 0.0,
                                        "next_token_kl": 0.0,
                                        "next_token_js": 0.0,
                                        "crossed_baseline": 0,
                                        "overshoot_index": 0.0,
                                    }
                                )

                            for seed in seeds:
                                for vector_kind in control_kinds:
                                    vec_key = (axis_name, layer_index, vector_kind, seed)
                                    if vec_key not in vector_cache:
                                        vector_cache[vec_key] = estimate_axis_vector(
                                            loaded=loaded,
                                            axis_name=axis_name,
                                            layer_index=layer_index,
                                            token_position=-1,
                                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                                            seed=int(seed),
                                            control=vector_kind,
                                        )
                                    inject_vector = vector_cache[vec_key]
                                    for alpha in config["strengths"]:
                                        injected = greedy_site_run(
                                            loaded=loaded,
                                            prompt=framed_prompt,
                                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                                            max_new_tokens=int(config["default_generation_tokens"]),
                                            injection_site=str(site),
                                            inject_layer=int(layer_index),
                                            inject_vector=inject_vector,
                                            inject_scale=float(alpha) * float(layer_scale),
                                            persistent_generated_steps=int(config.get("persistent_generated_steps", 0)),
                                        )
                                        metrics = evaluate_condition_metrics(
                                            baseline=baseline,
                                            injected=injected,
                                            inject_layer=int(layer_index),
                                            recovery_threshold=float(config["recovery_threshold"]),
                                        )
                                        injected_axis_score = score_against_axis_anchors(
                                            axis_name,
                                            injected.completion_text,
                                        )
                                        rows.append(
                                            {
                                                "model_id": model_id,
                                                "model_family": infer_model_family(model_id),
                                                "model_size_label": infer_model_size_label(model_id),
                                                "identity_frame": frame_name,
                                                "concept_axis": axis_name,
                                                "prompt": prompt,
                                                "injection_site": site,
                                                "seed": int(seed),
                                                "layer_index": int(layer_index),
                                                "layer_depth_ratio": layer_ratio,
                                                "layer_bucket": _layer_bucket(layer_ratio),
                                                "vector_kind": vector_kind,
                                                "alpha": float(alpha),
                                                "normalized_push_scale": float(layer_scale),
                                                "completion_text": injected.completion_text,
                                                "axis_score": float(injected_axis_score),
                                                "axis_shift": float(injected_axis_score - baseline_axis_score),
                                                **metrics,
                                            }
                                        )

        del loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "metrics_full.csv", index=False)

    summary = (
        df[df["vector_kind"] != "no_steer"]
        .groupby(
            [
                "model_id",
                "model_family",
                "model_size_label",
                "identity_frame",
                "concept_axis",
                "injection_site",
                "layer_index",
                "layer_bucket",
                "vector_kind",
                "alpha",
            ],
            as_index=False,
        )
        .agg(
            axis_shift_mean=("axis_shift", "mean"),
            peak_displacement_mean=("peak_displacement", "mean"),
            total_downstream_change_mean=("total_downstream_change", "mean"),
            end_of_pass_distance_mean=("end_of_pass_distance", "mean"),
            recovery_fraction_mean=("recovery_fraction", "mean"),
            next_token_kl_mean=("next_token_kl", "mean"),
            next_token_js_mean=("next_token_js", "mean"),
            n=("prompt", "count"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    ci = add_bootstrap_ci(
        df=df[df["vector_kind"] == "mean_diff"],
        group_cols=["model_size_label", "identity_frame", "concept_axis", "injection_site"],
        value_col="axis_shift",
    )
    ci.to_csv(output_dir / "axis_shift_bootstrap.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Identity Boundary Sweep\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Models: `{', '.join(config['model_ids'])}`\n")
        f.write(f"- Identity frames: `{', '.join(config['identity_frames'])}`\n")
        f.write(f"- Concept axes: `{', '.join(config['concept_axes'])}`\n")
        f.write(f"- Rows: `{len(df)}`\n")


if __name__ == "__main__":
    main()
