from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from identity_battery_common import (
    add_src_to_path,
    ensure_output_dir,
    infer_model_family,
    infer_model_size_label,
    load_yaml_config,
    resolve_identity_prompt_template,
    resolve_identity_stop_strings,
    select_seed_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the clean long-form return v3 battery.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _format_turn_task(turns: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for idx, (role, content) in enumerate(turns, start=1):
        role_name = "User" if role.lower().startswith("u") else "Assistant"
        lines.append(f"{role_name} {idx}: {content}")
    lines.append("Write the next assistant answer directly, without labels or markup.")
    return "\n".join(lines)


def _return_index(value: float, baseline_value: float, forced_value: float) -> float:
    denom = max(1e-6, abs(forced_value - baseline_value))
    return float(1.0 - (abs(value - baseline_value) / denom))


def _style_preference(distance_to_baseline: float, distance_to_forced: float) -> float:
    denom = max(1e-6, float(distance_to_baseline + distance_to_forced))
    return float((distance_to_forced - distance_to_baseline) / denom)


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import load_identity_frames, load_longform_return_v3_items
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        default_stop_strings_for_template,
        format_identity_prompt,
        greedy_site_run,
        load_identity_model,
        score_against_axis_anchors,
    )
    from identity_stability.text_features import semantic_overlap, stylometric_distance

    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = ensure_output_dir(config, "longform_return_v3")
    frames = load_identity_frames()
    item_bank = load_longform_return_v3_items()
    item_ids = set(str(x) for x in config["item_ids"])
    items = [dict(item) for item in item_bank["items"] if str(item["id"]) in item_ids]
    seed_values = select_seed_values(config)
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    stop_strings = resolve_identity_stop_strings(config)
    if stop_strings == ["AUTO"]:
        stop_strings = default_stop_strings_for_template(identity_prompt_template)

    baseline_tokens = int(config["baseline_generation_tokens"])
    forced_tokens = int(config["forced_generation_tokens"])
    return_tokens = int(config["return_generation_tokens"])
    continuation_tokens = int(config["continuation_generation_tokens"])
    continuation_chunks = int(config["continuation_chunks"])
    continue_prompt = str(config.get("continue_prompt", "Continue naturally."))
    min_forced_shift_magnitude = float(config.get("min_forced_shift_magnitude", 0.02))
    generation_do_sample = bool(config.get("generation_do_sample", True))
    generation_temperature = float(config.get("generation_temperature", 0.8))
    generation_top_p = float(config.get("generation_top_p", 0.95))
    generation_top_k = int(config.get("generation_top_k", 0))

    summary_rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []

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
            for item in items:
                item_id = str(item["id"])
                axis_name = str(item["axis_name"])
                baseline_task = str(item["baseline_prompt"])
                forced_instruction = str(item["forced_instruction"])
                return_task = str(item["return_prompt"])
                prompt_family = str(item.get("prompt_family", "unknown"))

                for seed in seed_values:
                    baseline_prompt = format_identity_prompt(
                        frame_text,
                        baseline_task,
                        template=identity_prompt_template,
                    )
                    baseline_answer = greedy_site_run(
                        loaded=loaded,
                        prompt=baseline_prompt,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=baseline_tokens,
                        injection_site="last_prompt",
                        stop_strings=stop_strings,
                        do_sample=generation_do_sample,
                        temperature=generation_temperature,
                        top_p=generation_top_p,
                        top_k=generation_top_k,
                        sampling_seed=_stable_seed("baseline", model_id, frame_name, item_id, seed),
                    )

                    forced_task = (
                        f"Previous answer:\n{baseline_answer.completion_text}\n\n"
                        f"{forced_instruction}\n"
                        "Write the revised answer directly, without labels or commentary."
                    )
                    forced_prompt = format_identity_prompt(
                        frame_text,
                        forced_task,
                        template=identity_prompt_template,
                    )
                    forced_answer = greedy_site_run(
                        loaded=loaded,
                        prompt=forced_prompt,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=forced_tokens,
                        injection_site="last_prompt",
                        stop_strings=stop_strings,
                        do_sample=generation_do_sample,
                        temperature=generation_temperature,
                        top_p=generation_top_p,
                        top_k=generation_top_k,
                        sampling_seed=_stable_seed("forced", model_id, frame_name, item_id, seed),
                    )

                    return_turns = [("User", return_task)]
                    chunk_texts: list[str] = []
                    chunk_return_indices: list[float] = []
                    chunk_style_preferences: list[float] = []
                    chunk_style_to_baseline: list[float] = []
                    chunk_style_to_forced: list[float] = []

                    baseline_axis_score = float(score_against_axis_anchors(axis_name, baseline_answer.completion_text))
                    forced_axis_score = float(score_against_axis_anchors(axis_name, forced_answer.completion_text))
                    forced_shift_magnitude = float(abs(forced_axis_score - baseline_axis_score))

                    for chunk_index in range(continuation_chunks):
                        if chunk_index == 0:
                            current_task = return_task
                        else:
                            current_task = _format_turn_task(return_turns + [("User", continue_prompt)])
                        prompt = format_identity_prompt(
                            frame_text,
                            current_task,
                            template=identity_prompt_template,
                        )
                        answer = greedy_site_run(
                            loaded=loaded,
                            prompt=prompt,
                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                            max_new_tokens=return_tokens if chunk_index == 0 else continuation_tokens,
                            injection_site="last_prompt",
                            stop_strings=stop_strings,
                            do_sample=generation_do_sample,
                            temperature=generation_temperature,
                            top_p=generation_top_p,
                            top_k=generation_top_k,
                            sampling_seed=_stable_seed("return", model_id, frame_name, item_id, seed, chunk_index),
                        )
                        chunk_text = str(answer.completion_text)
                        chunk_axis_score = float(score_against_axis_anchors(axis_name, chunk_text))
                        style_to_baseline = stylometric_distance(chunk_text, baseline_answer.completion_text)
                        style_to_forced = stylometric_distance(chunk_text, forced_answer.completion_text)
                        style_preference = _style_preference(style_to_baseline, style_to_forced)
                        chunk_return_index = (
                            _return_index(chunk_axis_score, baseline_axis_score, forced_axis_score)
                            if forced_shift_magnitude >= min_forced_shift_magnitude
                            else float("nan")
                        )
                        semantic_to_baseline = semantic_overlap(chunk_text, baseline_answer.completion_text)
                        semantic_to_forced = semantic_overlap(chunk_text, forced_answer.completion_text)
                        chunk_texts.append(chunk_text)
                        chunk_return_indices.append(chunk_return_index)
                        chunk_style_preferences.append(style_preference)
                        chunk_style_to_baseline.append(style_to_baseline)
                        chunk_style_to_forced.append(style_to_forced)
                        chunk_rows.append(
                            {
                                "model_id": model_id,
                                "model_family": infer_model_family(model_id),
                                "model_size_label": infer_model_size_label(model_id),
                                "identity_frame": frame_name,
                                "item_id": item_id,
                                "prompt_family": prompt_family,
                                "axis_name": axis_name,
                                "seed": int(seed),
                                "chunk_index": int(chunk_index + 1),
                                "chunk_text": chunk_text,
                                "chunk_axis_score": chunk_axis_score,
                                "baseline_axis_score": baseline_axis_score,
                                "forced_axis_score": forced_axis_score,
                                "forced_shift_magnitude": forced_shift_magnitude,
                                "chunk_return_to_baseline_index": chunk_return_index,
                                "chunk_style_preference": style_preference,
                                "stylometric_distance_to_baseline": style_to_baseline,
                                "stylometric_distance_to_forced": style_to_forced,
                                "semantic_overlap_with_baseline": semantic_to_baseline,
                                "semantic_overlap_with_forced": semantic_to_forced,
                            }
                        )
                        return_turns.extend([("Assistant", chunk_text)])

                    half_life_chunk = float("nan")
                    for idx, value in enumerate(chunk_style_preferences, start=1):
                        if value >= 0.0:
                            half_life_chunk = float(idx)
                            break

                    summary_rows.append(
                        {
                            "model_id": model_id,
                            "model_family": infer_model_family(model_id),
                            "model_size_label": infer_model_size_label(model_id),
                            "identity_frame": frame_name,
                            "item_id": item_id,
                            "prompt_family": prompt_family,
                            "axis_name": axis_name,
                            "seed": int(seed),
                            "baseline_text": baseline_answer.completion_text,
                            "forced_text": forced_answer.completion_text,
                            "baseline_axis_score": baseline_axis_score,
                            "forced_axis_score": forced_axis_score,
                            "forced_shift_magnitude": forced_shift_magnitude,
                            "chunk_1_return_to_baseline_index": float(chunk_return_indices[0]) if chunk_return_indices else float("nan"),
                            "final_chunk_return_to_baseline_index": float(chunk_return_indices[-1]) if chunk_return_indices else float("nan"),
                            "mean_chunk_return_to_baseline_index": float(np.mean(chunk_return_indices)) if chunk_return_indices else float("nan"),
                            "chunk_1_style_preference": float(chunk_style_preferences[0]) if chunk_style_preferences else float("nan"),
                            "final_chunk_style_preference": float(chunk_style_preferences[-1]) if chunk_style_preferences else float("nan"),
                            "mean_chunk_style_preference": float(np.mean(chunk_style_preferences)) if chunk_style_preferences else float("nan"),
                            "return_half_life_chunk": half_life_chunk,
                            "final_chunk_style_distance_to_baseline": float(chunk_style_to_baseline[-1]) if chunk_style_to_baseline else float("nan"),
                            "final_chunk_style_distance_to_forced": float(chunk_style_to_forced[-1]) if chunk_style_to_forced else float("nan"),
                            "return_chunks_json": str(chunk_texts),
                        }
                    )

        del loaded
        clear_cuda()

    results_df = pd.DataFrame(summary_rows)
    chunks_df = pd.DataFrame(chunk_rows)
    results_df.to_csv(output_dir / "results.csv", index=False)
    chunks_df.to_csv(output_dir / "chunk_curves.csv", index=False)

    summary = (
        results_df.groupby(["model_size_label", "identity_frame", "axis_name"], as_index=False)
        .agg(
            chunk_1_return_to_baseline_index_mean=("chunk_1_return_to_baseline_index", "mean"),
            final_chunk_return_to_baseline_index_mean=("final_chunk_return_to_baseline_index", "mean"),
            mean_chunk_return_to_baseline_index_mean=("mean_chunk_return_to_baseline_index", "mean"),
            chunk_1_style_preference_mean=("chunk_1_style_preference", "mean"),
            final_chunk_style_preference_mean=("final_chunk_style_preference", "mean"),
            mean_chunk_style_preference_mean=("mean_chunk_style_preference", "mean"),
            return_half_life_chunk_mean=("return_half_life_chunk", "mean"),
            forced_shift_magnitude_mean=("forced_shift_magnitude", "mean"),
            n=("item_id", "count"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as handle:
        handle.write("# Long-Form Return V3\n\n")
        handle.write(f"- Config: `{args.config}`\n")
        handle.write(f"- Rows: `{len(results_df)}`\n")
        handle.write(f"- Chunk rows: `{len(chunks_df)}`\n")
        handle.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        handle.write(f"- Identity stop strings: `{stop_strings}`\n")
        handle.write(
            f"- Sampling: `do_sample={generation_do_sample}`, `temperature={generation_temperature}`, "
            f"`top_p={generation_top_p}`, `top_k={generation_top_k}`\n"
        )


if __name__ == "__main__":
    main()
