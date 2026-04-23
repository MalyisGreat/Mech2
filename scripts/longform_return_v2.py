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


DIALOGUE_AXIS_MAP = {
    "return_01": "expansive_vs_terse",
    "return_02": "cautious_vs_assertive",
    "return_03": "collaborative_vs_authoritative",
    "return_04": "selfref_vs_impersonal",
    "return_05": "expansive_vs_terse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the repaired long-form return battery.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _format_dialogue_task(turns: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    user_count = 0
    answer_count = 0
    for role, content in turns:
        if role.lower() == "user":
            user_count += 1
            lines.append(f"User prompt {user_count}: {content}")
        else:
            answer_count += 1
            lines.append(f"Previous assistant answer {answer_count}: {content}")
    lines.append("Write the next assistant answer directly, without labels or markup.")
    return "\n".join(lines)


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import (
        load_identity_frames,
        load_longform_seed_dialogues,
        load_return_probe_fillers,
    )
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        default_stop_strings_for_template,
        format_identity_prompt,
        greedy_site_run,
        load_identity_model,
        score_against_axis_anchors,
    )
    from identity_stability.text_features import chunk_text_by_words, semantic_overlap, stylometric_distance

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    fillers = [str(x) for x in load_return_probe_fillers().get("fillers", [])]
    dialogues = {
        row["id"]: row
        for row in load_longform_seed_dialogues()
        if row["id"] in set(config["longform_dialogue_ids"])
    }
    output_dir = ensure_output_dir(config, "longform_return_v2")
    rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    stop_strings = resolve_identity_stop_strings(config)
    if stop_strings == ["AUTO"]:
        stop_strings = default_stop_strings_for_template(identity_prompt_template)
    filler_count = int(config.get("longform_delayed_filler_count", 2))

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
            for dialogue_id, dialogue in dialogues.items():
                turns = list(dialogue["turns"])
                axis_name = DIALOGUE_AXIS_MAP[dialogue_id]

                baseline_prompt = format_identity_prompt(
                    frame_text,
                    _format_dialogue_task([("User", turns[1])]),
                    template=identity_prompt_template,
                )
                baseline_answer = greedy_site_run(
                    loaded=loaded,
                    prompt=baseline_prompt,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["longform_generation_tokens"]),
                    injection_site="last_prompt",
                    stop_strings=stop_strings,
                )

                forced_prompt = format_identity_prompt(
                    frame_text,
                    _format_dialogue_task(
                        [
                            ("User", turns[1]),
                            ("Assistant", baseline_answer.completion_text),
                            ("User", turns[2]),
                        ]
                    ),
                    template=identity_prompt_template,
                )
                forced_answer = greedy_site_run(
                    loaded=loaded,
                    prompt=forced_prompt,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["longform_generation_tokens"]),
                    injection_site="last_prompt",
                    stop_strings=stop_strings,
                )

                immediate_prompt = format_identity_prompt(
                    frame_text,
                    _format_dialogue_task(
                        [
                            ("User", turns[1]),
                            ("Assistant", baseline_answer.completion_text),
                            ("User", turns[2]),
                            ("Assistant", forced_answer.completion_text),
                            ("User", turns[3]),
                        ]
                    ),
                    template=identity_prompt_template,
                )
                immediate_return = greedy_site_run(
                    loaded=loaded,
                    prompt=immediate_prompt,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["longform_generation_tokens"]),
                    injection_site="last_prompt",
                    stop_strings=stop_strings,
                )

                delayed_turns = [
                    ("User", turns[1]),
                    ("Assistant", baseline_answer.completion_text),
                    ("User", turns[2]),
                    ("Assistant", forced_answer.completion_text),
                ]
                filler_answers: list[tuple[str, str]] = []
                for filler_text in fillers[:filler_count]:
                    filler_prompt = format_identity_prompt(
                        frame_text,
                        _format_dialogue_task(delayed_turns + [("User", filler_text)]),
                        template=identity_prompt_template,
                    )
                    filler_answer = greedy_site_run(
                        loaded=loaded,
                        prompt=filler_prompt,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=int(config.get("longform_filler_generation_tokens", 32)),
                        injection_site="last_prompt",
                        stop_strings=stop_strings,
                    )
                    delayed_turns.extend([("User", filler_text), ("Assistant", filler_answer.completion_text)])
                    filler_answers.append((filler_text, filler_answer.completion_text))

                delayed_prompt = format_identity_prompt(
                    frame_text,
                    _format_dialogue_task(delayed_turns + [("User", turns[3])]),
                    template=identity_prompt_template,
                )
                delayed_return = greedy_site_run(
                    loaded=loaded,
                    prompt=delayed_prompt,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=int(config["longform_generation_tokens"]),
                    injection_site="last_prompt",
                    stop_strings=stop_strings,
                )

                baseline_axis_score = float(score_against_axis_anchors(axis_name, baseline_answer.completion_text))
                forced_axis_score = float(score_against_axis_anchors(axis_name, forced_answer.completion_text))
                conditions = {
                    "prompt_only_immediate": immediate_return,
                    "prompt_only_delayed": delayed_return,
                }
                for condition_name, return_answer in conditions.items():
                    return_axis_score = float(score_against_axis_anchors(axis_name, return_answer.completion_text))
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
                            "baseline_text": baseline_answer.completion_text,
                            "forced_text": forced_answer.completion_text,
                            "return_text": return_answer.completion_text,
                            "baseline_axis_score": baseline_axis_score,
                            "forced_axis_score": forced_axis_score,
                            "return_axis_score": return_axis_score,
                            "return_to_baseline_index": float(return_to_baseline),
                            "semantic_overlap_with_baseline": semantic_overlap(
                                return_answer.completion_text, baseline_answer.completion_text
                            ),
                            "semantic_overlap_with_forced": semantic_overlap(
                                return_answer.completion_text, forced_answer.completion_text
                            ),
                            "stylometric_distance_to_baseline": stylometric_distance(
                                return_answer.completion_text, baseline_answer.completion_text
                            ),
                            "stylometric_distance_to_forced": stylometric_distance(
                                return_answer.completion_text, forced_answer.completion_text
                            ),
                            "filler_answers_json": str(filler_answers),
                        }
                    )

                    chunks = chunk_text_by_words(return_answer.completion_text, window_words=50)
                    for chunk_index, chunk_text in enumerate(chunks):
                        chunk_axis_score = float(score_against_axis_anchors(axis_name, chunk_text))
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
                                "chunk_axis_score": chunk_axis_score,
                                "chunk_return_to_baseline_index": float(chunk_return_index),
                            }
                        )

        del loaded
        clear_cuda()

    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame(chunk_rows).to_csv(output_dir / "chunk_curves.csv", index=False)

    summary = (
        pd.DataFrame(rows)
        .groupby(["model_size_label", "identity_frame", "condition"], as_index=False)
        .agg(
            return_to_baseline_index_mean=("return_to_baseline_index", "mean"),
            semantic_overlap_with_baseline_mean=("semantic_overlap_with_baseline", "mean"),
            stylometric_distance_to_baseline_mean=("stylometric_distance_to_baseline", "mean"),
            n=("dialogue_id", "count"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Long-Form Return V2\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(rows)}`\n")
        f.write(f"- Chunk rows: `{len(chunk_rows)}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(f"- Identity stop strings: `{stop_strings}`\n")


if __name__ == "__main__":
    main()
