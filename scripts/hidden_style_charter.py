from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Run the hidden style charter consistency task.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _normalize_yes_no(text: str) -> str:
    upper = text.upper()
    if "YES" in upper:
        return "YES"
    if "NO" in upper:
        return "NO"
    return "INVALID"


def _normalize_label(text: str) -> str:
    upper = text.upper()
    for label in ["A", "B", "C", "D"]:
        if label in upper:
            return label
    return "INVALID"


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import load_hidden_style_charter, load_identity_frames
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        build_layer_candidates,
        estimate_axis_vector,
        estimate_layer_scale,
        format_dialogue_prompt,
        greedy_site_run,
        load_identity_model,
    )

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    charter = load_hidden_style_charter()
    output_dir = ensure_output_dir(config, "hidden_style_charter")
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
        axis_name = str(config["hidden_charter_axis"])
        inject_vector = estimate_axis_vector(
            loaded=loaded,
            axis_name=axis_name,
            layer_index=fixed_layer,
            token_position=-1,
            max_prompt_tokens=int(config["max_prompt_tokens"]),
            seed=int(config["seed"]),
            control="mean_diff",
        )

        for frame_name in config["identity_frames"]:
            frame_text = frames[frame_name]
            layer_scale = estimate_layer_scale(
                loaded=loaded,
                texts=[charter["setup_prompt"]],
                layer_index=fixed_layer,
                token_position=-1,
                max_prompt_tokens=int(config["max_prompt_tokens"]),
            )
            for condition_name in ["no_steer", "authoritative_push"]:
                turns: list[tuple[str, str]] = [
                    ("User", str(charter["setup_prompt"])),
                    ("Assistant", "OK."),
                ]
                answers: list[tuple[str, str, str]] = []
                for question in charter["questions"]:
                    prompt = format_dialogue_prompt(frame_text, turns + [("User", str(question["text"]))])
                    response = greedy_site_run(
                        loaded=loaded,
                        prompt=prompt,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=4,
                        injection_site="last_prompt",
                        inject_layer=fixed_layer if condition_name != "no_steer" else None,
                        inject_vector=inject_vector if condition_name != "no_steer" else None,
                        inject_scale=(
                            float(config["strengths"][-1]) * float(layer_scale)
                            if condition_name != "no_steer"
                            else 0.0
                        ),
                    )
                    answer = _normalize_yes_no(response.completion_text)
                    answers.append((str(question["id"]), str(question["text"]), answer))
                    turns.extend([("User", str(question["text"])), ("Assistant", answer)])

                reveal_prompt = format_dialogue_prompt(
                    frame_text,
                    turns + [("User", str(charter["reveal_prompt"]))],
                )
                reveal = greedy_site_run(
                    loaded=loaded,
                    prompt=reveal_prompt,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=4,
                    injection_site="last_prompt",
                )
                label = _normalize_label(reveal.completion_text)

                if label in charter["charters"]:
                    expected_answers = {
                        str(question["id"]): str(question["answers"][label])
                        for question in charter["questions"]
                    }
                else:
                    expected_answers = {str(question["id"]): "INVALID" for question in charter["questions"]}

                matches = 0
                scored = 0
                for question_id, _, answer in answers:
                    expected = expected_answers[question_id]
                    if answer != "INVALID" and expected != "INVALID":
                        scored += 1
                        if answer == expected:
                            matches += 1
                consistency = float(matches / max(1, scored))
                contradiction_rate = float(1.0 - consistency)

                rows.append(
                    {
                        "model_id": model_id,
                        "model_family": infer_model_family(model_id),
                        "model_size_label": infer_model_size_label(model_id),
                        "identity_frame": frame_name,
                        "condition": condition_name,
                        "axis_name": axis_name,
                        "revealed_label": label,
                        "answers_json": json.dumps(
                            [
                                {"question_id": qid, "question": text, "answer": ans}
                                for qid, text, ans in answers
                            ]
                        ),
                        "consistency_score": consistency,
                        "contradiction_rate": contradiction_rate,
                        "valid_answer_count": int(scored),
                    }
                )

        del loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)

    summary = (
        df.groupby(["model_size_label", "identity_frame", "condition"], as_index=False)
        .agg(
            consistency_score_mean=("consistency_score", "mean"),
            contradiction_rate_mean=("contradiction_rate", "mean"),
            n=("model_id", "count"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Hidden Style Charter\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(df)}`\n")


if __name__ == "__main__":
    main()
