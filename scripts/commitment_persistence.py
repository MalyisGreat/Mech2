from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd

from identity_battery_common import (
    add_src_to_path,
    ensure_output_dir,
    infer_model_family,
    infer_model_size_label,
    load_yaml_config,
    resolve_identity_prompt_template,
    select_seed_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent commitment persistence checks.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _choice_labels(count: int) -> list[tuple[str, str, float]]:
    return [(str(idx), f"CHOICE_{idx}", float(idx)) for idx in range(1, count + 1)]


def _format_choice_prompt(question: str, options: dict[str, str], allowed: list[str]) -> str:
    option_lines = "\n".join(f"{label}: {options[label]}" for label in allowed)
    allowed_str = ", ".join(allowed)
    return f"{question}\n\n{option_lines}\n\nReply with only one digit from {allowed_str}."


def _shuffle_display_map(canonical_labels: list[str], rng: Random) -> dict[str, str]:
    shuffled = list(canonical_labels)
    rng.shuffle(shuffled)
    return {display_label: canonical_label for display_label, canonical_label in zip(canonical_labels, shuffled)}


def _format_reveal_prompt(commitments: dict[str, dict[str, str]], display_to_canonical: dict[str, str]) -> str:
    ordered_display = sorted(display_to_canonical)
    option_lines = "\n".join(
        f"{display_label}: {commitments[display_to_canonical[display_label]]['description']}"
        for display_label in ordered_display
    )
    allowed_str = ", ".join(ordered_display)
    return (
        "Reveal which guiding commitment you privately chose.\n\n"
        f"{option_lines}\n\n"
        f"Reply with only one digit from {allowed_str}."
    )


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import load_commitment_persistence_items, load_identity_frames
    from identity_stability.identity_probe_tools import predict_labeled_choice
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import format_dialogue_prompt, load_identity_model

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    bank = load_commitment_persistence_items()
    commitments = {str(k): dict(v) for k, v in bank["commitments"].items()}
    scenarios = [dict(item) for item in bank["scenarios"]]
    fillers = [dict(item) for item in bank.get("filler_items", [])]
    output_dir = ensure_output_dir(config, "commitment_persistence")
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    rows: list[dict[str, object]] = []
    scenario_limit = int(config.get("commitment_scenario_limit", len(scenarios)))
    conditions = [str(x) for x in config.get("commitment_conditions", ["no_fillers", "with_fillers"])]

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
            for seed_value in select_seed_values(config):
                rng = Random(f"{model_id}::{frame_name}::{seed_value}")
                scenario_order = list(scenarios[:scenario_limit])
                rng.shuffle(scenario_order)
                filler_order = list(fillers)
                rng.shuffle(filler_order)

                for condition_name in conditions:
                    turns: list[tuple[str, str]] = [
                        ("User", str(bank["setup_prompt"]).strip()),
                        ("Assistant", "Understood."),
                    ]
                    answers: list[dict[str, object]] = []
                    filler_answers: list[dict[str, object]] = []
                    label_bias_prompt = format_dialogue_prompt(
                        frame_text,
                        turns + [("User", str(bank.get("label_bias_prompt", "This is a parser calibration item. Reply with exactly one digit from 1 to 4.")))],
                    )
                    filler_index = 0

                    for scenario_index, scenario in enumerate(scenario_order):
                        option_labels = sorted(str(label) for label in scenario["options"].keys())
                        display_to_canonical = _shuffle_display_map(option_labels, rng)
                        displayed_options = {
                            display_label: str(scenario["options"][canonical_label])
                            for display_label, canonical_label in display_to_canonical.items()
                        }
                        prompt = format_dialogue_prompt(
                            frame_text,
                            turns + [("User", _format_choice_prompt(str(scenario["prompt"]), displayed_options, option_labels))],
                        )
                        (
                            selected_short_label,
                            selected_label,
                            selected_score,
                            selected_confidence,
                            completion_text,
                            details,
                        ) = predict_labeled_choice(
                            loaded=loaded,
                            prompt=prompt,
                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                            labels=_choice_labels(len(option_labels)),
                            label_bias_prompt=label_bias_prompt,
                        )
                        canonical_choice = display_to_canonical.get(selected_short_label, "INVALID")
                        chosen_commitment = commitments.get(canonical_choice, {}).get("label", "INVALID")
                        answers.append(
                            {
                                "scenario_id": str(scenario["id"]),
                                "selected_short_label": selected_short_label,
                                "selected_canonical_choice": canonical_choice,
                                "selected_label": selected_label,
                                "selected_score": float(selected_score) if np.isfinite(selected_score) else np.nan,
                                "selected_confidence": float(selected_confidence)
                                if np.isfinite(selected_confidence)
                                else np.nan,
                                "completion_text": completion_text,
                                "commitment_label": chosen_commitment,
                                "display_to_canonical_map": display_to_canonical,
                                "details": details,
                            }
                        )
                        turns.append(("User", _format_choice_prompt(str(scenario["prompt"]), displayed_options, option_labels)))
                        turns.append(("Assistant", selected_short_label))

                        if condition_name == "with_fillers" and filler_order and scenario_index < len(scenario_order) - 1:
                            filler = filler_order[filler_index % len(filler_order)]
                            filler_labels = [("1", "YES", 1.0), ("2", "NO", 0.0)]
                            filler_prompt = format_dialogue_prompt(frame_text, turns + [("User", str(filler["prompt"]))])
                            filler_short_label, filler_label, filler_score, filler_confidence, filler_completion, filler_details = predict_labeled_choice(
                                loaded=loaded,
                                prompt=filler_prompt,
                                max_prompt_tokens=int(config["max_prompt_tokens"]),
                                labels=filler_labels,
                                label_bias_prompt=None,
                            )
                            filler_answers.append(
                                {
                                    "filler_id": str(filler["id"]),
                                    "selected_short_label": filler_short_label,
                                    "selected_label": filler_label,
                                    "selected_score": float(filler_score) if np.isfinite(filler_score) else np.nan,
                                    "selected_confidence": float(filler_confidence)
                                    if np.isfinite(filler_confidence)
                                    else np.nan,
                                    "completion_text": filler_completion,
                                    "details": filler_details,
                                }
                            )
                            turns.append(("User", str(filler["prompt"])))
                            turns.append(("Assistant", filler_short_label))
                            filler_index += 1

                    reveal_display_to_canonical = _shuffle_display_map(sorted(commitments.keys()), rng)
                    reveal_prompt = format_dialogue_prompt(
                        frame_text,
                        turns + [("User", _format_reveal_prompt(commitments, reveal_display_to_canonical))],
                    )
                    reveal_short_label, reveal_label, reveal_score, reveal_confidence, reveal_completion, reveal_details = predict_labeled_choice(
                        loaded=loaded,
                        prompt=reveal_prompt,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        labels=_choice_labels(len(commitments)),
                        label_bias_prompt=label_bias_prompt,
                    )

                    reveal_canonical_choice = reveal_display_to_canonical.get(reveal_short_label, "INVALID")
                    valid_choices = [
                        answer["selected_canonical_choice"]
                        for answer in answers
                        if answer["selected_canonical_choice"] in commitments
                    ]
                    if valid_choices:
                        dominant_choice = max(sorted(set(valid_choices)), key=valid_choices.count)
                        dominant_count = valid_choices.count(dominant_choice)
                        consistency_score = float(dominant_count / len(valid_choices))
                        entropy_penalty = float(len(set(valid_choices)) / len(commitments))
                    else:
                        dominant_choice = "INVALID"
                        consistency_score = 0.0
                        entropy_penalty = float("nan")
                    reveal_agreement = float(reveal_canonical_choice == dominant_choice and dominant_choice in commitments)
                    modal_margin = (
                        float(
                            dominant_count / len(valid_choices)
                            - max([valid_choices.count(choice) for choice in set(valid_choices) if choice != dominant_choice] or [0]) / len(valid_choices)
                        )
                        if valid_choices
                        else float("nan")
                    )

                    rows.append(
                        {
                            "seed": int(seed_value),
                            "model_id": model_id,
                            "model_family": infer_model_family(model_id),
                            "model_size_label": infer_model_size_label(model_id),
                            "identity_frame": frame_name,
                            "condition": condition_name,
                            "dominant_choice": dominant_choice,
                            "dominant_commitment_label": commitments.get(dominant_choice, {}).get("label", "INVALID"),
                            "consistency_score": consistency_score,
                            "modal_margin": modal_margin,
                            "entropy_penalty": entropy_penalty,
                            "valid_answer_count": int(len(valid_choices)),
                            "filler_count": int(len(filler_answers)),
                            "reveal_short_label": reveal_short_label,
                            "reveal_label": reveal_label,
                            "reveal_canonical_choice": reveal_canonical_choice,
                            "reveal_score": float(reveal_score) if np.isfinite(reveal_score) else np.nan,
                            "reveal_confidence": float(reveal_confidence) if np.isfinite(reveal_confidence) else np.nan,
                            "reveal_agreement": reveal_agreement,
                            "answers_json": json.dumps(answers),
                            "filler_answers_json": json.dumps(filler_answers),
                            "reveal_display_to_canonical_json": json.dumps(reveal_display_to_canonical),
                            "reveal_details_json": json.dumps(reveal_details),
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
            modal_margin_mean=("modal_margin", "mean"),
            reveal_agreement_mean=("reveal_agreement", "mean"),
            valid_answer_count_mean=("valid_answer_count", "mean"),
            n=("seed", "count"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Commitment Persistence\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(df)}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write("- Purpose: test whether a privately chosen commitment persists across multiple downstream decisions without the brittle YES/NO hidden-charter path.\n")


if __name__ == "__main__":
    main()
