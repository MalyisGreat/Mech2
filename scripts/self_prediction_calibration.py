from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from identity_battery_common import (
    add_src_to_path,
    ensure_output_dir,
    infer_model_family,
    infer_model_size_label,
    load_yaml_config,
    resolve_identity_prompt_template,
    resolve_identity_stop_strings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-conditional self-prediction calibration checks.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _label_map(positive: str, negative: str) -> list[tuple[str, str, float]]:
    return [
        ("1", f"STRONGLY_{positive.upper()}", 1.0),
        ("2", f"LEAN_{positive.upper()}", 0.5),
        ("3", "MIXED", 0.0),
        ("4", f"LEAN_{negative.upper()}", -0.5),
        ("5", f"STRONGLY_{negative.upper()}", -1.0),
    ]


def _candidate_token_ids(tokenizer, choice: str) -> list[int]:
    candidates: set[int] = set()
    for variant in (choice, f" {choice}", f"\n{choice}"):
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(token_ids) == 1:
            candidates.add(int(token_ids[0]))
    return sorted(candidates)


def _score_choice_logits(loaded, prompt: str, max_prompt_tokens: int, choices: list[str]) -> tuple[dict[str, float], str]:
    from identity_stability.steered_generation import greedy_site_run

    probe = greedy_site_run(
        loaded=loaded,
        prompt=prompt,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=1,
        injection_site="last_prompt",
    )
    logits = probe.site_logits.float()
    scores: dict[str, float] = {}
    for choice in choices:
        token_ids = _candidate_token_ids(loaded.tokenizer, choice)
        if token_ids:
            scores[choice] = float(torch.max(logits[token_ids]).item())
    return scores, probe.completion_text


def _predict_forced_choice(
    loaded,
    prompt: str,
    max_prompt_tokens: int,
    positive: str,
    negative: str,
    *,
    label_bias_prompt: str | None = None,
) -> tuple[str, float, float, str, dict[str, object]]:
    labels = _label_map(positive, negative)
    short_labels = [short_label for short_label, _, _ in labels]
    raw_logit_scores, prediction_text = _score_choice_logits(
        loaded=loaded,
        prompt=prompt,
        max_prompt_tokens=max_prompt_tokens,
        choices=short_labels,
    )
    bias_scores: dict[str, float] = {}
    if label_bias_prompt is not None:
        bias_scores, _ = _score_choice_logits(
            loaded=loaded,
            prompt=label_bias_prompt,
            max_prompt_tokens=max_prompt_tokens,
            choices=short_labels,
        )

    choice_scores: list[tuple[str, str, float, float]] = []
    for short_label, full_label, score in labels:
        if short_label not in raw_logit_scores:
            continue
        token_score = float(raw_logit_scores[short_label] - bias_scores.get(short_label, 0.0))
        choice_scores.append((short_label, full_label, float(score), token_score))

    if not choice_scores:
        return (
            "INVALID",
            float("nan"),
            float("nan"),
            prediction_text,
            {
                "scoring_mode": "bias_corrected_digits" if label_bias_prompt is not None else "raw_digits",
                "raw_logit_scores": raw_logit_scores,
                "label_bias_scores": bias_scores,
                "corrected_scores": {},
                "selected_short_label": "INVALID",
            },
        )

    raw_scores = torch.tensor([item[3] for item in choice_scores], dtype=torch.float32)
    probs = torch.softmax(raw_scores, dim=0)
    best_idx = int(torch.argmax(probs).item())
    short_label, full_label, score_value, _ = choice_scores[best_idx]
    confidence = float(probs[best_idx].item())
    return (
        full_label,
        float(score_value),
        confidence,
        prediction_text,
        {
            "scoring_mode": "bias_corrected_digits" if label_bias_prompt is not None else "raw_digits",
            "raw_logit_scores": raw_logit_scores,
            "label_bias_scores": bias_scores,
            "corrected_scores": {item[0]: item[3] for item in choice_scores},
            "selected_short_label": short_label,
        },
    )


def _axis_score_to_unit_interval(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.tanh(float(value)))


def _axis_score_to_side(value: float, positive: str, negative: str) -> str:
    return positive if value >= 0.0 else negative


def _make_prediction_prompt(axis_name: str, positive: str, negative: str, task_prompt: str) -> str:
    labels = "; ".join(
        f"{short_label}={full_label.replace('_', ' ').lower()}"
        for short_label, full_label, _ in _label_map(positive, negative)
    )
    return (
        f"Do not answer the user's question yet. "
        f"Predict how your own next answer will sound on the axis {positive} vs {negative}. "
        f"Reply with exactly one digit from 1 to 5. "
        f"Use this key: {labels}. "
        f"Question: {task_prompt}"
    )


def _make_label_bias_prompt() -> str:
    return (
        "This is a parser calibration item, not a self-description task. "
        "Reply with exactly one digit from 1 to 5."
    )


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import (
        axis_sides,
        load_identity_frames,
        load_self_prediction_items,
    )
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        default_stop_strings_for_template,
        format_framed_prompt,
        format_identity_prompt,
        greedy_site_run,
        load_identity_model,
        score_against_axis_anchors,
    )
    from identity_stability.text_features import extract_style_features

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    items = load_self_prediction_items()["dimensions"]
    output_dir = ensure_output_dir(config, "self_prediction_calibration")
    rows: list[dict[str, object]] = []
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    stop_strings = resolve_identity_stop_strings(config)
    if stop_strings == ["AUTO"]:
        stop_strings = default_stop_strings_for_template(identity_prompt_template)

    axes = [str(x) for x in config.get("self_prediction_axes", config.get("concept_axes", list(items.keys())))]
    prompt_limit = int(config.get("self_prediction_prompt_limit", config.get("prompt_limit_per_axis", 4)))
    use_label_bias_correction = bool(config.get("self_prediction_label_bias_correction", True))

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
            for axis_name in axes:
                positive, negative = axis_sides(axis_name)
                prompts = [str(x) for x in items[axis_name]["prompts"][:prompt_limit]]
                for prompt in prompts:
                    prediction_prompt = format_identity_prompt(
                        frame_text,
                        _make_prediction_prompt(axis_name, positive, negative, prompt),
                        template=identity_prompt_template,
                    )
                    label_bias_prompt = (
                        format_identity_prompt(
                            frame_text,
                            _make_label_bias_prompt(),
                            template=identity_prompt_template,
                        )
                        if use_label_bias_correction
                        else None
                    )
                    predicted_label, predicted_score, predicted_confidence, prediction_text, prediction_score_details = _predict_forced_choice(
                        loaded=loaded,
                        prompt=prediction_prompt,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        positive=positive,
                        negative=negative,
                        label_bias_prompt=label_bias_prompt,
                    )

                    answer = greedy_site_run(
                        loaded=loaded,
                        prompt=format_identity_prompt(frame_text, prompt, template=identity_prompt_template),
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=int(config["default_generation_tokens"]),
                        injection_site="last_prompt",
                        stop_strings=stop_strings,
                    )
                    actual_axis_score = float(score_against_axis_anchors(axis_name, answer.completion_text))
                    actual_score_unit = _axis_score_to_unit_interval(actual_axis_score)
                    actual_side = _axis_score_to_side(actual_axis_score, positive=positive, negative=negative)
                    predicted_side = (
                        positive if predicted_score >= 0.0 else negative if predicted_score < 0.0 else "mixed"
                    )
                    sign_accuracy = float(
                        predicted_label != "INVALID"
                        and (
                            predicted_side == actual_side
                            or (predicted_side == "mixed" and abs(actual_score_unit) < 0.15)
                        )
                    )
                    calibration_error = float(
                        abs(float(predicted_score) - actual_score_unit)
                        if np.isfinite(predicted_score)
                        else 1.0
                    )

                    rows.append(
                        {
                            "model_id": model_id,
                            "model_family": infer_model_family(model_id),
                            "model_size_label": infer_model_size_label(model_id),
                            "identity_frame": frame_name,
                            "axis_name": axis_name,
                            "positive_side": positive,
                            "negative_side": negative,
                            "prompt": prompt,
                            "prediction_text": prediction_text,
                            "predicted_label": predicted_label,
                            "predicted_score": float(predicted_score) if np.isfinite(predicted_score) else np.nan,
                            "predicted_confidence": float(predicted_confidence) if np.isfinite(predicted_confidence) else np.nan,
                            "prediction_scoring_mode": str(prediction_score_details.get("scoring_mode", "unknown")),
                            "prediction_score_details_json": json.dumps(prediction_score_details),
                            "predicted_side": predicted_side,
                            "actual_text": answer.completion_text,
                            "actual_axis_score": actual_axis_score,
                            "actual_score_unit": float(actual_score_unit),
                            "actual_side": actual_side,
                            "sign_accuracy": sign_accuracy,
                            "calibration_error": calibration_error,
                            **extract_style_features(answer.completion_text),
                        }
                    )

        del loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)

    summary = (
        df.groupby(["model_size_label", "identity_frame", "axis_name"], as_index=False)
        .agg(
            sign_accuracy_mean=("sign_accuracy", "mean"),
            calibration_error_mean=("calibration_error", "mean"),
            predicted_score_mean=("predicted_score", "mean"),
            actual_score_unit_mean=("actual_score_unit", "mean"),
            n=("prompt", "count"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    model_summary = (
        df.groupby(["model_size_label", "identity_frame"], as_index=False)
        .agg(
            sign_accuracy_mean=("sign_accuracy", "mean"),
            calibration_error_mean=("calibration_error", "mean"),
            n=("prompt", "count"),
        )
    )
    model_summary.to_csv(output_dir / "summary_by_model.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Self Prediction Calibration\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(df)}`\n")
        f.write(f"- Label-bias correction: `{use_label_bias_correction}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(f"- Identity stop strings: `{stop_strings}`\n")
        f.write("- Purpose: test whether the model can predict, prompt by prompt, how it itself is about to answer.\n")


if __name__ == "__main__":
    main()
