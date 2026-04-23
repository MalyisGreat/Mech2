from __future__ import annotations

import argparse
import json
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
)


DEFAULT_OTHER_FRAME_MAP = {
    "baseline_helpful": "tool_only",
    "instance_self": "tool_only",
    "family_self": "tool_only",
    "weights_self": "tool_only",
    "tool_only": "family_self",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-conditional self-vs-other prediction transfer checks.")
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


def _axis_score_to_unit_interval(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.tanh(float(value)))


def _make_prediction_prompt(target_description: str, positive: str, negative: str, task_prompt: str) -> str:
    labels = "; ".join(
        f"{short_label}={full_label.replace('_', ' ').lower()}"
        for short_label, full_label, _ in _label_map(positive, negative)
    )
    return (
        f"Do not answer the user's question yet. Predict how {target_description} would answer "
        f"on the axis {positive} vs {negative}. Reply with exactly one digit from 1 to 5. "
        f"Use this key: {labels}. Question: {task_prompt}"
    )


def _make_label_bias_prompt() -> str:
    return "This is a parser calibration item. Reply with exactly one digit from 1 to 5."


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import axis_sides, load_identity_frames, load_self_prediction_items
    from identity_stability.identity_probe_tools import predict_labeled_choice
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        default_stop_strings_for_template,
        format_identity_prompt,
        greedy_site_run,
        load_identity_model,
        score_against_axis_anchors,
    )

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    prompt_bank = load_self_prediction_items()["dimensions"]
    output_dir = ensure_output_dir(config, "self_prediction_transfer_v2")
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    stop_strings = resolve_identity_stop_strings(config)
    if stop_strings == ["AUTO"]:
        stop_strings = default_stop_strings_for_template(identity_prompt_template)

    use_label_bias_correction = bool(config.get("self_prediction_label_bias_correction", True))
    axes = [str(x) for x in config.get("self_prediction_axes", config.get("concept_axes", list(prompt_bank.keys())))]
    prompt_limit = int(config.get("self_prediction_prompt_limit", config.get("prompt_limit_per_axis", 4)))
    rows: list[dict[str, object]] = []

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
            other_frame_name = str(
                config.get("self_prediction_other_frame_map", {}).get(
                    frame_name, DEFAULT_OTHER_FRAME_MAP.get(frame_name, "tool_only")
                )
            )
            other_frame_text = frames[other_frame_name]
            for axis_name in axes:
                positive, negative = axis_sides(axis_name)
                prompts = [str(x) for x in prompt_bank[axis_name]["prompts"][:prompt_limit]]
                for prompt_index, prompt in enumerate(prompts):
                    label_bias_prompt = (
                        format_identity_prompt(
                            frame_text,
                            _make_label_bias_prompt(),
                            template=identity_prompt_template,
                        )
                        if use_label_bias_correction
                        else None
                    )
                    self_pred_prompt = format_identity_prompt(
                        frame_text,
                        _make_prediction_prompt("you yourself in this setting", positive, negative, prompt),
                        template=identity_prompt_template,
                    )
                    other_pred_prompt = format_identity_prompt(
                        frame_text,
                        _make_prediction_prompt(
                            f"an assistant operating under this system instruction: {other_frame_text}",
                            positive,
                            negative,
                            prompt,
                        ),
                        template=identity_prompt_template,
                    )

                    (
                        self_short_label,
                        self_full_label,
                        self_pred_score,
                        self_pred_confidence,
                        self_prediction_text,
                        self_prediction_details,
                    ) = predict_labeled_choice(
                        loaded=loaded,
                        prompt=self_pred_prompt,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        labels=_label_map(positive, negative),
                        label_bias_prompt=label_bias_prompt,
                    )
                    (
                        other_short_label,
                        other_full_label,
                        other_pred_score,
                        other_pred_confidence,
                        other_prediction_text,
                        other_prediction_details,
                    ) = predict_labeled_choice(
                        loaded=loaded,
                        prompt=other_pred_prompt,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        labels=_label_map(positive, negative),
                        label_bias_prompt=label_bias_prompt,
                    )

                    self_answer = greedy_site_run(
                        loaded=loaded,
                        prompt=format_identity_prompt(frame_text, prompt, template=identity_prompt_template),
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=int(config["default_generation_tokens"]),
                        injection_site="last_prompt",
                        stop_strings=stop_strings,
                    )
                    other_answer = greedy_site_run(
                        loaded=loaded,
                        prompt=format_identity_prompt(other_frame_text, prompt, template=identity_prompt_template),
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=int(config["default_generation_tokens"]),
                        injection_site="last_prompt",
                        stop_strings=stop_strings,
                    )
                    self_actual_score = _axis_score_to_unit_interval(
                        float(score_against_axis_anchors(axis_name, self_answer.completion_text))
                    )
                    other_actual_score = _axis_score_to_unit_interval(
                        float(score_against_axis_anchors(axis_name, other_answer.completion_text))
                    )
                    self_error_to_self = (
                        float(abs(self_pred_score - self_actual_score)) if np.isfinite(self_pred_score) else float("nan")
                    )
                    self_error_to_other = (
                        float(abs(self_pred_score - other_actual_score)) if np.isfinite(self_pred_score) else float("nan")
                    )
                    other_error_to_other = (
                        float(abs(other_pred_score - other_actual_score)) if np.isfinite(other_pred_score) else float("nan")
                    )
                    other_error_to_self = (
                        float(abs(other_pred_score - self_actual_score)) if np.isfinite(other_pred_score) else float("nan")
                    )
                    self_advantage = (
                        float(self_error_to_other - self_error_to_self)
                        if np.isfinite(self_error_to_self) and np.isfinite(self_error_to_other)
                        else float("nan")
                    )
                    other_advantage = (
                        float(other_error_to_self - other_error_to_other)
                        if np.isfinite(other_error_to_other) and np.isfinite(other_error_to_self)
                        else float("nan")
                    )
                    discriminative_win = float(
                        np.isfinite(self_advantage)
                        and np.isfinite(other_advantage)
                        and self_advantage > 0.0
                        and other_advantage > 0.0
                    )

                    rows.append(
                        {
                            "model_id": model_id,
                            "model_family": infer_model_family(model_id),
                            "model_size_label": infer_model_size_label(model_id),
                            "identity_frame": frame_name,
                            "other_frame": other_frame_name,
                            "axis_name": axis_name,
                            "prompt_index": int(prompt_index),
                            "prompt": prompt,
                            "self_prediction_text": self_prediction_text,
                            "self_predicted_short_label": self_short_label,
                            "self_predicted_label": self_full_label,
                            "self_predicted_score": float(self_pred_score) if np.isfinite(self_pred_score) else np.nan,
                            "self_predicted_confidence": float(self_pred_confidence)
                            if np.isfinite(self_pred_confidence)
                            else np.nan,
                            "self_prediction_details_json": json.dumps(self_prediction_details),
                            "other_prediction_text": other_prediction_text,
                            "other_predicted_short_label": other_short_label,
                            "other_predicted_label": other_full_label,
                            "other_predicted_score": float(other_pred_score) if np.isfinite(other_pred_score) else np.nan,
                            "other_predicted_confidence": float(other_pred_confidence)
                            if np.isfinite(other_pred_confidence)
                            else np.nan,
                            "other_prediction_details_json": json.dumps(other_prediction_details),
                            "self_actual_text": self_answer.completion_text,
                            "other_actual_text": other_answer.completion_text,
                            "self_actual_score_unit": float(self_actual_score),
                            "other_actual_score_unit": float(other_actual_score),
                            "self_error_to_self": self_error_to_self,
                            "self_error_to_other": self_error_to_other,
                            "other_error_to_self": other_error_to_self,
                            "other_error_to_other": other_error_to_other,
                            "self_prediction_advantage": self_advantage,
                            "other_prediction_advantage": other_advantage,
                            "discriminative_win": discriminative_win,
                        }
                    )

        del loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)

    summary = (
        df.groupby(["model_size_label", "identity_frame", "axis_name"], as_index=False)
        .agg(
            self_prediction_advantage_mean=("self_prediction_advantage", "mean"),
            other_prediction_advantage_mean=("other_prediction_advantage", "mean"),
            discriminative_win_rate=("discriminative_win", "mean"),
            self_error_to_self_mean=("self_error_to_self", "mean"),
            self_error_to_other_mean=("self_error_to_other", "mean"),
            other_error_to_other_mean=("other_error_to_other", "mean"),
            other_error_to_self_mean=("other_error_to_self", "mean"),
            n=("prompt", "count"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    summary_by_model = (
        df.groupby(["model_size_label", "identity_frame"], as_index=False)
        .agg(
            self_prediction_advantage_mean=("self_prediction_advantage", "mean"),
            other_prediction_advantage_mean=("other_prediction_advantage", "mean"),
            discriminative_win_rate=("discriminative_win", "mean"),
            n=("prompt", "count"),
        )
    )
    summary_by_model.to_csv(output_dir / "summary_by_model.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Self Prediction Transfer V2\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(df)}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(f"- Identity stop strings: `{stop_strings}`\n")
        f.write("- Purpose: test whether prompt-conditional self-predictions are more accurate for self than for a nearby other assistant.\n")


if __name__ == "__main__":
    main()
