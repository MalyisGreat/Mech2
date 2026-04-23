from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

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
    select_seed_values,
)


OTHER_FRAME_MAP = {
    "baseline_helpful": "persona_editor",
    "instance_self": "persona_editor",
    "family_self": "persona_editor",
    "weights_self": "persona_editor",
    "tool_only": "family_self",
    "persona_editor": "tool_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self/other boundary transfer checks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Apply config.smoke overrides if present.")
    return parser.parse_args()


def _apply_mode_overrides(config: dict, *, smoke: bool) -> dict:
    resolved = dict(config)
    if smoke and isinstance(config.get("smoke"), dict):
        resolved.update(dict(config["smoke"]))
    return resolved


def _label_map(positive: str, negative: str) -> list[tuple[str, str, float]]:
    return [
        ("1", f"STRONGLY_{positive.upper()}", 1.0),
        ("2", f"LEAN_{positive.upper()}", 0.5),
        ("3", "MIXED", 0.0),
        ("4", f"LEAN_{negative.upper()}", -0.5),
        ("5", f"STRONGLY_{negative.upper()}", -1.0),
    ]


def _boundary_label_map(positive: str) -> list[tuple[str, str, float]]:
    return [
        ("1", f"A_STRONGLY_MORE_{positive.upper()}", 1.0),
        ("2", f"A_SLIGHTLY_MORE_{positive.upper()}", 0.5),
        ("3", "NO_MEANINGFUL_DIFFERENCE", 0.0),
        ("4", f"B_SLIGHTLY_MORE_{positive.upper()}", -0.5),
        ("5", f"B_STRONGLY_MORE_{positive.upper()}", -1.0),
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


def _predict_weighted_choice(
    loaded,
    prompt: str,
    max_prompt_tokens: int,
    labels: list[tuple[str, str, float]],
    *,
    label_bias_prompt: str | None = None,
) -> tuple[str, float, float, str, dict[str, object]]:
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
        corrected_score = float(raw_logit_scores[short_label] - bias_scores.get(short_label, 0.0))
        choice_scores.append((short_label, full_label, float(score), corrected_score))

    score_details = {
        "raw_logit_scores": raw_logit_scores,
        "label_bias_scores": bias_scores,
        "corrected_scores": {item[0]: item[3] for item in choice_scores},
        "scoring_mode": "bias_corrected_digits" if label_bias_prompt is not None else "raw_digits",
    }
    if not choice_scores:
        score_details["selected_short_label"] = "INVALID"
        return "INVALID", float("nan"), float("nan"), prediction_text, score_details

    score_tensor = torch.tensor([score for _, _, _, score in choice_scores], dtype=torch.float32)
    probs = torch.softmax(score_tensor, dim=0)
    best_idx = int(torch.argmax(probs).item())
    short_label, full_label, score_value, _ = choice_scores[best_idx]
    score_details["selected_short_label"] = short_label
    return full_label, float(score_value), float(probs[best_idx].item()), prediction_text, score_details


def _predict_forced_choice(
    loaded,
    prompt: str,
    max_prompt_tokens: int,
    positive: str,
    negative: str,
    *,
    label_bias_prompt: str | None = None,
) -> tuple[str, float, float, str, dict[str, object]]:
    return _predict_weighted_choice(
        loaded=loaded,
        prompt=prompt,
        max_prompt_tokens=max_prompt_tokens,
        labels=_label_map(positive, negative),
        label_bias_prompt=label_bias_prompt,
    )


def _predict_boundary_relation(
    loaded,
    prompt: str,
    max_prompt_tokens: int,
    positive: str,
    *,
    label_bias_prompt: str | None = None,
) -> tuple[str, float, float, str, dict[str, object]]:
    return _predict_weighted_choice(
        loaded=loaded,
        prompt=prompt,
        max_prompt_tokens=max_prompt_tokens,
        labels=_boundary_label_map(positive),
        label_bias_prompt=label_bias_prompt,
    )


def _normalize_boundary_label(label: str, positive: str, *, self_is_a: bool) -> str:
    if label == "INVALID":
        return label
    if label == "NO_MEANINGFUL_DIFFERENCE":
        return label
    if label.startswith("A_"):
        return label.replace("A_", "SELF_", 1) if self_is_a else label.replace("A_", "OTHER_", 1)
    if label.startswith("B_"):
        return label.replace("B_", "OTHER_", 1) if self_is_a else label.replace("B_", "SELF_", 1)
    return f"{label}_ON_{positive.upper()}"


def _axis_score_to_unit_interval(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.tanh(float(value)))


def _axis_score_to_side(value: float, positive: str, negative: str) -> str:
    return positive if value >= 0.0 else negative


def _score_side(value: float) -> str:
    if not np.isfinite(value):
        return "invalid"
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "mixed"


def _sign_accuracy(predicted_score: float, actual_score_unit: float) -> float:
    if not np.isfinite(predicted_score):
        return 0.0
    predicted_side = _score_side(predicted_score)
    actual_side = _score_side(actual_score_unit)
    if predicted_side == "mixed":
        return float(abs(actual_score_unit) < 0.15)
    return float(predicted_side == actual_side)


def _relation(left: float, right: float, eps: float = 0.05) -> int:
    delta = float(left) - float(right)
    if delta > eps:
        return 1
    if delta < -eps:
        return -1
    return 0


def _make_self_prediction_prompt(positive: str, negative: str, task_prompt: str) -> str:
    labels = "; ".join(
        f"{short_label}={full_label.replace('_', ' ').lower()}"
        for short_label, full_label, _ in _label_map(positive, negative)
    )
    return (
        "Do not answer the user's question yet. "
        f"Predict how your own next answer will sound on the axis {positive} vs {negative}. "
        "Reply with exactly one digit from 1 to 5. "
        f"Use this key: {labels}. "
        f"Question: {task_prompt}"
    )


def _make_other_prediction_prompt(positive: str, negative: str, task_prompt: str, other_frame_text: str) -> str:
    labels = "; ".join(
        f"{short_label}={full_label.replace('_', ' ').lower()}"
        for short_label, full_label, _ in _label_map(positive, negative)
    )
    return (
        "Do not answer the user's question yet. "
        f"Consider a different assistant operating under this system instruction: {other_frame_text} "
        f"Predict how that assistant's next answer would sound on the axis {positive} vs {negative}. "
        "Reply with exactly one digit from 1 to 5. "
        f"Use this key: {labels}. "
        f"Question: {task_prompt}"
    )


def _make_boundary_prompt(
    positive: str,
    negative: str,
    task_prompt: str,
    frame_a_text: str,
    frame_b_text: str,
) -> str:
    labels = "; ".join(
        f"{short_label}={full_label.replace('_', ' ').lower()}"
        for short_label, full_label, _ in _boundary_label_map(positive)
    )
    return (
        "Do not answer the user's question yet. "
        "Imagine two assistants answering the same question. "
        f"Assistant A operates under this system instruction: {frame_a_text} "
        f"Assistant B operates under this system instruction: {frame_b_text} "
        f"On the axis {positive} vs {negative}, which assistant's answer would land further toward {positive}? "
        "Reply with exactly one digit from 1 to 5. "
        f"Use this key: {labels}. "
        f"Question: {task_prompt}"
    )


def _make_label_bias_prompt() -> str:
    return (
        "This is a parser calibration item, not a self-description task. "
        "Reply with exactly one digit from 1 to 5."
    )


def _row_key(
    *,
    seed: int,
    model_id: str,
    identity_frame: str,
    other_frame: str,
    axis_name: str,
    prompt_index: int,
    prompt: str,
) -> tuple[object, ...]:
    return (
        int(seed),
        str(model_id),
        str(identity_frame),
        str(other_frame),
        str(axis_name),
        int(prompt_index),
        str(prompt),
    )


def _load_existing_rows(output_dir: Path, resume_if_exists: bool) -> tuple[list[dict[str, object]], set[tuple[object, ...]]]:
    if not resume_if_exists:
        return [], set()
    for candidate in (output_dir / "results.csv", output_dir / "results.partial.csv"):
        if not candidate.exists():
            continue
        existing = pd.read_csv(candidate)
        rows = existing.to_dict(orient="records")
        completed = {
            _row_key(
                seed=int(row.get("seed", 0)),
                model_id=str(row["model_id"]),
                identity_frame=str(row["identity_frame"]),
                other_frame=str(row["other_frame"]),
                axis_name=str(row["axis_name"]),
                prompt_index=int(row["prompt_index"]),
                prompt=str(row["prompt"]),
            )
            for row in rows
        }
        return rows, completed
    return [], set()


def _write_checkpoint(rows: list[dict[str, object]], output_dir: Path, *, final: bool) -> None:
    if not rows:
        return
    target = output_dir / ("results.csv" if final else "results.partial.csv")
    pd.DataFrame(rows).to_csv(target, index=False)


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
        estimate_axis_vector,
        estimate_layer_scale,
        format_identity_prompt,
        greedy_site_run,
        load_identity_model,
        score_against_axis_anchors,
    )
    from identity_stability.text_features import semantic_overlap, stylometric_distance

    args = parse_args()
    config = _apply_mode_overrides(load_yaml_config(args.config), smoke=args.smoke)
    frames = load_identity_frames()
    prompt_bank = load_self_prediction_items()["dimensions"]
    output_dir = ensure_output_dir(config, "self_other_boundary")
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    stop_strings = resolve_identity_stop_strings(config)
    if stop_strings == ["AUTO"]:
        stop_strings = default_stop_strings_for_template(identity_prompt_template)
    resume_if_exists = bool(config.get("self_other_resume_if_exists", False))
    rows, completed_keys = _load_existing_rows(output_dir, resume_if_exists)
    checkpoint_every_rows = int(config.get("self_other_checkpoint_every_rows", 250))
    pending_rows_since_checkpoint = 0

    axes = [str(x) for x in config.get("self_other_axes", config.get("concept_axes", list(prompt_bank.keys())))]
    prompt_limit = int(config.get("self_other_prompt_limit", config.get("prompt_limit_per_axis", 4)))
    strength = float(config.get("self_other_strength", config["strengths"][-1]))
    seed_values = select_seed_values(config)
    use_label_bias_correction = bool(config.get("self_other_label_bias_correction", True))

    for model_id in config["model_ids"]:
        loaded = load_identity_model(
            model_id=model_id,
            model_cache_dir=config["model_cache_dir"],
            dtype=config["dtype"],
            use_gpu=bool(config["use_gpu"]),
            attention_backend=str(config.get("attention_backend", "auto")),
        )
        fixed_layer = int(round(float(config.get("best_fixed_layer", 0.6)) * max(1, loaded.n_layers - 1)))

        for frame_name in config["identity_frames"]:
            frame_text = frames[frame_name]
            other_frame_name = str(
                config.get("self_other_other_frame_map", {}).get(frame_name, OTHER_FRAME_MAP.get(frame_name, "tool_only"))
            )
            other_frame_text = frames[other_frame_name]

            for seed_value in seed_values:
                for axis_name in axes:
                    positive, negative = axis_sides(axis_name)
                    prompts = [str(x) for x in prompt_bank[axis_name]["prompts"][:prompt_limit]]
                    axis_vector = estimate_axis_vector(
                        loaded=loaded,
                        axis_name=axis_name,
                        layer_index=fixed_layer,
                        token_position=-1,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        seed=int(seed_value),
                        control="mean_diff",
                    )
                    layer_scale = estimate_layer_scale(
                        loaded=loaded,
                        texts=[
                            format_identity_prompt(frame_text, prompt, template=identity_prompt_template)
                            for prompt in prompts[: min(2, len(prompts))]
                        ],
                        layer_index=fixed_layer,
                        token_position=-1,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                    )

                    for prompt_index, prompt in enumerate(prompts):
                        row_key = _row_key(
                            seed=int(seed_value),
                            model_id=model_id,
                            identity_frame=frame_name,
                            other_frame=other_frame_name,
                            axis_name=axis_name,
                            prompt_index=int(prompt_index),
                            prompt=prompt,
                        )
                        if row_key in completed_keys:
                            continue

                        prompt_rng = Random(f"{seed_value}::{model_id}::{frame_name}::{axis_name}::{prompt_index}")
                        label_bias_prompt = (
                            format_identity_prompt(
                                frame_text,
                                _make_label_bias_prompt(),
                                template=identity_prompt_template,
                            )
                            if use_label_bias_correction
                            else None
                        )

                        self_prediction_prompt = format_identity_prompt(
                            frame_text,
                            _make_self_prediction_prompt(positive, negative, prompt),
                            template=identity_prompt_template,
                        )
                        (
                            predicted_self_label,
                            predicted_self_score,
                            predicted_self_confidence,
                            predicted_self_text,
                            predicted_self_score_details,
                        ) = _predict_forced_choice(
                            loaded=loaded,
                            prompt=self_prediction_prompt,
                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                            positive=positive,
                            negative=negative,
                            label_bias_prompt=label_bias_prompt,
                        )

                        other_prediction_prompt = format_identity_prompt(
                            frame_text,
                            _make_other_prediction_prompt(positive, negative, prompt, other_frame_text),
                            template=identity_prompt_template,
                        )
                        (
                            predicted_other_label,
                            predicted_other_score,
                            predicted_other_confidence,
                            predicted_other_text,
                            predicted_other_score_details,
                        ) = _predict_forced_choice(
                            loaded=loaded,
                            prompt=other_prediction_prompt,
                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                            positive=positive,
                            negative=negative,
                            label_bias_prompt=label_bias_prompt,
                        )

                        self_is_a = bool(prompt_rng.randint(0, 1) == 0)
                        frame_a_text, frame_b_text = (
                            (frame_text, other_frame_text) if self_is_a else (other_frame_text, frame_text)
                        )
                        boundary_prompt = format_identity_prompt(
                            frame_text,
                            _make_boundary_prompt(positive, negative, prompt, frame_a_text, frame_b_text),
                            template=identity_prompt_template,
                        )
                        (
                            predicted_boundary_label_raw,
                            predicted_boundary_relation,
                            predicted_boundary_confidence,
                            predicted_boundary_text,
                            predicted_boundary_score_details,
                        ) = _predict_boundary_relation(
                            loaded=loaded,
                            prompt=boundary_prompt,
                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                            positive=positive,
                            label_bias_prompt=label_bias_prompt,
                        )
                        if not self_is_a and np.isfinite(predicted_boundary_relation):
                            predicted_boundary_relation = -float(predicted_boundary_relation)
                        predicted_boundary_label = _normalize_boundary_label(
                            predicted_boundary_label_raw,
                            positive,
                            self_is_a=self_is_a,
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

                        self_axis_score = float(score_against_axis_anchors(axis_name, self_answer.completion_text))
                        other_axis_score = float(score_against_axis_anchors(axis_name, other_answer.completion_text))
                        self_score_unit = _axis_score_to_unit_interval(self_axis_score)
                        other_score_unit = _axis_score_to_unit_interval(other_axis_score)

                        contrary_sign = -1.0 if self_axis_score >= 0.0 else 1.0
                        steered_self_answer = greedy_site_run(
                            loaded=loaded,
                            prompt=format_identity_prompt(frame_text, prompt, template=identity_prompt_template),
                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                            max_new_tokens=int(config["default_generation_tokens"]),
                            injection_site="last_prompt",
                            inject_layer=fixed_layer,
                            inject_vector=axis_vector,
                            inject_scale=float(contrary_sign * strength * layer_scale),
                            stop_strings=stop_strings,
                        )
                        steered_self_axis_score = float(score_against_axis_anchors(axis_name, steered_self_answer.completion_text))
                        steered_self_score_unit = _axis_score_to_unit_interval(steered_self_axis_score)

                        predicted_boundary_margin = float(abs(predicted_boundary_relation))
                        actual_boundary_diff_no_steer = float(self_score_unit - other_score_unit)
                        actual_boundary_diff_steer = float(steered_self_score_unit - other_score_unit)
                        actual_boundary_margin_no_steer = float(abs(actual_boundary_diff_no_steer))
                        actual_boundary_margin_steer = float(abs(actual_boundary_diff_steer))

                        self_adv_no_steer = float(
                            abs(self_score_unit - predicted_other_score) - abs(self_score_unit - predicted_self_score)
                        )
                        self_adv_steer = float(
                            abs(steered_self_score_unit - predicted_other_score) - abs(steered_self_score_unit - predicted_self_score)
                        )
                        other_adv = float(
                            abs(other_score_unit - predicted_self_score) - abs(other_score_unit - predicted_other_score)
                        )

                        actual_boundary_relation_no_steer = float(_relation(actual_boundary_diff_no_steer, 0.0))
                        actual_boundary_relation_steer = float(_relation(actual_boundary_diff_steer, 0.0))
                        boundary_order_match_no_steer = (
                            float(predicted_boundary_relation == actual_boundary_relation_no_steer)
                            if np.isfinite(predicted_boundary_relation)
                            else np.nan
                        )
                        boundary_order_match_steer = (
                            float(predicted_boundary_relation == actual_boundary_relation_steer)
                            if np.isfinite(predicted_boundary_relation)
                            else np.nan
                        )
                        boundary_retention_no_steer = boundary_order_match_no_steer
                        boundary_retention_steer = boundary_order_match_steer
                        boundary_collapse = (
                            float(boundary_retention_no_steer > 0.5 and boundary_retention_steer < 0.5)
                            if np.isfinite(boundary_retention_no_steer) and np.isfinite(boundary_retention_steer)
                            else np.nan
                        )

                        rows.append(
                            {
                                "seed": int(seed_value),
                                "model_id": model_id,
                                "model_family": infer_model_family(model_id),
                                "model_size_label": infer_model_size_label(model_id),
                                "identity_frame": frame_name,
                                "other_frame": other_frame_name,
                                "axis_name": axis_name,
                                "prompt_index": int(prompt_index),
                                "prompt": prompt,
                                "fixed_layer": int(fixed_layer),
                                "contrary_strength": float(contrary_sign * strength),
                                "predicted_self_text": predicted_self_text,
                                "predicted_self_label": predicted_self_label,
                                "predicted_self_score": float(predicted_self_score) if np.isfinite(predicted_self_score) else np.nan,
                                "predicted_self_confidence": float(predicted_self_confidence)
                                if np.isfinite(predicted_self_confidence)
                                else np.nan,
                                "predicted_self_score_details_json": json.dumps(predicted_self_score_details),
                                "predicted_other_text": predicted_other_text,
                                "predicted_other_label": predicted_other_label,
                                "predicted_other_score": float(predicted_other_score) if np.isfinite(predicted_other_score) else np.nan,
                                "predicted_other_confidence": float(predicted_other_confidence)
                                if np.isfinite(predicted_other_confidence)
                                else np.nan,
                                "predicted_other_score_details_json": json.dumps(predicted_other_score_details),
                                "predicted_boundary_text": predicted_boundary_text,
                                "predicted_boundary_label": predicted_boundary_label,
                                "predicted_boundary_label_raw": predicted_boundary_label_raw,
                                "predicted_boundary_score": float(predicted_boundary_relation)
                                if np.isfinite(predicted_boundary_relation)
                                else np.nan,
                                "predicted_boundary_confidence": float(predicted_boundary_confidence)
                                if np.isfinite(predicted_boundary_confidence)
                                else np.nan,
                                "predicted_boundary_score_details_json": json.dumps(predicted_boundary_score_details),
                                "predicted_boundary_relation": float(predicted_boundary_relation)
                                if np.isfinite(predicted_boundary_relation)
                                else np.nan,
                                "boundary_self_is_a": float(self_is_a),
                                "self_text": self_answer.completion_text,
                                "other_text": other_answer.completion_text,
                                "steered_self_text": steered_self_answer.completion_text,
                                "self_axis_score": self_axis_score,
                                "other_axis_score": other_axis_score,
                                "steered_self_axis_score": steered_self_axis_score,
                                "self_score_unit": float(self_score_unit),
                                "other_score_unit": float(other_score_unit),
                                "steered_self_score_unit": float(steered_self_score_unit),
                                "self_side": _axis_score_to_side(self_axis_score, positive=positive, negative=negative),
                                "other_side": _axis_score_to_side(other_axis_score, positive=positive, negative=negative),
                                "steered_self_side": _axis_score_to_side(steered_self_axis_score, positive=positive, negative=negative),
                                "self_sign_accuracy": _sign_accuracy(predicted_self_score, self_score_unit),
                                "other_sign_accuracy": _sign_accuracy(predicted_other_score, other_score_unit),
                                "predicted_boundary_margin": predicted_boundary_margin,
                                "actual_boundary_relation_no_steer": actual_boundary_relation_no_steer,
                                "actual_boundary_relation_steer": actual_boundary_relation_steer,
                                "actual_boundary_margin_no_steer": actual_boundary_margin_no_steer,
                                "actual_boundary_margin_steer": actual_boundary_margin_steer,
                                "boundary_margin_delta": float(actual_boundary_margin_steer - actual_boundary_margin_no_steer),
                                "boundary_order_match_no_steer": boundary_order_match_no_steer,
                                "boundary_order_match_steer": boundary_order_match_steer,
                                "self_prediction_advantage_no_steer": self_adv_no_steer,
                                "self_prediction_advantage_steer": self_adv_steer,
                                "other_prediction_advantage": other_adv,
                                "boundary_retention_no_steer": boundary_retention_no_steer,
                                "boundary_retention_steer": boundary_retention_steer,
                                "boundary_collapse": boundary_collapse,
                                "self_moved_toward_other_actual": float(
                                    abs(steered_self_score_unit - other_score_unit) < abs(self_score_unit - other_score_unit)
                                ),
                                "self_moved_toward_other_prediction": float(
                                    abs(steered_self_score_unit - predicted_other_score)
                                    < abs(self_score_unit - predicted_other_score)
                                ),
                                "self_vs_other_style_distance": float(
                                    stylometric_distance(self_answer.completion_text, other_answer.completion_text)
                                ),
                                "steered_self_vs_other_style_distance": float(
                                    stylometric_distance(steered_self_answer.completion_text, other_answer.completion_text)
                                ),
                                "self_vs_other_semantic_overlap": float(
                                    semantic_overlap(self_answer.completion_text, other_answer.completion_text)
                                ),
                                "steered_self_vs_other_semantic_overlap": float(
                                    semantic_overlap(steered_self_answer.completion_text, other_answer.completion_text)
                                ),
                                "candidate_frames_json": json.dumps(
                                    {
                                        "self_frame": frame_name,
                                        "other_frame": other_frame_name,
                                        "frame_a": frame_a_text,
                                        "frame_b": frame_b_text,
                                    }
                                ),
                            }
                        )
                        completed_keys.add(row_key)
                        pending_rows_since_checkpoint += 1
                        if pending_rows_since_checkpoint >= checkpoint_every_rows:
                            _write_checkpoint(rows, output_dir, final=False)
                            pending_rows_since_checkpoint = 0

        del loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)
    partial_path = output_dir / "results.partial.csv"
    if partial_path.exists():
        partial_path.unlink()

    summary = (
        df.groupby(["model_size_label", "identity_frame", "axis_name"], as_index=False)
        .agg(
            self_sign_accuracy_mean=("self_sign_accuracy", "mean"),
            other_sign_accuracy_mean=("other_sign_accuracy", "mean"),
            predicted_boundary_margin_mean=("predicted_boundary_margin", "mean"),
            predicted_boundary_confidence_mean=("predicted_boundary_confidence", "mean"),
            actual_boundary_margin_no_steer_mean=("actual_boundary_margin_no_steer", "mean"),
            actual_boundary_margin_steer_mean=("actual_boundary_margin_steer", "mean"),
            boundary_margin_delta_mean=("boundary_margin_delta", "mean"),
            boundary_order_match_no_steer_mean=("boundary_order_match_no_steer", "mean"),
            boundary_order_match_steer_mean=("boundary_order_match_steer", "mean"),
            boundary_retention_no_steer_mean=("boundary_retention_no_steer", "mean"),
            boundary_retention_steer_mean=("boundary_retention_steer", "mean"),
            boundary_collapse_rate=("boundary_collapse", "mean"),
            boundary_prediction_valid_rate=("predicted_boundary_relation", lambda s: float(s.notna().mean())),
            self_prediction_advantage_no_steer_mean=("self_prediction_advantage_no_steer", "mean"),
            self_prediction_advantage_steer_mean=("self_prediction_advantage_steer", "mean"),
            other_prediction_advantage_mean=("other_prediction_advantage", "mean"),
            self_moved_toward_other_actual_mean=("self_moved_toward_other_actual", "mean"),
            self_moved_toward_other_prediction_mean=("self_moved_toward_other_prediction", "mean"),
            self_vs_other_style_distance_mean=("self_vs_other_style_distance", "mean"),
            steered_self_vs_other_style_distance_mean=("steered_self_vs_other_style_distance", "mean"),
            self_vs_other_semantic_overlap_mean=("self_vs_other_semantic_overlap", "mean"),
            steered_self_vs_other_semantic_overlap_mean=("steered_self_vs_other_semantic_overlap", "mean"),
            seed_count=("seed", "nunique"),
            n=("prompt", "count"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    summary_by_model_frame = (
        df.groupby(["model_size_label", "identity_frame"], as_index=False)
        .agg(
            self_sign_accuracy_mean=("self_sign_accuracy", "mean"),
            other_sign_accuracy_mean=("other_sign_accuracy", "mean"),
            predicted_boundary_margin_mean=("predicted_boundary_margin", "mean"),
            boundary_retention_no_steer_mean=("boundary_retention_no_steer", "mean"),
            boundary_retention_steer_mean=("boundary_retention_steer", "mean"),
            boundary_collapse_rate=("boundary_collapse", "mean"),
            boundary_prediction_valid_rate=("predicted_boundary_relation", lambda s: float(s.notna().mean())),
            self_prediction_advantage_no_steer_mean=("self_prediction_advantage_no_steer", "mean"),
            self_prediction_advantage_steer_mean=("self_prediction_advantage_steer", "mean"),
            self_moved_toward_other_actual_mean=("self_moved_toward_other_actual", "mean"),
            self_vs_other_style_distance_mean=("self_vs_other_style_distance", "mean"),
            steered_self_vs_other_style_distance_mean=("steered_self_vs_other_style_distance", "mean"),
            seed_count=("seed", "nunique"),
            n=("prompt", "count"),
        )
    )
    summary_by_model_frame.to_csv(output_dir / "summary_by_model_frame.csv", index=False)

    summary_by_model = (
        df.groupby(["model_size_label"], as_index=False)
        .agg(
            self_sign_accuracy_mean=("self_sign_accuracy", "mean"),
            other_sign_accuracy_mean=("other_sign_accuracy", "mean"),
            boundary_retention_no_steer_mean=("boundary_retention_no_steer", "mean"),
            boundary_retention_steer_mean=("boundary_retention_steer", "mean"),
            boundary_collapse_rate=("boundary_collapse", "mean"),
            boundary_prediction_valid_rate=("predicted_boundary_relation", lambda s: float(s.notna().mean())),
            self_moved_toward_other_actual_mean=("self_moved_toward_other_actual", "mean"),
            self_prediction_advantage_no_steer_mean=("self_prediction_advantage_no_steer", "mean"),
            self_prediction_advantage_steer_mean=("self_prediction_advantage_steer", "mean"),
            seed_count=("seed", "nunique"),
            n=("prompt", "count"),
        )
    )
    summary_by_model.to_csv(output_dir / "summary_by_model.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Self Other Boundary\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Smoke mode: `{args.smoke}`\n")
        f.write(f"- Rows: `{len(df)}`\n")
        f.write(f"- Seeds: `{seed_values}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(f"- Identity stop strings: `{stop_strings}`\n")
        f.write(
            "- Purpose: test whether the model keeps a discriminable self/other answer boundary under identity framing and whether contrary steering collapses that boundary.\n"
        )


if __name__ == "__main__":
    main()
