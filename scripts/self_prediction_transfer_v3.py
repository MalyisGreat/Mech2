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
    select_seed_values,
)


DEFAULT_OTHER_DESCRIPTION = "the exact same model weights answering the same question in a different chat with no access to this conversation"

FEATURE_SPECS = {
    "ELABORATE": {
        "feature_key": "avg_sentence_length",
        "present_threshold": 11.0,
        "question": "give a relatively elaborate answer rather than a terse one",
    },
    "HEDGE": {
        "feature_key": "hedge_density",
        "present_threshold": 0.02,
        "question": "use noticeable hedging or caution words such as maybe, likely, or could",
    },
    "FIRST_PERSON": {
        "feature_key": "first_person_rate",
        "present_threshold": 0.005,
        "question": "use first-person words such as I, me, my, or we",
    },
    "DIRECTIVE": {
        "feature_key": "directive_rate",
        "present_threshold": 0.01,
        "question": "give direct instructions or imperatives",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-conditional self-vs-other prediction transfer checks (v3).")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _yes_no_labels() -> list[tuple[str, str, float]]:
    return [("YES", "YES", 1.0), ("NO", "NO", 0.0)]


def _make_binary_prediction_prompt(
    *,
    target_name: str,
    target_description: str,
    task_prompt: str,
    feature_question: str,
) -> str:
    return (
        "You are predicting one concrete style property of a future answer.\n"
        f"Target answerer: {target_name}. {target_description}\n"
        f"Question they will answer: {task_prompt}\n"
        f"Property to predict: Will that answer {feature_question}?\n\n"
        "Do not answer the question itself.\n"
        "Reply with exactly one label from: YES, NO.\n"
        "Label:"
    )


def _make_yes_no_bias_prompt() -> str:
    return "Formatting-only task.\nReply with exactly one label from: YES, NO.\nLabel:"


def _feature_present(value: float, threshold: float) -> bool:
    return float(value) >= float(threshold)


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bootstrap_mean_ci(values: list[float]) -> tuple[float, float]:
    add_src_to_path()
    from identity_stability.identity_analysis import bootstrap_mean_ci

    return bootstrap_mean_ci(values, iters=2000, seed=123)


def _summary_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        "self_accuracy_mean",
        "other_accuracy_mean",
        "gap_direction_accuracy_mean",
        "valid_choice_rate",
        "predicted_gap_rate",
        "actual_gap_rate",
        "discriminative_win",
    ]
    rows: list[dict[str, object]] = []
    for keys, sub in df.groupby(group_cols, as_index=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["n"] = int(len(sub))
        row["seed_count"] = int(sub["seed"].nunique()) if "seed" in sub.columns else 1
        for metric in metric_cols:
            values = sub[metric].dropna().astype(float).tolist()
            row[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
            ci_low, ci_high = _bootstrap_mean_ci(values) if values else (float("nan"), float("nan"))
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_optional_path(repo_root: Path, value: object) -> Path | None:
    if value is None:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import load_identity_frames, load_self_prediction_items_v2, load_yaml_file
    from identity_stability.identity_probe_tools import predict_labeled_choice_batch
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        default_stop_strings_for_template,
        format_identity_prompt,
        generate_completion_texts_batch,
        load_identity_model,
    )
    from identity_stability.text_features import extract_style_features

    args = parse_args()
    config = load_yaml_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    frames_path = _resolve_optional_path(repo_root, config.get("identity_frames_path"))
    prompt_items_path = _resolve_optional_path(repo_root, config.get("self_prediction_items_path"))
    frames = (
        {str(k): str(v) for k, v in load_yaml_file(frames_path).items()}
        if frames_path is not None
        else load_identity_frames()
    )
    bank = dict(load_yaml_file(prompt_items_path)) if prompt_items_path is not None else load_self_prediction_items_v2()
    output_dir = ensure_output_dir(config, "self_prediction_transfer_v3")
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    stop_strings = resolve_identity_stop_strings(config)
    if stop_strings == ["AUTO"]:
        stop_strings = default_stop_strings_for_template(identity_prompt_template)

    prompt_items = [dict(item) for item in bank["items"][: int(config.get("self_prediction_v3_prompt_limit", len(bank["items"])))]]
    yes_no_labels = _yes_no_labels()
    generation_tokens = int(config.get("default_generation_tokens", 96))
    generation_do_sample = bool(config.get("generation_do_sample", True))
    generation_temperature = float(config.get("generation_temperature", 0.8))
    generation_top_p = float(config.get("generation_top_p", 0.9))
    generation_top_k = int(config.get("generation_top_k", 0))
    generation_presence_penalty = float(config.get("generation_presence_penalty", 0.0))

    rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

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
            other_description = str(config.get("self_prediction_v3_other_description", DEFAULT_OTHER_DESCRIPTION))

            for seed_value in select_seed_values(config):
                for prompt_item in prompt_items:
                    prompt_id = str(prompt_item["id"])
                    prompt_text = str(prompt_item["prompt"])
                    self_answer_prompt = format_identity_prompt(frame_text, prompt_text, template=identity_prompt_template)
                    other_answer_prompt = format_identity_prompt(frame_text, prompt_text, template=identity_prompt_template)

                    actual_texts = generate_completion_texts_batch(
                        loaded=loaded,
                        prompts=[self_answer_prompt, other_answer_prompt],
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=generation_tokens,
                        stop_strings=stop_strings,
                        do_sample=generation_do_sample,
                        temperature=generation_temperature,
                        top_p=generation_top_p,
                        top_k=generation_top_k,
                        presence_penalty=generation_presence_penalty,
                        sampling_seeds=[int(seed_value), int(seed_value) + 10000],
                    )

                    self_actual_text, other_actual_text = actual_texts
                    self_actual_features = extract_style_features(self_actual_text)
                    other_actual_features = extract_style_features(other_actual_text)

                    prediction_prompts: list[str] = []
                    prediction_metadata: list[dict[str, object]] = []
                    for feature_name, spec in FEATURE_SPECS.items():
                        for target_name, target_description in (
                            ("SELF", "This is you yourself in this exact chat."),
                            ("OTHER", f"This is {other_description}."),
                        ):
                            prediction_prompts.append(
                                format_identity_prompt(
                                    frame_text,
                                    _make_binary_prediction_prompt(
                                        target_name=target_name,
                                        target_description=target_description,
                                        task_prompt=prompt_text,
                                        feature_question=str(spec["question"]),
                                    ),
                                    template=identity_prompt_template,
                                )
                            )
                            prediction_metadata.append(
                                {
                                    "feature_name": feature_name,
                                    "target_name": target_name,
                                    "feature_key": str(spec["feature_key"]),
                                    "present_threshold": float(spec["present_threshold"]),
                                }
                            )

                    predictions = predict_labeled_choice_batch(
                        loaded=loaded,
                        prompts=prediction_prompts,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        labels=yes_no_labels,
                        label_bias_prompts=[_make_yes_no_bias_prompt() for _ in prediction_prompts],
                    )

                    feature_result_map: dict[str, dict[str, object]] = {}
                    self_accuracy_values: list[float] = []
                    other_accuracy_values: list[float] = []
                    gap_accuracy_values: list[float] = []
                    valid_choice_values: list[float] = []
                    predicted_gap_values: list[float] = []
                    actual_gap_values: list[float] = []

                    grouped_predictions: dict[str, dict[str, object]] = {}
                    for meta, prediction in zip(prediction_metadata, predictions):
                        selected_short_label, selected_label, _, selected_prob, completion_text, details = prediction
                        feature_name = str(meta["feature_name"])
                        target_name = str(meta["target_name"])
                        feature_key = str(meta["feature_key"])
                        threshold = float(meta["present_threshold"])
                        actual_value = _safe_float(
                            self_actual_features[feature_key] if target_name == "SELF" else other_actual_features[feature_key]
                        )
                        actual_present = _feature_present(actual_value, threshold)
                        valid_choice = float(selected_label in {"YES", "NO"})
                        predicted_present = selected_label == "YES"
                        correct = float(predicted_present == actual_present) if valid_choice == 1.0 else float("nan")

                        grouped_predictions.setdefault(feature_name, {})[target_name] = {
                            "selected_short_label": selected_short_label,
                            "selected_label": selected_label,
                            "valid_choice": valid_choice,
                            "predicted_present": predicted_present if valid_choice == 1.0 else None,
                            "actual_present": actual_present,
                            "correct": correct,
                            "selected_prob": float(selected_prob) if np.isfinite(selected_prob) else float("nan"),
                            "actual_feature_value": actual_value,
                            "completion_text": completion_text,
                            "details": details,
                        }

                    for feature_name, spec in FEATURE_SPECS.items():
                        feature_predictions = grouped_predictions.get(feature_name, {})
                        self_pred = dict(feature_predictions.get("SELF", {}))
                        other_pred = dict(feature_predictions.get("OTHER", {}))

                        self_valid = float(self_pred.get("valid_choice", 0.0))
                        other_valid = float(other_pred.get("valid_choice", 0.0))
                        pair_valid = float(self_valid == 1.0 and other_valid == 1.0)
                        valid_choice_values.append(pair_valid)

                        self_accuracy = _safe_float(self_pred.get("correct"))
                        other_accuracy = _safe_float(other_pred.get("correct"))
                        if np.isfinite(self_accuracy):
                            self_accuracy_values.append(self_accuracy)
                        if np.isfinite(other_accuracy):
                            other_accuracy_values.append(other_accuracy)

                        self_actual_present = bool(self_pred.get("actual_present", False))
                        other_actual_present = bool(other_pred.get("actual_present", False))
                        actual_gap = float(self_actual_present != other_actual_present)
                        actual_gap_values.append(actual_gap)

                        if pair_valid == 1.0:
                            predicted_gap = float(bool(self_pred.get("predicted_present")) != bool(other_pred.get("predicted_present")))
                            predicted_gap_values.append(predicted_gap)
                        else:
                            predicted_gap = float("nan")

                        if actual_gap == 1.0 and pair_valid == 1.0:
                            actual_gap_label = "SELF" if self_actual_present and not other_actual_present else "OTHER"
                            predicted_gap_label = (
                                "SELF"
                                if bool(self_pred.get("predicted_present")) and not bool(other_pred.get("predicted_present"))
                                else "OTHER"
                            )
                            gap_correct = float(predicted_gap_label == actual_gap_label)
                            gap_accuracy_values.append(gap_correct)
                        else:
                            actual_gap_label = "TIE"
                            predicted_gap_label = (
                                "TIE"
                                if not np.isfinite(predicted_gap) or predicted_gap == 0.0
                                else (
                                    "SELF"
                                    if bool(self_pred.get("predicted_present")) and not bool(other_pred.get("predicted_present"))
                                    else "OTHER"
                                )
                            )
                            gap_correct = float("nan")

                        feature_result_map[feature_name] = {
                            "feature_name": feature_name,
                            "feature_key": str(spec["feature_key"]),
                            "self_selected_label": self_pred.get("selected_label", "INVALID"),
                            "self_valid_choice": self_valid,
                            "self_correct": self_accuracy,
                            "self_actual_present": float(self_actual_present),
                            "self_selected_prob": _safe_float(self_pred.get("selected_prob")),
                            "self_actual_feature_value": _safe_float(self_pred.get("actual_feature_value")),
                            "other_selected_label": other_pred.get("selected_label", "INVALID"),
                            "other_valid_choice": other_valid,
                            "other_correct": other_accuracy,
                            "other_actual_present": float(other_actual_present),
                            "other_selected_prob": _safe_float(other_pred.get("selected_prob")),
                            "other_actual_feature_value": _safe_float(other_pred.get("actual_feature_value")),
                            "actual_gap": actual_gap,
                            "predicted_gap": predicted_gap,
                            "actual_gap_label": actual_gap_label,
                            "predicted_gap_label": predicted_gap_label,
                            "gap_correct": gap_correct,
                            "self_completion_text": str(self_pred.get("completion_text", "")),
                            "other_completion_text": str(other_pred.get("completion_text", "")),
                            "self_details": self_pred.get("details", {}),
                            "other_details": other_pred.get("details", {}),
                        }

                        feature_rows.append(
                            {
                                "seed": int(seed_value),
                                "model_id": model_id,
                                "model_family": infer_model_family(model_id),
                                "model_size_label": infer_model_size_label(model_id),
                                "identity_frame": frame_name,
                                "other_frame": "same_weights_other_chat",
                                "prompt_id": prompt_id,
                                "prompt_family": str(prompt_item["family"]),
                                "prompt": prompt_text,
                                "feature_name": feature_name,
                                "feature_key": str(spec["feature_key"]),
                                "self_selected_label": self_pred.get("selected_label", "INVALID"),
                                "self_valid_choice": self_valid,
                                "self_correct": self_accuracy,
                                "self_actual_present": float(self_actual_present),
                                "self_selected_prob": _safe_float(self_pred.get("selected_prob")),
                                "self_actual_feature_value": _safe_float(self_pred.get("actual_feature_value")),
                                "other_selected_label": other_pred.get("selected_label", "INVALID"),
                                "other_valid_choice": other_valid,
                                "other_correct": other_accuracy,
                                "other_actual_present": float(other_actual_present),
                                "other_selected_prob": _safe_float(other_pred.get("selected_prob")),
                                "other_actual_feature_value": _safe_float(other_pred.get("actual_feature_value")),
                                "actual_gap": actual_gap,
                                "predicted_gap": predicted_gap,
                                "actual_gap_label": actual_gap_label,
                                "predicted_gap_label": predicted_gap_label,
                                "gap_correct": gap_correct,
                                "self_completion_text": str(self_pred.get("completion_text", "")),
                                "other_completion_text": str(other_pred.get("completion_text", "")),
                                "self_details_json": json.dumps(self_pred.get("details", {})),
                                "other_details_json": json.dumps(other_pred.get("details", {})),
                            }
                        )

                    informative_gap_count = int(sum(value == 1.0 for value in actual_gap_values))
                    discriminative_win = float(
                        informative_gap_count >= 2
                        and len(gap_accuracy_values) == informative_gap_count
                        and all(value == 1.0 for value in gap_accuracy_values)
                    )

                    rows.append(
                        {
                            "seed": int(seed_value),
                            "model_id": model_id,
                            "model_family": infer_model_family(model_id),
                            "model_size_label": infer_model_size_label(model_id),
                            "identity_frame": frame_name,
                            "other_frame": "same_weights_other_chat",
                            "other_target_description": other_description,
                            "prompt_id": prompt_id,
                            "prompt_family": str(prompt_item["family"]),
                            "prompt": prompt_text,
                            "self_accuracy_mean": float(np.mean(self_accuracy_values)) if self_accuracy_values else float("nan"),
                            "other_accuracy_mean": float(np.mean(other_accuracy_values)) if other_accuracy_values else float("nan"),
                            "gap_direction_accuracy_mean": float(np.mean(gap_accuracy_values)) if gap_accuracy_values else float("nan"),
                            "valid_choice_rate": float(np.mean(valid_choice_values)) if valid_choice_values else float("nan"),
                            "predicted_gap_rate": float(np.mean(predicted_gap_values)) if predicted_gap_values else float("nan"),
                            "actual_gap_rate": float(np.mean(actual_gap_values)) if actual_gap_values else float("nan"),
                            "informative_gap_count": informative_gap_count,
                            "discriminative_win": discriminative_win,
                            "feature_results_json": json.dumps(feature_result_map),
                            "self_actual_features_json": json.dumps(self_actual_features),
                            "other_actual_features_json": json.dumps(other_actual_features),
                            "self_actual_text": self_actual_text,
                            "other_actual_text": other_actual_text,
                        }
                    )

        del loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(output_dir / "feature_results.csv", index=False)

    summary_by_model_frame = _summary_table(df, ["model_size_label", "identity_frame"])
    summary_by_model_frame.to_csv(output_dir / "summary_by_model_frame.csv", index=False)
    summary_by_family = _summary_table(df, ["model_size_label", "identity_frame", "prompt_family"])
    summary_by_family.to_csv(output_dir / "summary_by_family.csv", index=False)

    feature_summary_input = feature_df.assign(
        self_accuracy_mean=feature_df["self_correct"],
        other_accuracy_mean=feature_df["other_correct"],
        gap_direction_accuracy_mean=feature_df["gap_correct"],
        valid_choice_rate=(feature_df["self_valid_choice"].astype(float) * feature_df["other_valid_choice"].astype(float)),
        predicted_gap_rate=feature_df["predicted_gap"],
        actual_gap_rate=feature_df["actual_gap"],
        discriminative_win=np.nan,
    )
    summary_by_feature = _summary_table(feature_summary_input, ["model_size_label", "identity_frame", "feature_name"])
    summary_by_feature.to_csv(output_dir / "summary_by_feature.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Self Prediction Transfer V3\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Prompt rows: `{len(df)}`\n")
        f.write(f"- Feature rows: `{len(feature_df)}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(
            "- Purpose: measure prompt-conditional self-vs-other forecasting with paired binary predictions over concrete style features, "
            "using a same-weights other-chat comparison rather than an explicitly reframed assistant target.\n"
        )


if __name__ == "__main__":
    main()
