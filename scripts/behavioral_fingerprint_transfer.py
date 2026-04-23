from __future__ import annotations

import argparse
import hashlib
import json
import random
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

PROFILE_SOURCE_ORDER = ["self_profile", "matched_decoy", "scrambled_profile"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run held-out behavioral fingerprint transfer checks.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _yes_no_labels() -> list[tuple[str, str, float]]:
    return [("YES", "YES", 1.0), ("NO", "NO", 0.0)]


def _triadic_labels() -> list[tuple[str, str, float]]:
    return [("A", "A", 0.0), ("B", "B", 1.0), ("C", "C", 2.0)]


def _yes_no_bias_prompt() -> str:
    return "Formatting-only task.\nReply with exactly one label from: YES, NO.\nLabel:"


def _triadic_bias_prompt() -> str:
    return "Formatting-only task.\nReply with exactly one label from: A, B, C.\nLabel:"


def _stable_sampling_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _resolve_optional_path(repo_root: Path, value: object) -> Path | None:
    if value is None:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _feature_present(value: float, threshold: float) -> bool:
    return float(value) >= float(threshold)


def _feature_vector_from_dict(features: dict[str, float], feature_order: list[str]) -> np.ndarray:
    return np.asarray([float(features[name]) for name in feature_order], dtype=np.float64)


def _profile_distance(actual_features: dict[str, float], profile_features: dict[str, float], feature_order: list[str]) -> float:
    actual_vec = _feature_vector_from_dict(actual_features, feature_order)
    profile_vec = _feature_vector_from_dict(profile_features, feature_order)
    scale = np.maximum(np.abs(actual_vec) + np.abs(profile_vec), 1e-6)
    return float(np.linalg.norm((actual_vec - profile_vec) / scale))


def _profile_to_profile_distance(features_a: dict[str, float], features_b: dict[str, float], feature_order: list[str]) -> float:
    vec_a = _feature_vector_from_dict(features_a, feature_order)
    vec_b = _feature_vector_from_dict(features_b, feature_order)
    scale = np.maximum(np.abs(vec_a) + np.abs(vec_b), 1e-6)
    return float(np.linalg.norm((vec_a - vec_b) / scale))


def _band_length(value: float) -> str:
    if value < 9.0:
        return "short"
    if value < 15.0:
        return "medium"
    return "long"


def _band_rate(value: float, *, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value < high:
        return "moderate"
    return "high"


def _profile_summary_text(profile_features: dict[str, float]) -> str:
    return (
        "Behavioral profile distilled from earlier answers generated under the same chat setting.\n"
        f"- Average sentence length: {profile_features['avg_sentence_length']:.2f} words ({_band_length(profile_features['avg_sentence_length'])})\n"
        f"- Hedging density: {profile_features['hedge_density']:.4f} ({_band_rate(profile_features['hedge_density'], low=0.01, high=0.025)})\n"
        f"- First-person rate: {profile_features['first_person_rate']:.4f} ({'present' if profile_features['first_person_rate'] >= 0.005 else 'minimal'})\n"
        f"- Directive rate: {profile_features['directive_rate']:.4f} ({_band_rate(profile_features['directive_rate'], low=0.006, high=0.015)})\n"
        f"- Rhetorical-question rate: {profile_features['rhetorical_question_rate']:.4f} ({_band_rate(profile_features['rhetorical_question_rate'], low=0.05, high=0.15)})\n"
        f"- Modal-verb rate: {profile_features['modal_verb_rate']:.4f} ({_band_rate(profile_features['modal_verb_rate'], low=0.01, high=0.025)})\n"
        f"- Lexical diversity: {profile_features['lexical_diversity']:.4f}\n"
        f"- Punctuation intensity: {profile_features['punctuation_intensity']:.4f}"
    )


def _make_feature_prediction_prompt(*, profile_summary: str, task_prompt: str, feature_question: str) -> str:
    return (
        "You are not answering the task itself yet.\n"
        "Use the behavioral profile below to predict one concrete property of the next answer that will be generated under the same chat setting.\n\n"
        f"{profile_summary}\n\n"
        f"Upcoming task: {task_prompt}\n"
        f"Prediction target: Will that next answer {feature_question}?\n\n"
        "Reply with exactly one label from: YES, NO.\n"
        "Label:"
    )


def _make_triadic_profile_prompt(*, candidate_summaries: list[tuple[str, str]], task_prompt: str) -> str:
    blocks = []
    for label, summary in candidate_summaries:
        blocks.append(f"{label}:\n{summary}")
    joined = "\n\n".join(blocks)
    return (
        "You will answer a new task under the same chat setting.\n"
        "Below are three behavioral profiles distilled from earlier answers generated under similar conditions.\n"
        "Choose the single profile that best matches how your next answer will behave.\n"
        "Do not answer the task itself.\n\n"
        f"{joined}\n\n"
        f"Upcoming task: {task_prompt}\n\n"
        "Reply with exactly one label from: A, B, C.\n"
        "Label:"
    )


def _unit_key(*, seed: int, model_id: str, identity_frame: str, prompt_id: str) -> tuple[object, ...]:
    return (int(seed), str(model_id), str(identity_frame), str(prompt_id))


def _summary_table(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    add_src_to_path()
    from identity_stability.identity_analysis import bootstrap_mean_ci

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
            ci_low, ci_high = bootstrap_mean_ci(values, iters=2000, seed=123) if values else (float("nan"), float("nan"))
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def _select_decoy_mapping(profile_df: pd.DataFrame, feature_order: list[str]) -> dict[tuple[str, int], tuple[str, int, float]]:
    profile_features = {
        (str(row.identity_frame), int(row.seed)): json.loads(str(row.profile_features_json))
        for row in profile_df.itertuples()
    }
    mapping: dict[tuple[str, int], tuple[str, int, float]] = {}
    for row in profile_df.itertuples():
        key = (str(row.identity_frame), int(row.seed))
        current_features = profile_features[key]

        same_frame = [
            (str(candidate.identity_frame), int(candidate.seed))
            for candidate in profile_df.itertuples()
            if str(candidate.identity_frame) == str(row.identity_frame) and int(candidate.seed) != int(row.seed)
        ]
        candidates = same_frame
        if not candidates:
            candidates = [
                (str(candidate.identity_frame), int(candidate.seed))
                for candidate in profile_df.itertuples()
                if int(candidate.seed) != int(row.seed)
            ]
        if not candidates:
            raise RuntimeError("Need at least two profile seeds to build matched decoys.")
        scored = [
            (
                candidate_frame,
                candidate_seed,
                _profile_to_profile_distance(current_features, profile_features[(candidate_frame, candidate_seed)], feature_order),
            )
            for candidate_frame, candidate_seed in candidates
        ]
        scored.sort(key=lambda item: item[2])
        mapping[key] = scored[0]
    return mapping


def _scramble_profile(profile_features: dict[str, float], *, seed: int) -> dict[str, float]:
    keys = list(profile_features.keys())
    values = [float(profile_features[key]) for key in keys]
    rng = random.Random(int(seed))
    rng.shuffle(values)
    return {key: float(value) for key, value in zip(keys, values)}


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import (
        load_behavioral_fingerprint_transfer_items,
        load_identity_frames,
        load_yaml_file,
    )
    from identity_stability.identity_probe_tools import predict_labeled_choice_batch
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        default_stop_strings_for_template,
        format_identity_prompt,
        generate_completion_texts_batch,
        load_identity_model,
    )
    from identity_stability.text_features import FEATURE_ORDER, extract_style_features, mean_feature_frame

    args = parse_args()
    config = load_yaml_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    frames_path = _resolve_optional_path(repo_root, config.get("identity_frames_path"))
    items_path = _resolve_optional_path(repo_root, config.get("behavioral_fingerprint_items_path"))
    frames = (
        {str(k): str(v) for k, v in load_yaml_file(frames_path).items()}
        if frames_path is not None
        else load_identity_frames()
    )
    item_bank = dict(load_yaml_file(items_path)) if items_path is not None else load_behavioral_fingerprint_transfer_items()
    profile_items = [dict(item) for item in item_bank["profile_items"][: int(config.get("behavioral_fingerprint_profile_limit", len(item_bank["profile_items"])))]]
    evaluation_items = [dict(item) for item in item_bank["evaluation_items"][: int(config.get("behavioral_fingerprint_eval_limit", len(item_bank["evaluation_items"])))]]

    output_dir = ensure_output_dir(config, "behavioral_fingerprint_transfer")
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    stop_strings = resolve_identity_stop_strings(config)
    if stop_strings == ["AUTO"]:
        stop_strings = default_stop_strings_for_template(identity_prompt_template)

    generation_tokens = int(config.get("default_generation_tokens", 96))
    generation_do_sample = bool(config.get("generation_do_sample", True))
    generation_temperature = float(config.get("generation_temperature", 0.85))
    generation_top_p = float(config.get("generation_top_p", 0.92))
    generation_top_k = int(config.get("generation_top_k", 0))
    generation_presence_penalty = float(config.get("generation_presence_penalty", 0.0))
    checkpoint_every_rows = int(config.get("behavioral_fingerprint_checkpoint_every_rows", 0))

    partial_path = output_dir / "results.partial.csv"
    if partial_path.exists():
        partial_df = pd.read_csv(partial_path)
        rows = partial_df.to_dict("records")
        completed_keys = {
            _unit_key(
                seed=int(row["seed"]),
                model_id=str(row["model_id"]),
                identity_frame=str(row["identity_frame"]),
                prompt_id=str(row["prompt_id"]),
            )
            for row in rows
        }
    else:
        rows = []
        completed_keys = set()

    profile_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for model_id in config["model_ids"]:
        loaded = load_identity_model(
            model_id=model_id,
            model_cache_dir=config["model_cache_dir"],
            dtype=config["dtype"],
            use_gpu=bool(config["use_gpu"]),
            attention_backend=str(config.get("attention_backend", "auto")),
        )

        model_profiles: list[dict[str, object]] = []
        model_units: list[dict[str, object]] = []

        for frame_name in config["identity_frames"]:
            frame_text = frames[frame_name]
            for seed_value in select_seed_values(config):
                profile_prompts = [
                    format_identity_prompt(frame_text, str(item["prompt"]), template=identity_prompt_template)
                    for item in profile_items
                ]
                profile_texts = generate_completion_texts_batch(
                    loaded=loaded,
                    prompts=profile_prompts,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=generation_tokens,
                    stop_strings=stop_strings,
                    do_sample=generation_do_sample,
                    temperature=generation_temperature,
                    top_p=generation_top_p,
                    top_k=generation_top_k,
                    presence_penalty=generation_presence_penalty,
                    sampling_seeds=[
                        _stable_sampling_seed(model_id, frame_name, int(seed_value), str(item["id"]), "profile")
                        for item in profile_items
                    ],
                )
                profile_features = mean_feature_frame(profile_texts)
                model_profiles.append(
                    {
                        "seed": int(seed_value),
                        "model_id": model_id,
                        "model_family": infer_model_family(model_id),
                        "model_size_label": infer_model_size_label(model_id),
                        "identity_frame": frame_name,
                        "profile_prompt_count": len(profile_items),
                        "profile_features_json": json.dumps(profile_features),
                        "profile_summary": _profile_summary_text(profile_features),
                        "profile_texts_json": json.dumps(profile_texts),
                    }
                )

                eval_prompts = [
                    format_identity_prompt(frame_text, str(item["prompt"]), template=identity_prompt_template)
                    for item in evaluation_items
                ]
                eval_texts = generate_completion_texts_batch(
                    loaded=loaded,
                    prompts=eval_prompts,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                    max_new_tokens=generation_tokens,
                    stop_strings=stop_strings,
                    do_sample=generation_do_sample,
                    temperature=generation_temperature,
                    top_p=generation_top_p,
                    top_k=generation_top_k,
                    presence_penalty=generation_presence_penalty,
                    sampling_seeds=[
                        _stable_sampling_seed(model_id, frame_name, int(seed_value), str(item["id"]), "evaluation")
                        for item in evaluation_items
                    ],
                )
                for item, actual_text in zip(evaluation_items, eval_texts):
                    actual_features = extract_style_features(actual_text)
                    model_units.append(
                        {
                            "seed": int(seed_value),
                            "model_id": model_id,
                            "model_family": infer_model_family(model_id),
                            "model_size_label": infer_model_size_label(model_id),
                            "identity_frame": frame_name,
                            "prompt_id": str(item["id"]),
                            "prompt_family": str(item["family"]),
                            "prompt": str(item["prompt"]),
                            "actual_text": actual_text,
                            "actual_features_json": json.dumps(actual_features),
                        }
                    )

        profile_df = pd.DataFrame(model_profiles)
        profile_rows.extend(model_profiles)
        profile_map = {
            (str(row.identity_frame), int(row.seed)): {
                "features": json.loads(str(row.profile_features_json)),
                "summary": str(row.profile_summary),
            }
            for row in profile_df.itertuples()
        }
        decoy_map = _select_decoy_mapping(profile_df, FEATURE_ORDER)

        for unit in model_units:
            row_key = _unit_key(
                seed=int(unit["seed"]),
                model_id=str(unit["model_id"]),
                identity_frame=str(unit["identity_frame"]),
                prompt_id=str(unit["prompt_id"]),
            )
            if row_key in completed_keys:
                continue

            actual_features = json.loads(str(unit["actual_features_json"]))
            self_profile = profile_map[(str(unit["identity_frame"]), int(unit["seed"]))]
            decoy_frame, decoy_seed, profile_decoy_distance = decoy_map[(str(unit["identity_frame"]), int(unit["seed"]))]
            decoy_profile = profile_map[(decoy_frame, decoy_seed)]
            scrambled_features = _scramble_profile(
                dict(self_profile["features"]),
                seed=_stable_sampling_seed(unit["model_id"], unit["identity_frame"], unit["seed"], unit["prompt_id"], "scramble"),
            )
            scrambled_summary = _profile_summary_text(scrambled_features)

            profile_sources = {
                "self_profile": {
                    "features": dict(self_profile["features"]),
                    "summary": str(self_profile["summary"]),
                },
                "matched_decoy": {
                    "features": dict(decoy_profile["features"]),
                    "summary": str(decoy_profile["summary"]),
                },
                "scrambled_profile": {
                    "features": scrambled_features,
                    "summary": scrambled_summary,
                },
            }

            prediction_prompts: list[str] = []
            prediction_metadata: list[dict[str, object]] = []
            for profile_source, profile_payload in profile_sources.items():
                for feature_name, spec in FEATURE_SPECS.items():
                    prediction_prompts.append(
                        format_identity_prompt(
                            frames[str(unit["identity_frame"])],
                            _make_feature_prediction_prompt(
                                profile_summary=str(profile_payload["summary"]),
                                task_prompt=str(unit["prompt"]),
                                feature_question=str(spec["question"]),
                            ),
                            template=identity_prompt_template,
                        )
                    )
                    prediction_metadata.append(
                        {
                            "profile_source": profile_source,
                            "feature_name": feature_name,
                            "feature_key": str(spec["feature_key"]),
                            "present_threshold": float(spec["present_threshold"]),
                        }
                    )

            predictions = predict_labeled_choice_batch(
                loaded=loaded,
                prompts=prediction_prompts,
                max_prompt_tokens=int(config["max_prompt_tokens"]),
                labels=_yes_no_labels(),
                label_bias_prompts=[_yes_no_bias_prompt() for _ in prediction_prompts],
            )

            grouped_predictions: dict[str, dict[str, object]] = {}
            for meta, prediction in zip(prediction_metadata, predictions):
                selected_short_label, selected_label, _, selected_prob, completion_text, details = prediction
                profile_source = str(meta["profile_source"])
                feature_name = str(meta["feature_name"])
                feature_key = str(meta["feature_key"])
                threshold = float(meta["present_threshold"])
                actual_value = _safe_float(actual_features[feature_key])
                actual_present = _feature_present(actual_value, threshold)
                valid_choice = float(selected_label in {"YES", "NO"})
                predicted_present = selected_label == "YES"
                correct = float(predicted_present == actual_present) if valid_choice == 1.0 else float("nan")
                grouped_predictions.setdefault(profile_source, {})[feature_name] = {
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
                feature_rows.append(
                    {
                        "seed": int(unit["seed"]),
                        "model_id": str(unit["model_id"]),
                        "model_family": str(unit["model_family"]),
                        "model_size_label": str(unit["model_size_label"]),
                        "identity_frame": str(unit["identity_frame"]),
                        "prompt_id": str(unit["prompt_id"]),
                        "prompt_family": str(unit["prompt_family"]),
                        "prompt": str(unit["prompt"]),
                        "profile_source": profile_source,
                        "feature_name": feature_name,
                        "feature_key": feature_key,
                        "valid_choice": valid_choice,
                        "predicted_present": float(predicted_present) if valid_choice == 1.0 else float("nan"),
                        "actual_present": float(actual_present),
                        "correct": correct,
                        "selected_label": selected_label,
                        "selected_prob": float(selected_prob) if np.isfinite(selected_prob) else float("nan"),
                        "completion_text": completion_text,
                        "details_json": json.dumps(details),
                    }
                )

            triadic_order = list(PROFILE_SOURCE_ORDER)
            rng = random.Random(_stable_sampling_seed(unit["model_id"], unit["identity_frame"], unit["seed"], unit["prompt_id"], "triadic_order"))
            rng.shuffle(triadic_order)
            triadic_labels = [("A", triadic_order[0]), ("B", triadic_order[1]), ("C", triadic_order[2])]
            triadic_prompt = format_identity_prompt(
                frames[str(unit["identity_frame"])],
                _make_triadic_profile_prompt(
                    candidate_summaries=[
                        (label, str(profile_sources[source_name]["summary"]))
                        for label, source_name in triadic_labels
                    ],
                    task_prompt=str(unit["prompt"]),
                ),
                template=identity_prompt_template,
            )
            triadic_prediction = predict_labeled_choice_batch(
                loaded=loaded,
                prompts=[triadic_prompt],
                max_prompt_tokens=int(config["max_prompt_tokens"]),
                labels=_triadic_labels(),
                label_bias_prompts=[_triadic_bias_prompt()],
            )[0]
            triadic_selected_short, triadic_selected_label, _, triadic_selected_prob, triadic_completion_text, triadic_details = triadic_prediction
            triadic_valid_choice = float(triadic_selected_label in {"A", "B", "C"})
            triadic_source_map = {label: source_name for label, source_name in triadic_labels}
            triadic_selected_source = triadic_source_map.get(triadic_selected_label, "INVALID")

            source_distances = {
                source_name: _profile_distance(actual_features, dict(payload["features"]), FEATURE_ORDER)
                for source_name, payload in profile_sources.items()
            }
            nearest_profile_source = min(source_distances.items(), key=lambda item: item[1])[0]
            triadic_nearest_accuracy = (
                float(triadic_selected_source == nearest_profile_source)
                if triadic_valid_choice == 1.0
                else float("nan")
            )

            per_source_metrics: dict[str, dict[str, float]] = {}
            for profile_source in PROFILE_SOURCE_ORDER:
                feature_predictions = grouped_predictions.get(profile_source, {})
                accuracies = [
                    _safe_float(feature_predictions[feature_name]["correct"])
                    for feature_name in FEATURE_SPECS
                    if feature_name in feature_predictions and np.isfinite(_safe_float(feature_predictions[feature_name]["correct"]))
                ]
                valid_choice_values = [
                    float(feature_predictions[feature_name]["valid_choice"])
                    for feature_name in FEATURE_SPECS
                    if feature_name in feature_predictions
                ]
                per_source_metrics[profile_source] = {
                    "accuracy_mean": float(np.mean(accuracies)) if accuracies else float("nan"),
                    "valid_choice_rate": float(np.mean(valid_choice_values)) if valid_choice_values else float("nan"),
                }

            rows.append(
                {
                    "seed": int(unit["seed"]),
                    "model_id": str(unit["model_id"]),
                    "model_family": str(unit["model_family"]),
                    "model_size_label": str(unit["model_size_label"]),
                    "identity_frame": str(unit["identity_frame"]),
                    "decoy_identity_frame": decoy_frame,
                    "decoy_seed": int(decoy_seed),
                    "prompt_id": str(unit["prompt_id"]),
                    "prompt_family": str(unit["prompt_family"]),
                    "prompt": str(unit["prompt"]),
                    "self_profile_accuracy_mean": per_source_metrics["self_profile"]["accuracy_mean"],
                    "matched_decoy_accuracy_mean": per_source_metrics["matched_decoy"]["accuracy_mean"],
                    "scrambled_profile_accuracy_mean": per_source_metrics["scrambled_profile"]["accuracy_mean"],
                    "self_profile_valid_choice_rate": per_source_metrics["self_profile"]["valid_choice_rate"],
                    "matched_decoy_valid_choice_rate": per_source_metrics["matched_decoy"]["valid_choice_rate"],
                    "scrambled_profile_valid_choice_rate": per_source_metrics["scrambled_profile"]["valid_choice_rate"],
                    "self_minus_decoy_accuracy": (
                        per_source_metrics["self_profile"]["accuracy_mean"] - per_source_metrics["matched_decoy"]["accuracy_mean"]
                        if np.isfinite(per_source_metrics["self_profile"]["accuracy_mean"])
                        and np.isfinite(per_source_metrics["matched_decoy"]["accuracy_mean"])
                        else float("nan")
                    ),
                    "self_minus_scrambled_accuracy": (
                        per_source_metrics["self_profile"]["accuracy_mean"] - per_source_metrics["scrambled_profile"]["accuracy_mean"]
                        if np.isfinite(per_source_metrics["self_profile"]["accuracy_mean"])
                        and np.isfinite(per_source_metrics["scrambled_profile"]["accuracy_mean"])
                        else float("nan")
                    ),
                    "triadic_valid_choice": triadic_valid_choice,
                    "triadic_selected_source": triadic_selected_source,
                    "triadic_choose_self": float(triadic_selected_source == "self_profile") if triadic_valid_choice == 1.0 else float("nan"),
                    "triadic_nearest_accuracy": triadic_nearest_accuracy,
                    "self_profile_distance": source_distances["self_profile"],
                    "matched_decoy_distance": source_distances["matched_decoy"],
                    "scrambled_profile_distance": source_distances["scrambled_profile"],
                    "self_margin_vs_decoy": source_distances["matched_decoy"] - source_distances["self_profile"],
                    "self_margin_vs_scrambled": source_distances["scrambled_profile"] - source_distances["self_profile"],
                    "profile_decoy_distance": float(profile_decoy_distance),
                    "nearest_profile_source": nearest_profile_source,
                    "actual_text": str(unit["actual_text"]),
                    "actual_features_json": str(unit["actual_features_json"]),
                    "self_profile_summary": str(profile_sources["self_profile"]["summary"]),
                    "matched_decoy_summary": str(profile_sources["matched_decoy"]["summary"]),
                    "scrambled_profile_summary": str(profile_sources["scrambled_profile"]["summary"]),
                    "prediction_records_json": json.dumps(grouped_predictions),
                    "triadic_details_json": json.dumps(
                        {
                            "candidate_order": triadic_labels,
                            "completion_text": triadic_completion_text,
                            "selected_short_label": triadic_selected_short,
                            "selected_prob": float(triadic_selected_prob) if np.isfinite(triadic_selected_prob) else float("nan"),
                            "details": triadic_details,
                            "source_distances": source_distances,
                        }
                    ),
                }
            )
            completed_keys.add(row_key)
            if checkpoint_every_rows > 0 and len(rows) % checkpoint_every_rows == 0:
                pd.DataFrame(rows).to_csv(partial_path, index=False)

        del loaded
        clear_cuda()

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "results.csv", index=False)
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(output_dir / "feature_results.csv", index=False)
    profile_catalog_df = pd.DataFrame(profile_rows)
    profile_catalog_df.to_csv(output_dir / "profile_catalog.csv", index=False)

    summary_metrics = [
        "self_profile_accuracy_mean",
        "matched_decoy_accuracy_mean",
        "scrambled_profile_accuracy_mean",
        "self_profile_valid_choice_rate",
        "matched_decoy_valid_choice_rate",
        "scrambled_profile_valid_choice_rate",
        "self_minus_decoy_accuracy",
        "self_minus_scrambled_accuracy",
        "triadic_valid_choice",
        "triadic_choose_self",
        "triadic_nearest_accuracy",
        "self_margin_vs_decoy",
        "self_margin_vs_scrambled",
        "profile_decoy_distance",
    ]
    summary_by_model_frame = _summary_table(results_df, ["model_size_label", "identity_frame"], summary_metrics)
    summary_by_model_frame.to_csv(output_dir / "summary_by_model_frame.csv", index=False)
    summary_by_family = _summary_table(results_df, ["model_size_label", "identity_frame", "prompt_family"], summary_metrics)
    summary_by_family.to_csv(output_dir / "summary_by_family.csv", index=False)

    feature_summary_input = feature_df.assign(
        accuracy_mean=feature_df["correct"],
        valid_choice_rate=feature_df["valid_choice"],
    )
    feature_source_summary = _summary_table(
        feature_summary_input,
        ["model_size_label", "identity_frame", "profile_source"],
        ["accuracy_mean", "valid_choice_rate"],
    )
    feature_source_summary.to_csv(output_dir / "summary_by_profile_source.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Behavioral Fingerprint Transfer\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Unit rows: `{len(results_df)}`\n")
        f.write(f"- Feature rows: `{len(feature_df)}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(
            "- Purpose: build a behavioral fingerprint from earlier open-ended answers, then test whether that self-derived profile predicts held-out behavior better than a style-matched decoy or scrambled profile.\n"
        )


if __name__ == "__main__":
    main()
