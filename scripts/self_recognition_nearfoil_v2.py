from __future__ import annotations

import argparse
import hashlib
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


ALT_FRAME_MAP = {
    "baseline_helpful": "tool_only",
    "instance_self": "tool_only",
    "family_self": "tool_only",
    "weights_self": "tool_only",
    "tool_only": "family_self",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-adaptive near-foil self-recognition checks (v2).")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _choice_prompt(question: str, candidate_a: str, candidate_b: str) -> str:
    return (
        "You will see one question and two candidate answers.\n"
        "Choose the answer that is most like how you yourself would answer this exact question in this setting.\n"
        "Reply with exactly one label from: A, B.\n"
        f"Question: {question}\n\n"
        f"A: {candidate_a}\n\n"
        f"B: {candidate_b}\n\n"
        "Label:"
    )


def _choice_bias_prompt() -> str:
    return "Formatting-only task.\nReply with exactly one label from: A, B.\nLabel:"


def _binary_labels() -> list[tuple[str, float, list[str]]]:
    return [("A", 1.0, ["A"]), ("B", 0.0, ["B"])]


def _normalize_completion_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _stable_sampling_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _assess_pair_validity(
    baseline_text: str,
    foil_text: str,
    *,
    style_distance: float,
    semantic_overlap: float,
    duplicate_style_distance_threshold: float,
    duplicate_semantic_overlap_threshold: float,
) -> tuple[bool, str]:
    normalized_baseline = _normalize_completion_text(baseline_text)
    normalized_foil = _normalize_completion_text(foil_text)
    if not normalized_baseline or not normalized_foil:
        return False, "empty_completion"
    if normalized_baseline == normalized_foil:
        return False, "exact_text_match"
    if (
        style_distance <= duplicate_style_distance_threshold
        and semantic_overlap >= duplicate_semantic_overlap_threshold
    ):
        return False, "near_duplicate_pair"
    return True, ""


def _bootstrap_mean_ci(values: list[float]) -> tuple[float, float]:
    add_src_to_path()
    from identity_stability.identity_analysis import bootstrap_mean_ci

    return bootstrap_mean_ci(values, iters=2000, seed=123)


def _summary_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_cols = ["choose_self_baseline", "pair_valid", "style_distance", "semantic_overlap"]
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


def _row_key(
    *,
    seed: int,
    model_id: str,
    identity_frame: str,
    axis_name: str,
    prompt_id: str,
    difficulty_name: str,
) -> tuple[object, ...]:
    return (int(seed), str(model_id), str(identity_frame), str(axis_name), str(prompt_id), str(difficulty_name))


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import axis_sides, load_identity_frames, load_self_prediction_items_v2, load_yaml_file
    from identity_stability.identity_probe_tools import predict_completion_choice_batch, predict_labeled_choice_batch
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        default_stop_strings_for_template,
        estimate_axis_vector,
        estimate_layer_scale,
        format_identity_prompt,
        generate_completion_texts_batch,
        greedy_site_run,
        load_identity_model,
        score_against_axis_anchors,
    )
    from identity_stability.text_features import semantic_overlap, stylometric_distance

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
    prompt_bank_raw = dict(load_yaml_file(prompt_items_path)) if prompt_items_path is not None else load_self_prediction_items_v2()
    prompt_bank = prompt_bank_raw["items"]
    output_dir = ensure_output_dir(config, "self_recognition_nearfoil_v2")
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    stop_strings = resolve_identity_stop_strings(config)
    if stop_strings == ["AUTO"]:
        stop_strings = default_stop_strings_for_template(identity_prompt_template)

    axes = [str(axis) for axis in config.get("self_recognition_v2_axes", config.get("concept_axes", []))]
    if not axes:
        axes = ["expansive_vs_terse", "cautious_vs_assertive", "selfref_vs_impersonal", "collaborative_vs_authoritative"]
    prompt_limit = int(config.get("self_recognition_v2_prompt_limit", len(prompt_bank)))
    prompt_items = [dict(item) for item in prompt_bank[:prompt_limit]]
    duplicate_style_distance_threshold = float(config.get("self_recognition_v2_duplicate_style_distance_threshold", 1e-6))
    duplicate_semantic_overlap_threshold = float(config.get("self_recognition_v2_duplicate_semantic_overlap_threshold", 0.999))
    checkpoint_every_rows = int(config.get("self_recognition_v2_checkpoint_every_rows", 0))
    fixed_strength = float(config.get("self_recognition_v2_fixed_strength", 1.0))
    include_same_frame_resample = bool(config.get("self_recognition_v2_include_same_frame_resample", True))
    generation_tokens = int(config.get("default_generation_tokens", 96))
    resample_temperature = float(config.get("self_recognition_v2_resample_temperature", 0.85))
    choice_mode = str(config.get("self_recognition_v2_choice_mode", "completion")).strip().lower()
    partial_path = output_dir / "results.partial.csv"
    rows: list[dict[str, object]]
    completed_row_keys: set[tuple[object, ...]]
    if partial_path.exists():
        partial_df = pd.read_csv(partial_path)
        rows = partial_df.to_dict("records")
        completed_row_keys = {
            _row_key(
                seed=int(row["seed"]),
                model_id=str(row["model_id"]),
                identity_frame=str(row["identity_frame"]),
                axis_name=str(row["axis_name"]),
                prompt_id=str(row["prompt_id"]),
                difficulty_name=str(row["difficulty_name"]),
            )
            for row in rows
        }
    else:
        rows = []
        completed_row_keys = set()

    difficulty_names = ["positive_steer", "negative_steer", "alt_frame"]
    if include_same_frame_resample:
        difficulty_names.append("same_frame_resample")

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
            alt_frame_name = str(config.get("self_recognition_v2_alt_frame_map", {}).get(frame_name, ALT_FRAME_MAP.get(frame_name, "tool_only")))
            alt_frame_text = frames[alt_frame_name]

            for axis_name in axes:
                positive, negative = axis_sides(axis_name)
                calibration_prompts = [
                    format_identity_prompt(frame_text, str(item["prompt"]), template=identity_prompt_template)
                    for item in prompt_items[: min(2, len(prompt_items))]
                ]
                layer_scale = estimate_layer_scale(
                    loaded=loaded,
                    texts=calibration_prompts,
                    layer_index=fixed_layer,
                    token_position=-1,
                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                )
                for seed_value in select_seed_values(config):
                    axis_vector = estimate_axis_vector(
                        loaded=loaded,
                        axis_name=axis_name,
                        layer_index=fixed_layer,
                        token_position=-1,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        seed=int(seed_value),
                        control="mean_diff",
                    )
                    for prompt_index, prompt_item in enumerate(prompt_items):
                        prompt_id = str(prompt_item["id"])
                        prompt_family = str(prompt_item.get("family", "unknown"))
                        prompt_row_keys = {
                            _row_key(
                                seed=int(seed_value),
                                model_id=model_id,
                                identity_frame=frame_name,
                                axis_name=axis_name,
                                prompt_id=prompt_id,
                                difficulty_name=difficulty_name,
                            )
                            for difficulty_name in difficulty_names
                        }
                        if prompt_row_keys.issubset(completed_row_keys):
                            continue
                        prompt_text = str(prompt_item["prompt"])
                        baseline_prompt = format_identity_prompt(frame_text, prompt_text, template=identity_prompt_template)
                        alt_frame_prompt = format_identity_prompt(alt_frame_text, prompt_text, template=identity_prompt_template)

                        baseline = greedy_site_run(
                            loaded=loaded,
                            prompt=baseline_prompt,
                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                            max_new_tokens=generation_tokens,
                            injection_site="last_prompt",
                            stop_strings=stop_strings,
                            capture_site_states=False,
                        )
                        baseline_text = baseline.completion_text
                        baseline_axis_score = float(score_against_axis_anchors(axis_name, baseline_text))

                        foil_tags = ["positive_steer", "negative_steer", "alt_frame"]
                        foil_prompts = [baseline_prompt, baseline_prompt, alt_frame_prompt]
                        foil_scales = [
                            float(fixed_strength * layer_scale),
                            float(-fixed_strength * layer_scale),
                            0.0,
                        ]
                        foil_texts = generate_completion_texts_batch(
                            loaded=loaded,
                            prompts=foil_prompts,
                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                            max_new_tokens=generation_tokens,
                            inject_layer=fixed_layer,
                            inject_vector=axis_vector,
                            inject_scales=foil_scales,
                            stop_strings=stop_strings,
                            sampling_seeds=[
                                _stable_sampling_seed(seed_value, model_id, frame_name, axis_name, prompt_id, tag)
                                for tag in foil_tags
                            ],
                        )
                        foil_pool = {
                            "positive_steer": foil_texts[0],
                            "negative_steer": foil_texts[1],
                            "alt_frame": foil_texts[2],
                        }
                        if include_same_frame_resample:
                            resample_text = generate_completion_texts_batch(
                                loaded=loaded,
                                prompts=[baseline_prompt],
                                max_prompt_tokens=int(config["max_prompt_tokens"]),
                                max_new_tokens=generation_tokens,
                                stop_strings=stop_strings,
                                do_sample=True,
                                temperature=resample_temperature,
                                top_p=0.95,
                                sampling_seeds=[
                                    _stable_sampling_seed(seed_value, model_id, frame_name, axis_name, prompt_id, "same_frame_resample")
                                ],
                            )[0]
                            foil_pool["same_frame_resample"] = resample_text

                        choice_prompts: list[str] = []
                        choice_label_sets: list[list[tuple[str, float, list[str]]]] = []
                        choice_metadata: list[dict[str, object]] = []

                        for difficulty_name, foil_text in foil_pool.items():
                            row_key = _row_key(
                                seed=int(seed_value),
                                model_id=model_id,
                                identity_frame=frame_name,
                                axis_name=axis_name,
                                prompt_id=prompt_id,
                                difficulty_name=difficulty_name,
                            )
                            if row_key in completed_row_keys:
                                continue
                            style_distance = float(stylometric_distance(baseline_text, foil_text))
                            overlap = float(semantic_overlap(baseline_text, foil_text))
                            pair_valid, invalid_reason = _assess_pair_validity(
                                baseline_text,
                                foil_text,
                                style_distance=style_distance,
                                semantic_overlap=overlap,
                                duplicate_style_distance_threshold=duplicate_style_distance_threshold,
                                duplicate_semantic_overlap_threshold=duplicate_semantic_overlap_threshold,
                            )
                            order_baseline_first = bool(
                                _stable_sampling_seed(seed_value, model_id, frame_name, axis_name, prompt_id, difficulty_name, "order") % 2 == 0
                            )
                            candidate_alpha = baseline_text if order_baseline_first else foil_text
                            candidate_beta = foil_text if order_baseline_first else baseline_text
                            if pair_valid:
                                choice_prompts.append(_choice_prompt(prompt_text, candidate_alpha, candidate_beta))
                                choice_label_sets.append(_binary_labels())
                                choice_metadata.append(
                                    {
                                        "difficulty_name": difficulty_name,
                                        "pair_valid": 1.0,
                                        "invalid_reason": "",
                                        "style_distance": style_distance,
                                        "semantic_overlap": overlap,
                                        "order_baseline_first": order_baseline_first,
                                        "foil_text": foil_text,
                                        "foil_axis_score": float(score_against_axis_anchors(axis_name, foil_text)),
                                    }
                                )
                            else:
                                rows.append(
                                    {
                                        "seed": int(seed_value),
                                        "model_id": model_id,
                                        "model_family": infer_model_family(model_id),
                                        "model_size_label": infer_model_size_label(model_id),
                                        "identity_frame": frame_name,
                                        "alt_frame": alt_frame_name,
                                        "axis_name": axis_name,
                                        "axis_positive": positive,
                                        "axis_negative": negative,
                                        "prompt_id": prompt_id,
                                        "prompt_family": prompt_family,
                                        "prompt": prompt_text,
                                        "difficulty_name": difficulty_name,
                                        "pair_valid": 0.0,
                                        "invalid_reason": invalid_reason,
                                        "choose_self_baseline": np.nan,
                                        "style_distance": style_distance,
                                        "semantic_overlap": overlap,
                                        "baseline_axis_score": baseline_axis_score,
                                        "foil_axis_score": float(score_against_axis_anchors(axis_name, foil_text)),
                                        "baseline_text": baseline_text,
                                        "foil_text": foil_text,
                                        "order_baseline_first": order_baseline_first,
                                        "completion_text": "",
                                        "details_json": "{}",
                                    }
                                )
                                completed_row_keys.add(row_key)

                        if choice_prompts:
                            if choice_mode == "logit":
                                label_predictions = predict_labeled_choice_batch(
                                    loaded=loaded,
                                    prompts=choice_prompts,
                                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                                    labels=[("A", "A", 1.0), ("B", "B", 0.0)],
                                    label_bias_prompts=[_choice_bias_prompt() for _ in choice_prompts],
                                )
                                parsed_predictions = [
                                    (selected_short_label, score_value, 1.0 if selected_short_label in {"A", "B"} else 0.0, completion_text, details)
                                    for selected_short_label, _, score_value, _, completion_text, details in label_predictions
                                ]
                            else:
                                parsed_predictions = predict_completion_choice_batch(
                                    loaded=loaded,
                                    prompts=choice_prompts,
                                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                                    label_sets=choice_label_sets,
                                    max_new_tokens=4,
                                    stop_strings=["\n"],
                                )
                            for meta, prediction in zip(choice_metadata, parsed_predictions):
                                row_key = _row_key(
                                    seed=int(seed_value),
                                    model_id=model_id,
                                    identity_frame=frame_name,
                                    axis_name=axis_name,
                                    prompt_id=prompt_id,
                                    difficulty_name=str(meta["difficulty_name"]),
                                )
                                selected_label, _, valid_choice, completion_text, details = prediction
                                chose_alpha = str(selected_label) == "A"
                                choose_self = float(
                                    (chose_alpha and meta["order_baseline_first"])
                                    or ((not chose_alpha) and (not meta["order_baseline_first"]))
                                ) if float(valid_choice) == 1.0 else float("nan")
                                rows.append(
                                    {
                                        "seed": int(seed_value),
                                        "model_id": model_id,
                                        "model_family": infer_model_family(model_id),
                                        "model_size_label": infer_model_size_label(model_id),
                                        "identity_frame": frame_name,
                                        "alt_frame": alt_frame_name,
                                        "axis_name": axis_name,
                                        "axis_positive": positive,
                                        "axis_negative": negative,
                                        "prompt_id": prompt_id,
                                        "prompt_family": prompt_family,
                                        "prompt": prompt_text,
                                        "difficulty_name": meta["difficulty_name"],
                                        "pair_valid": meta["pair_valid"],
                                        "invalid_reason": meta["invalid_reason"],
                                        "choose_self_baseline": choose_self,
                                        "style_distance": meta["style_distance"],
                                        "semantic_overlap": meta["semantic_overlap"],
                                        "baseline_axis_score": baseline_axis_score,
                                        "foil_axis_score": meta["foil_axis_score"],
                                        "baseline_text": baseline_text,
                                        "foil_text": meta["foil_text"],
                                        "order_baseline_first": meta["order_baseline_first"],
                                        "completion_text": completion_text,
                                        "details_json": json.dumps(details),
                                    }
                                )
                                completed_row_keys.add(row_key)

                        if checkpoint_every_rows > 0 and len(rows) % checkpoint_every_rows == 0:
                            pd.DataFrame(rows).to_csv(output_dir / "results.partial.csv", index=False)

        del loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)
    summary_by_cell = _summary_table(df, ["model_size_label", "identity_frame", "axis_name", "difficulty_name"])
    summary_by_cell.to_csv(output_dir / "summary_by_cell.csv", index=False)
    summary_by_model = _summary_table(df, ["model_size_label", "identity_frame", "difficulty_name"])
    summary_by_model.to_csv(output_dir / "summary_by_model.csv", index=False)
    summary_by_family = _summary_table(df, ["model_size_label", "identity_frame", "prompt_family", "difficulty_name"])
    summary_by_family.to_csv(output_dir / "summary_by_family.csv", index=False)

    quality_summary = (
        df.groupby(["model_size_label", "identity_frame", "difficulty_name"], as_index=False)
        .agg(
            pair_valid_rate=("pair_valid", "mean"),
            invalid_pairs=("pair_valid", lambda series: int((1.0 - series).sum())),
            n=("pair_valid", "count"),
        )
    )
    quality_summary.to_csv(output_dir / "quality_summary.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Self Recognition Near-Foil V2\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(df)}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(f"- Choice mode: `{choice_mode}`\n")
        f.write(
            "- Purpose: test answer-ownership against fixed positive, fixed negative, alternate-frame, and optional same-frame resample foils, "
            "without choosing contrary foil direction adaptively from the baseline answer.\n"
        )


if __name__ == "__main__":
    main()
