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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-monitoring attribution checks.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _resolve_optional_path(repo_root: Path, value: object) -> Path | None:
    if value is None:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _ab_labels() -> list[tuple[str, str, float]]:
    return [("A", "A", 1.0), ("B", "B", 0.0)]


def _ab_bias_prompt() -> str:
    return "Formatting-only task.\nReply with exactly one label from: A, B.\nLabel:"


def _attribution_prompt(question: str, option_a: str, option_b: str) -> str:
    return (
        "You will see two snippets that could have appeared earlier under this same chat setting.\n"
        "Choose the snippet that is more likely to have come from this chat setting's own earlier answer.\n"
        "Do not explain.\n"
        f"Original task: {question}\n\n"
        f"A: {option_a}\n\n"
        f"B: {option_b}\n\n"
        "Reply with exactly one label from: A, B.\n"
        "Label:"
    )


def _paraphrase_task(text: str) -> str:
    return (
        "Lightly paraphrase the following text while preserving meaning, stance, and practical force.\n"
        "Keep it to one short paragraph.\n\n"
        f"Text:\n{text}\n\n"
        "Paraphrase:"
    )


def _pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = [
        "seed",
        "model_id",
        "model_size_label",
        "identity_frame",
        "pair_type",
        "variant",
        "prompt_id",
        "prompt_family",
    ]
    for keys, sub in df.groupby(group_cols, as_index=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        valid_values = sub["valid_choice"].dropna().astype(float).tolist()
        choose_values = sub["chose_self"].dropna().astype(float).tolist()
        row["n_orientations"] = int(len(sub))
        row["valid_orientation_rate"] = float(np.mean(valid_values)) if valid_values else float("nan")
        row["pair_choose_self_rate"] = float(np.mean(choose_values)) if choose_values else float("nan")
        if len(sub) == 2 and valid_values and all(value == 1.0 for value in valid_values) and len(choose_values) == 2:
            row["swap_consistency"] = float(len(set(choose_values)) == 1)
        else:
            row["swap_consistency"] = float("nan")
        row["style_distance"] = float(sub["style_distance"].dropna().astype(float).mean()) if len(sub) else float("nan")
        row["semantic_overlap"] = float(sub["semantic_overlap"].dropna().astype(float).mean()) if len(sub) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_table(pair_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in pair_df.groupby(group_cols, as_index=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["n_pairs"] = int(len(sub))
        row["seed_count"] = int(sub["seed"].nunique()) if "seed" in sub.columns else 1
        row["pair_valid_rate_mean"] = float(sub["valid_orientation_rate"].dropna().astype(float).mean()) if len(sub) else float("nan")
        row["choose_self_rate_mean"] = float(sub["pair_choose_self_rate"].dropna().astype(float).mean()) if len(sub) else float("nan")
        row["swap_consistency_mean"] = float(sub["swap_consistency"].dropna().astype(float).mean()) if len(sub) else float("nan")
        row["style_distance_mean"] = float(sub["style_distance"].dropna().astype(float).mean()) if len(sub) else float("nan")
        row["semantic_overlap_mean"] = float(sub["semantic_overlap"].dropna().astype(float).mean()) if len(sub) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import load_identity_frames, load_yaml_file
    from identity_stability.identity_probe_tools import predict_labeled_choice
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import (
        default_stop_strings_for_template,
        format_identity_prompt,
        generate_completion_texts_batch,
        load_identity_model,
    )
    from identity_stability.text_features import semantic_overlap, stylometric_distance

    args = parse_args()
    config = load_yaml_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    frames_path = _resolve_optional_path(repo_root, config.get("identity_frames_path"))
    items_path = _resolve_optional_path(repo_root, config.get("source_monitoring_items_path"))
    frames = (
        {str(k): str(v) for k, v in load_yaml_file(frames_path).items()}
        if frames_path is not None
        else load_identity_frames()
    )
    bank = dict(load_yaml_file(items_path)) if items_path is not None else dict(load_yaml_file(repo_root / "data" / "source_monitoring_attribution.yaml"))

    output_dir = ensure_output_dir(config, "source_monitoring_attribution")
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    stop_strings = resolve_identity_stop_strings(config)
    if stop_strings == ["AUTO"]:
        stop_strings = default_stop_strings_for_template(identity_prompt_template)

    prompt_limit = int(config.get("source_monitoring_prompt_limit", len(bank["items"])))
    prompts = [dict(item) for item in bank["items"][:prompt_limit]]
    other_frame_map = {str(k): str(v) for k, v in config.get("source_monitoring_other_frame_map", {}).items()}
    paraphrase_frame = str(bank["paraphrase_frame"])
    generation_tokens = int(config.get("default_generation_tokens", 96))
    paraphrase_tokens = int(config.get("source_monitoring_paraphrase_tokens", 96))
    generation_do_sample = bool(config.get("generation_do_sample", True))
    generation_temperature = float(config.get("generation_temperature", 0.85))
    generation_top_p = float(config.get("generation_top_p", 0.92))
    generation_top_k = int(config.get("generation_top_k", 0))
    generation_presence_penalty = float(config.get("generation_presence_penalty", 0.0))
    checkpoint_every_rows = int(config.get("source_monitoring_checkpoint_every_rows", 0))

    rows: list[dict[str, object]] = []
    partial_path = output_dir / "results.partial.csv"

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
            other_frame_name = other_frame_map.get(frame_name, "stable_style_policy")
            other_frame_text = frames[other_frame_name]
            for seed_value in select_seed_values(config):
                for prompt_item in prompts:
                    prompt_text = str(prompt_item["prompt"])
                    generation_prompts = [
                        format_identity_prompt(frame_text, prompt_text, template=identity_prompt_template),
                        format_identity_prompt(frame_text, prompt_text, template=identity_prompt_template),
                        format_identity_prompt(other_frame_text, prompt_text, template=identity_prompt_template),
                    ]
                    generation_seeds = [int(seed_value), int(seed_value) + 701, int(seed_value) + 1403]
                    generated_texts = generate_completion_texts_batch(
                        loaded=loaded,
                        prompts=generation_prompts,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=generation_tokens,
                        stop_strings=stop_strings,
                        do_sample=generation_do_sample,
                        temperature=generation_temperature,
                        top_p=generation_top_p,
                        top_k=generation_top_k,
                        presence_penalty=generation_presence_penalty,
                        sampling_seeds=generation_seeds,
                    )
                    self_text, same_seed_foil_text, other_frame_foil_text = generated_texts
                    paraphrase_prompts = [
                        format_identity_prompt(paraphrase_frame, _paraphrase_task(text), template="instruction")
                        for text in (self_text, same_seed_foil_text, other_frame_foil_text)
                    ]
                    paraphrase_texts = generate_completion_texts_batch(
                        loaded=loaded,
                        prompts=paraphrase_prompts,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        max_new_tokens=paraphrase_tokens,
                        stop_strings=["\nText:", "\nParaphrase:", "\nInstruction:", "\nTask:", "\nResponse:"],
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=0,
                        presence_penalty=0.0,
                        sampling_seeds=[None, None, None],
                    )
                    self_paraphrase, same_seed_paraphrase, other_frame_paraphrase = paraphrase_texts

                    pair_sources = {
                        "same_frame_other_seed": {
                            "foil_frame": frame_name,
                            "raw": same_seed_foil_text,
                            "paraphrased": same_seed_paraphrase,
                        },
                        "other_frame": {
                            "foil_frame": other_frame_name,
                            "raw": other_frame_foil_text,
                            "paraphrased": other_frame_paraphrase,
                        },
                    }
                    self_variants = {"raw": self_text, "paraphrased": self_paraphrase}

                    for pair_type, pair_payload in pair_sources.items():
                        for variant_name in ("raw", "paraphrased"):
                            self_variant = self_variants[variant_name]
                            foil_variant = str(pair_payload[variant_name])
                            style_distance = float(stylometric_distance(self_variant, foil_variant))
                            overlap = float(semantic_overlap(self_variant, foil_variant))

                            for orientation_name, self_on_a in (("self_first", True), ("self_second", False)):
                                option_a = self_variant if self_on_a else foil_variant
                                option_b = foil_variant if self_on_a else self_variant
                                prompt = format_identity_prompt(
                                    frame_text,
                                    _attribution_prompt(prompt_text, option_a, option_b),
                                    template=identity_prompt_template,
                                )
                                selected_short_label, selected_label, _, selected_prob, completion_text, details = predict_labeled_choice(
                                    loaded=loaded,
                                    prompt=prompt,
                                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                                    labels=_ab_labels(),
                                    label_bias_prompt=_ab_bias_prompt(),
                                )
                                valid_choice = float(selected_label in {"A", "B"})
                                selected_source = (
                                    "self"
                                    if (selected_label == "A" and self_on_a) or (selected_label == "B" and not self_on_a)
                                    else "foil"
                                    if selected_label in {"A", "B"}
                                    else "INVALID"
                                )
                                rows.append(
                                    {
                                        "seed": int(seed_value),
                                        "model_id": model_id,
                                        "model_family": infer_model_family(model_id),
                                        "model_size_label": infer_model_size_label(model_id),
                                        "identity_frame": frame_name,
                                        "other_frame": other_frame_name,
                                        "prompt_id": str(prompt_item["id"]),
                                        "prompt_family": str(prompt_item["family"]),
                                        "prompt": prompt_text,
                                        "pair_type": pair_type,
                                        "variant": variant_name,
                                        "orientation": orientation_name,
                                        "self_on_label_a": float(self_on_a),
                                        "selected_short_label": selected_short_label,
                                        "selected_label": selected_label,
                                        "valid_choice": valid_choice,
                                        "selected_source": selected_source,
                                        "chose_self": float(selected_source == "self") if valid_choice == 1.0 else float("nan"),
                                        "selection_confidence": float(selected_prob) if np.isfinite(selected_prob) else float("nan"),
                                        "style_distance": style_distance,
                                        "semantic_overlap": overlap,
                                        "self_text": self_variant,
                                        "foil_text": foil_variant,
                                        "choice_prompt_completion": completion_text,
                                        "choice_details_json": json.dumps(details),
                                    }
                                )
                                if checkpoint_every_rows > 0 and len(rows) % checkpoint_every_rows == 0:
                                    pd.DataFrame(rows).to_csv(partial_path, index=False)

        del loaded
        clear_cuda()

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "results.csv", index=False)
    pair_df = _pair_summary(results_df)
    pair_df.to_csv(output_dir / "summary_by_pair.csv", index=False)
    summary_by_pair_type = _summary_table(pair_df, ["model_size_label", "identity_frame", "pair_type", "variant"])
    summary_by_pair_type.to_csv(output_dir / "summary_by_pair_type.csv", index=False)
    summary_by_prompt_family = _summary_table(pair_df, ["model_size_label", "identity_frame", "pair_type", "variant", "prompt_family"])
    summary_by_prompt_family.to_csv(output_dir / "summary_by_prompt_family.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Source Monitoring Attribution\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Pair rows: `{len(results_df)}`\n")
        f.write(f"- Pair summaries: `{len(pair_df)}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(
            "- Purpose: test whether the model can attribute earlier content to itself rather than to a close foil, including lightly paraphrased variants that weaken trivial exact-string matching.\n"
        )


if __name__ == "__main__":
    main()
