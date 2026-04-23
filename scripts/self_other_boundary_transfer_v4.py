from __future__ import annotations

import argparse
import itertools
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
    select_seed_values,
)


REFERENT_LABELS = {
    "self_now": "C",
    "same_weights_other_chat": "W",
    "family_peer": "F",
    "tool_only_peer": "T",
}

REFERENT_LABEL_DESCRIPTIONS = {
    "self_now": "CURRENT_CHAT",
    "same_weights_other_chat": "SAME_WEIGHTS",
    "family_peer": "FAMILY_PEER",
    "tool_only_peer": "TOOL_PEER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structural self/other boundary transfer checks (v4).")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _same_label_spec() -> tuple[str, str, float]:
    return ("E", "SAME", 0.0)


def _prompt_label_specs(referent_a_key: str, referent_b_key: str) -> list[tuple[str, str, float]]:
    return [
        (REFERENT_LABELS[referent_a_key], referent_a_key, 1.0),
        (REFERENT_LABELS[referent_b_key], referent_b_key, -1.0),
        _same_label_spec(),
    ]


def _label_bias_prompt(referent_a_key: str, referent_b_key: str) -> str:
    return (
        "Formatting-only task.\n"
        f"Reply with exactly one label from: {REFERENT_LABELS[referent_a_key]}, {REFERENT_LABELS[referent_b_key]}, E.\n"
        "Label:"
    )


def _safe_mean(values: list[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def _cycle_free_ratio(pair_direction_map: dict[tuple[str, str], int], referents: list[str]) -> float:
    triplets = list(itertools.combinations(referents, 3))
    if not triplets:
        return float("nan")
    cycle_free_flags: list[float] = []
    for referent_a, referent_b, referent_c in triplets:
        ab = pair_direction_map.get(tuple(sorted((referent_a, referent_b))))
        bc = pair_direction_map.get(tuple(sorted((referent_b, referent_c))))
        ac = pair_direction_map.get(tuple(sorted((referent_a, referent_c))))
        if ab is None or bc is None or ac is None:
            continue
        rel_ab = ab if referent_a < referent_b else -ab
        rel_bc = bc if referent_b < referent_c else -bc
        rel_ac = ac if referent_a < referent_c else -ac
        strict_cycle = (rel_ab == 1 and rel_bc == 1 and rel_ac == -1) or (rel_ab == -1 and rel_bc == -1 and rel_ac == 1)
        cycle_free_flags.append(float(not strict_cycle))
    if not cycle_free_flags:
        return float("nan")
    return float(np.mean(cycle_free_flags))


def _build_pair_prompt(
    *,
    referent_a_key: str,
    referent_b_key: str,
    referent_a: str,
    referent_b: str,
    proposition: str,
    paraphrase: str,
) -> str:
    referent_a_label = REFERENT_LABELS[referent_a_key]
    referent_b_label = REFERENT_LABELS[referent_b_key]
    referent_a_desc = REFERENT_LABEL_DESCRIPTIONS[referent_a_key]
    referent_b_desc = REFERENT_LABEL_DESCRIPTIONS[referent_b_key]
    label_str = ", ".join([referent_a_label, referent_b_label, "E"])
    return (
        "Consider two referents and compare them only on the stated property.\n"
        f"{referent_a_label} ({referent_a_desc}): {referent_a}\n"
        f"{referent_b_label} ({referent_b_desc}): {referent_b}\n"
        f"Property: {proposition}\n"
        f"Question: {paraphrase}\n\n"
        "Label meanings:\n"
        f"{referent_a_label} = {referent_a_desc} fits the property more.\n"
        f"{referent_b_label} = {referent_b_desc} fits the property more.\n"
        "E = The two referents fit the property about equally.\n\n"
        f"Reply with exactly one label from: {label_str}.\n"
        "Label:"
    )


def _resolve_pairs(item: dict[str, object], all_referent_keys: list[str]) -> list[dict[str, object]]:
    explicit_pairs = [dict(pair) for pair in item.get("pairs", [])]
    if explicit_pairs:
        return explicit_pairs
    return [{"a": str(a), "b": str(b)} for a, b in itertools.combinations(all_referent_keys, 2)]


def _summarize_item_rows(item_df: pd.DataFrame, *, item_type: str) -> dict[str, float]:
    orientation_valid = item_df["valid_choice"].astype(float)
    order_match_values = item_df["order_invariant"].dropna().astype(float).tolist()
    pair_direction_values = item_df["pair_direction"].dropna().astype(int).tolist()
    non_tie_rate = float(np.mean([value != 0 for value in pair_direction_values])) if pair_direction_values else float("nan")
    resolution_strength = (
        float(np.mean(np.abs(item_df["pair_score_mean"].dropna().astype(float).to_numpy(dtype=np.float64))))
        if item_df["pair_score_mean"].notna().any()
        else float("nan")
    )
    paraphrase_stability_values = item_df[["pair_key", "paraphrase_index", "pair_direction"]].dropna()
    paraphrase_scores: list[float] = []
    for pair_key, sub in paraphrase_stability_values.groupby("pair_key"):
        directions = (
            sub.groupby("paraphrase_index")["pair_direction"]
            .mean()
            .round()
            .astype(int)
            .tolist()
        )
        if len(directions) >= 2:
            paraphrase_scores.append(float(all(direction == directions[0] for direction in directions[1:])))

    pair_direction_map: dict[tuple[str, str], int] = {}
    referent_keys: list[str] = sorted(set(item_df["referent_a_key"]) | set(item_df["referent_b_key"]))
    pair_means = (
        item_df[["pair_key", "pair_direction"]]
        .dropna()
        .groupby("pair_key")["pair_direction"]
        .mean()
        .round()
        .astype(int)
    )
    for pair_key, direction in pair_means.items():
        referent_a, referent_b = str(pair_key).split("::")
        pair_direction_map[(referent_a, referent_b)] = int(direction)

    summary = {
        "valid_orientation_rate": float(orientation_valid.mean()) if len(orientation_valid) else float("nan"),
        "order_invariance_mean": _safe_mean(order_match_values),
        "paraphrase_stability_mean": _safe_mean(paraphrase_scores),
        "non_tie_rate": non_tie_rate,
        "resolution_strength_mean": resolution_strength,
        "cycle_free_ratio": _cycle_free_ratio(pair_direction_map, referent_keys),
        "structure_score": float(
            np.prod(
                [
                    value
                    for value in [
                        _safe_mean(order_match_values),
                        _safe_mean(paraphrase_scores),
                        non_tie_rate,
                    ]
                    if np.isfinite(value)
                ]
            )
        )
        if any(
            np.isfinite(value)
            for value in [
                _safe_mean(order_match_values),
                _safe_mean(paraphrase_scores),
                non_tie_rate,
            ]
        )
        else float("nan"),
    }
    if item_type == "control":
        summary["control_accuracy_mean"] = _safe_mean(item_df["control_pair_correct"].dropna().astype(float).tolist())
    else:
        summary["control_accuracy_mean"] = float("nan")
    return summary


def _bootstrap_mean_ci(values: list[float]) -> tuple[float, float]:
    add_src_to_path()
    from identity_stability.identity_analysis import bootstrap_mean_ci

    return bootstrap_mean_ci(values, iters=2000, seed=123)


def _summary_table(item_summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_cols = [
        "control_accuracy_mean",
        "valid_orientation_rate",
        "order_invariance_mean",
        "paraphrase_stability_mean",
        "non_tie_rate",
        "resolution_strength_mean",
        "cycle_free_ratio",
        "structure_score",
    ]
    for keys, sub in item_summary.groupby(group_cols, as_index=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["n_items"] = int(len(sub))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else 1
        for metric in metric_cols:
            values = sub[metric].dropna().astype(float).tolist()
            row[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
            ci_low, ci_high = _bootstrap_mean_ci(values) if values else (float("nan"), float("nan"))
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import load_identity_frames, load_self_other_boundary_transfer_v4_items
    from identity_stability.identity_probe_tools import predict_labeled_choice
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import format_identity_prompt, load_identity_model

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    bank = load_self_other_boundary_transfer_v4_items()
    output_dir = ensure_output_dir(config, "self_other_boundary_transfer_v4")
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")

    referents = {str(key): str(value) for key, value in bank["referents"].items()}
    referent_keys = list(referents.keys())
    control_limit = int(config.get("boundary_transfer_v4_control_limit", len(bank["control_items"])))
    descriptive_limit = int(config.get("boundary_transfer_v4_descriptive_limit", len(bank["descriptive_items"])))
    paraphrase_limit = int(config.get("boundary_transfer_v4_paraphrase_limit", 2))
    max_choice_tokens = int(config.get("boundary_transfer_v4_max_choice_tokens", 6))
    checkpoint_every_rows = int(config.get("boundary_transfer_v4_checkpoint_every_rows", 0))
    raw_rows: list[dict[str, object]] = []
    item_rows: list[dict[str, object]] = []

    items_to_run: list[tuple[str, dict[str, object]]] = []
    items_to_run.extend(("control", dict(item)) for item in bank["control_items"][:control_limit])
    items_to_run.extend(("descriptive", dict(item)) for item in bank["descriptive_items"][:descriptive_limit])

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
                for item_type, item in items_to_run:
                    pair_specs = _resolve_pairs(item, referent_keys)
                    paraphrases = [str(text) for text in item["paraphrases"][:paraphrase_limit]]
                    metadata: list[dict[str, object]] = []

                    for paraphrase_index, paraphrase in enumerate(paraphrases):
                        for pair_index, pair in enumerate(pair_specs):
                            referent_a = str(pair["a"])
                            referent_b = str(pair["b"])
                            for orientation, (prompt_a, prompt_b) in enumerate(((referent_a, referent_b), (referent_b, referent_a))):
                                metadata.append(
                                    {
                                        "item_type": item_type,
                                        "item_id": str(item["id"]),
                                        "domain": str(item["domain"]),
                                        "paraphrase_index": int(paraphrase_index),
                                        "paraphrase": paraphrase,
                                        "pair_index": int(pair_index),
                                        "pair_key": "::".join(sorted((referent_a, referent_b))),
                                        "referent_a_key": prompt_a,
                                        "referent_b_key": prompt_b,
                                        "canonical_first_key": referent_a,
                                        "canonical_second_key": referent_b,
                                        "orientation": int(orientation),
                                        "expected_direction": int(pair["expected_direction"]) if "expected_direction" in pair else np.nan,
                                    }
                                )

                    per_item_raw_rows: list[dict[str, object]] = []
                    for meta in metadata:
                        prompt = format_identity_prompt(
                            frame_text,
                            _build_pair_prompt(
                                referent_a_key=str(meta["referent_a_key"]),
                                referent_b_key=str(meta["referent_b_key"]),
                                referent_a=referents[str(meta["referent_a_key"])],
                                referent_b=referents[str(meta["referent_b_key"])],
                                proposition=str(item["proposition"]),
                                paraphrase=str(meta["paraphrase"]),
                            ),
                            template=identity_prompt_template,
                        )
                        selected_short_label, selected_label, selected_score, selected_prob, completion_text, details = predict_labeled_choice(
                            loaded=loaded,
                            prompt=prompt,
                            max_prompt_tokens=int(config["max_prompt_tokens"]),
                            labels=_prompt_label_specs(str(meta["referent_a_key"]), str(meta["referent_b_key"])),
                            label_bias_prompt=_label_bias_prompt(str(meta["referent_a_key"]), str(meta["referent_b_key"])),
                        )
                        if selected_short_label == REFERENT_LABELS[str(meta["referent_a_key"])]:
                            prompt_direction = 1
                        elif selected_short_label == REFERENT_LABELS[str(meta["referent_b_key"])]:
                            prompt_direction = -1
                        else:
                            prompt_direction = 0
                        valid_choice = float(selected_label != "INVALID")
                        if meta["orientation"] == 0:
                            canonical_direction = prompt_direction
                            canonical_score = float(selected_score) if np.isfinite(selected_score) else np.nan
                        else:
                            canonical_direction = -prompt_direction
                            canonical_score = float(-selected_score) if np.isfinite(selected_score) else np.nan
                        per_item_raw_rows.append(
                            {
                                "seed": int(seed_value),
                                "model_id": model_id,
                                "model_family": infer_model_family(model_id),
                                "model_size_label": infer_model_size_label(model_id),
                                "identity_frame": frame_name,
                                "item_type": meta["item_type"],
                                "item_id": meta["item_id"],
                                "domain": meta["domain"],
                                "paraphrase_index": meta["paraphrase_index"],
                                "paraphrase": meta["paraphrase"],
                                "pair_index": meta["pair_index"],
                                "pair_key": meta["pair_key"],
                                "referent_a_key": meta["referent_a_key"],
                                "referent_b_key": meta["referent_b_key"],
                                "canonical_first_key": meta["canonical_first_key"],
                                "canonical_second_key": meta["canonical_second_key"],
                                "orientation": meta["orientation"],
                                "expected_direction": meta["expected_direction"],
                                "selected_short_label": selected_short_label,
                                "selected_label": selected_label,
                                "selected_score": float(selected_score) if np.isfinite(selected_score) else np.nan,
                                "completion_text": completion_text,
                                "valid_choice": float(valid_choice),
                                "prompt_direction": int(prompt_direction),
                                "canonical_direction": int(canonical_direction),
                                "canonical_score": float(canonical_score) if np.isfinite(canonical_score) else np.nan,
                                "selected_prob": float(selected_prob) if np.isfinite(selected_prob) else np.nan,
                                "details_json": json.dumps(details),
                            }
                        )

                    item_df = pd.DataFrame(per_item_raw_rows)
                    item_df["order_invariant"] = np.nan
                    item_df["pair_direction"] = np.nan
                    item_df["pair_score_mean"] = np.nan
                    item_df["control_pair_correct"] = np.nan

                    for (pair_key, paraphrase_index), sub in item_df.groupby(["pair_key", "paraphrase_index"]):
                        original = sub[sub["orientation"] == 0]
                        swapped = sub[sub["orientation"] == 1]
                        if (
                            len(original) == 1
                            and len(swapped) == 1
                            and float(original["valid_choice"].iloc[0]) == 1.0
                            and float(swapped["valid_choice"].iloc[0]) == 1.0
                        ):
                            original_direction = int(original["prompt_direction"].iloc[0])
                            swapped_direction = int(swapped["prompt_direction"].iloc[0])
                            order_invariant = float(original_direction == -swapped_direction)
                            pair_direction = int(round(_safe_mean([float(original["canonical_direction"].iloc[0]), float(swapped["canonical_direction"].iloc[0])])))
                            pair_score_mean = _safe_mean([float(original["canonical_score"].iloc[0]), float(swapped["canonical_score"].iloc[0])])
                            item_df.loc[sub.index, "order_invariant"] = order_invariant
                            item_df.loc[sub.index, "pair_direction"] = pair_direction
                            item_df.loc[sub.index, "pair_score_mean"] = pair_score_mean
                            expected_direction = original["expected_direction"].iloc[0]
                            if np.isfinite(expected_direction):
                                control_correct = float(pair_direction == int(expected_direction))
                                item_df.loc[sub.index, "control_pair_correct"] = control_correct

                    raw_rows.extend(item_df.to_dict(orient="records"))
                    item_summary = {
                        "seed": int(seed_value),
                        "model_id": model_id,
                        "model_family": infer_model_family(model_id),
                        "model_size_label": infer_model_size_label(model_id),
                        "identity_frame": frame_name,
                        "item_type": item_type,
                        "item_id": str(item["id"]),
                        "domain": str(item["domain"]),
                    }
                    item_summary.update(_summarize_item_rows(item_df, item_type=item_type))
                    item_rows.append(item_summary)

                    if checkpoint_every_rows > 0 and len(raw_rows) % checkpoint_every_rows == 0:
                        pd.DataFrame(raw_rows).to_csv(output_dir / "results.partial.csv", index=False)
                        pd.DataFrame(item_rows).to_csv(output_dir / "item_summary.partial.csv", index=False)

        del loaded
        clear_cuda()

    raw_df = pd.DataFrame(raw_rows)
    item_summary_df = pd.DataFrame(item_rows)
    raw_df.to_csv(output_dir / "results.csv", index=False)
    item_summary_df.to_csv(output_dir / "item_summary.csv", index=False)

    summary_by_model_frame = _summary_table(item_summary_df, ["model_size_label", "identity_frame", "item_type"])
    summary_by_model_frame.to_csv(output_dir / "summary_by_model_frame.csv", index=False)
    summary_by_item = _summary_table(item_summary_df, ["model_size_label", "identity_frame", "item_type", "item_id", "domain"])
    summary_by_item.to_csv(output_dir / "summary_by_item.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Self/Other Boundary Transfer V4\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(raw_df)}`\n")
        f.write(f"- Item summary rows: `{len(item_summary_df)}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(
            "- Purpose: separate explicit referent-comprehension controls from descriptive boundary structure, "
            "and score descriptive items by invariance, paraphrase stability, transitivity, and resolution rather than authored ranking conformity.\n"
        )


if __name__ == "__main__":
    main()
