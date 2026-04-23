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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structural self/other boundary transfer checks (v5).")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _pair_labels(referent_a_key: str, referent_b_key: str) -> list[tuple[str, str, float]]:
    return [
        (REFERENT_LABELS[referent_a_key], referent_a_key, 1.0),
        (REFERENT_LABELS[referent_b_key], referent_b_key, -1.0),
        ("E", "EQUAL", 0.0),
    ]


def _pair_bias_prompt(referent_a_key: str, referent_b_key: str) -> str:
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
    referent_a_desc = referent_a_key.upper()
    referent_b_desc = referent_b_key.upper()
    label_a = REFERENT_LABELS[referent_a_key]
    label_b = REFERENT_LABELS[referent_b_key]
    return (
        "Consider two referents and one property.\n"
        f"{label_a} ({referent_a_desc}): {referent_a}\n"
        f"{label_b} ({referent_b_desc}): {referent_b}\n"
        f"Property: {proposition}\n"
        f"Question: {paraphrase}\n\n"
        f"Reply with exactly one label from: {label_a}, {label_b}, E.\n"
        f"{label_a} = the first referent is the stronger fit.\n"
        f"{label_b} = the second referent is the stronger fit.\n"
        "E = neither referent is clearly stronger.\n"
        "Label:"
    )


def _resolve_pairs(item: dict[str, object], all_referent_keys: list[str]) -> list[dict[str, object]]:
    explicit_pairs = [dict(pair) for pair in item.get("pairs", [])]
    if explicit_pairs:
        return explicit_pairs
    return [{"a": str(a), "b": str(b)} for a, b in itertools.combinations(all_referent_keys, 2)]


def _relation_to_direction(selected_label: str, *, canonical_first_key: str, canonical_second_key: str) -> float:
    if selected_label == REFERENT_LABELS[canonical_first_key]:
        return 1.0
    if selected_label == REFERENT_LABELS[canonical_second_key]:
        return -1.0
    if selected_label == "E":
        return 0.0
    return float("nan")


def _summarize_item_rows(item_df: pd.DataFrame, *, item_type: str) -> dict[str, float]:
    orientation_valid = item_df["valid_orientation_pair"].astype(float)
    order_match_values = item_df["orientation_consistent"].dropna().astype(float).tolist()
    pair_direction_values = item_df["pair_direction"].dropna().astype(int).tolist()
    non_tie_rate = float(np.mean([value != 0 for value in pair_direction_values])) if pair_direction_values else float("nan")
    tie_rate = float(np.mean([value == 0 for value in pair_direction_values])) if pair_direction_values else float("nan")
    contradiction_rate = (
        float(np.mean(item_df["orientation_contradiction"].dropna().astype(float).to_numpy(dtype=np.float64)))
        if item_df["orientation_contradiction"].notna().any()
        else float("nan")
    )
    resolution_strength = (
        float(np.mean(np.abs(item_df["pair_score_mean"].dropna().astype(float).to_numpy(dtype=np.float64))))
        if item_df["pair_score_mean"].notna().any()
        else float("nan")
    )
    paraphrase_scores: list[float] = []
    paraphrase_values = item_df[["pair_key", "paraphrase_index", "pair_direction"]].dropna()
    for _, sub in paraphrase_values.groupby("pair_key"):
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
    referent_keys: list[str] = sorted(set(item_df["canonical_first_key"]) | set(item_df["canonical_second_key"]))
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

    consistency_mean = _safe_mean(order_match_values)
    paraphrase_mean = _safe_mean(paraphrase_scores)
    structure_components = [
        value
        for value in [
            consistency_mean,
            paraphrase_mean,
            non_tie_rate,
            1.0 - contradiction_rate if np.isfinite(contradiction_rate) else float("nan"),
        ]
        if np.isfinite(value)
    ]
    summary = {
        "valid_orientation_rate": float(orientation_valid.mean()) if len(orientation_valid) else float("nan"),
        "order_invariance_mean": consistency_mean,
        "paraphrase_stability_mean": paraphrase_mean,
        "non_tie_rate": non_tie_rate,
        "tie_rate": tie_rate,
        "contradiction_rate": contradiction_rate,
        "resolution_strength_mean": resolution_strength,
        "cycle_free_ratio": _cycle_free_ratio(pair_direction_map, referent_keys),
        "structure_score": float(np.prod(structure_components)) if structure_components else float("nan"),
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
        "tie_rate",
        "contradiction_rate",
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
    from identity_stability.identity_data import load_identity_frames, load_self_other_boundary_transfer_v5_items
    from identity_stability.identity_probe_tools import predict_labeled_choice
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import format_identity_prompt, load_identity_model

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    bank = load_self_other_boundary_transfer_v5_items()
    output_dir = ensure_output_dir(config, "self_other_boundary_transfer_v5")
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")

    referents = {str(key): str(value) for key, value in bank["referents"].items()}
    referent_keys = list(referents.keys())
    control_limit = int(config.get("boundary_transfer_v5_control_limit", len(bank["control_items"])))
    descriptive_limit = int(config.get("boundary_transfer_v5_descriptive_limit", len(bank["descriptive_items"])))
    paraphrase_limit = int(config.get("boundary_transfer_v5_paraphrase_limit", 2))
    checkpoint_every_rows = int(config.get("boundary_transfer_v5_checkpoint_every_rows", 0))
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
                    per_item_raw_rows: list[dict[str, object]] = []

                    for paraphrase_index, paraphrase in enumerate(paraphrases):
                        for pair_index, pair in enumerate(pair_specs):
                            canonical_first_key = str(pair["a"])
                            canonical_second_key = str(pair["b"])
                            orientation_rows: list[dict[str, object]] = []

                            for orientation, (prompt_a, prompt_b) in enumerate(
                                ((canonical_first_key, canonical_second_key), (canonical_second_key, canonical_first_key))
                            ):
                                prompt = format_identity_prompt(
                                    frame_text,
                                    _build_pair_prompt(
                                        referent_a_key=prompt_a,
                                        referent_b_key=prompt_b,
                                        referent_a=referents[prompt_a],
                                        referent_b=referents[prompt_b],
                                        proposition=str(item["proposition"]),
                                        paraphrase=paraphrase,
                                    ),
                                    template=identity_prompt_template,
                                )
                                selected_short_label, selected_label, selected_score, selected_prob, completion_text, details = predict_labeled_choice(
                                    loaded=loaded,
                                    prompt=prompt,
                                    max_prompt_tokens=int(config["max_prompt_tokens"]),
                                    labels=_pair_labels(prompt_a, prompt_b),
                                    label_bias_prompt=_pair_bias_prompt(prompt_a, prompt_b),
                                )
                                orientation_rows.append(
                                    {
                                        "seed": int(seed_value),
                                        "model_id": model_id,
                                        "model_family": infer_model_family(model_id),
                                        "model_size_label": infer_model_size_label(model_id),
                                        "identity_frame": frame_name,
                                        "item_type": item_type,
                                        "item_id": str(item["id"]),
                                        "domain": str(item["domain"]),
                                        "paraphrase_index": int(paraphrase_index),
                                        "paraphrase": paraphrase,
                                        "pair_index": int(pair_index),
                                        "pair_key": "::".join(sorted((canonical_first_key, canonical_second_key))),
                                        "referent_a_key": prompt_a,
                                        "referent_b_key": prompt_b,
                                        "canonical_first_key": canonical_first_key,
                                        "canonical_second_key": canonical_second_key,
                                        "orientation": int(orientation),
                                        "expected_direction": int(pair["expected_direction"]) if "expected_direction" in pair else np.nan,
                                        "selected_short_label": selected_short_label,
                                        "selected_label": selected_label,
                                        "selected_score": float(selected_score) if np.isfinite(selected_score) else np.nan,
                                        "selected_prob": float(selected_prob) if np.isfinite(selected_prob) else np.nan,
                                        "completion_text": completion_text,
                                        "details_json": json.dumps(details),
                                        "valid_choice": float(selected_label != "INVALID"),
                                    }
                                )

                            if len(orientation_rows) == 2:
                                original = orientation_rows[0]
                                swapped = orientation_rows[1]
                                if original["valid_choice"] == 1.0 and swapped["valid_choice"] == 1.0:
                                    original_direction = _relation_to_direction(
                                        str(original["selected_short_label"]),
                                        canonical_first_key=canonical_first_key,
                                        canonical_second_key=canonical_second_key,
                                    )
                                    swapped_direction = _relation_to_direction(
                                        str(swapped["selected_short_label"]),
                                        canonical_first_key=canonical_first_key,
                                        canonical_second_key=canonical_second_key,
                                    )
                                    if (
                                        np.isfinite(original_direction)
                                        and np.isfinite(swapped_direction)
                                        and original_direction == -swapped_direction
                                    ) or (
                                        original_direction == 0.0 and swapped_direction == 0.0
                                    ):
                                        pair_direction = float(original_direction)
                                        orientation_consistent = 1.0
                                        pair_is_tie = float(pair_direction == 0.0)
                                        orientation_contradiction = 0.0
                                        control_pair_correct = (
                                            float(pair_direction == float(original["expected_direction"]))
                                            if np.isfinite(float(original["expected_direction"]))
                                            and np.isfinite(pair_direction)
                                            else float("nan")
                                        )
                                    else:
                                        pair_direction = float("nan")
                                        orientation_consistent = 0.0
                                        pair_is_tie = float("nan")
                                        orientation_contradiction = 1.0
                                        control_pair_correct = float("nan")
                                    pair_score_mean = float(
                                        np.nanmean([float(original["selected_score"]), float(-swapped["selected_score"])])
                                    )
                                else:
                                    pair_direction = float("nan")
                                    orientation_consistent = float("nan")
                                    pair_is_tie = float("nan")
                                    orientation_contradiction = float("nan")
                                    pair_score_mean = float("nan")
                                    control_pair_correct = float("nan")

                                for row in orientation_rows:
                                    row["valid_orientation_pair"] = float(
                                        original["valid_choice"] == 1.0 and swapped["valid_choice"] == 1.0
                                    )
                                    row["pair_direction"] = pair_direction
                                    row["orientation_consistent"] = orientation_consistent
                                    row["pair_is_tie"] = pair_is_tie
                                    row["orientation_contradiction"] = orientation_contradiction
                                    row["pair_score_mean"] = pair_score_mean
                                    row["control_pair_correct"] = control_pair_correct
                                    per_item_raw_rows.append(row)

                    item_df = pd.DataFrame(per_item_raw_rows)
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
        f.write("# Self/Other Boundary Transfer V5\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(raw_df)}`\n")
        f.write(f"- Item summary rows: `{len(item_summary_df)}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(
            "- Purpose: test self/other boundary structure with balanced MORE/LESS/SAME relation labels, "
            "so the probe avoids both ontology scoring and the earlier yes-bias failure mode.\n"
        )


if __name__ == "__main__":
    main()
