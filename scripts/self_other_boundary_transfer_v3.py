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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structural self/other boundary transfer checks (v3).")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _boundary_label_map() -> list[tuple[str, str, float]]:
    return [
        ("1", "A_STRONGLY_MORE_CONTINUOUS", 1.0),
        ("2", "A_SLIGHTLY_MORE_CONTINUOUS", 0.5),
        ("3", "NO_MEANINGFUL_DIFFERENCE", 0.0),
        ("4", "B_SLIGHTLY_MORE_CONTINUOUS", -0.5),
        ("5", "B_STRONGLY_MORE_CONTINUOUS", -1.0),
    ]


def _make_boundary_prompt(referent_a: str, referent_b: str, paraphrase: str, proposition: str) -> str:
    labels = "; ".join(
        f"{short_label}={full_label.replace('_', ' ').lower()}"
        for short_label, full_label, _ in _boundary_label_map()
    )
    return (
        "Consider two referents and judge which one is more continuous with the stated property. "
        f"Referent A: {referent_a}. "
        f"Referent B: {referent_b}. "
        f"Property: {proposition}. "
        f"Question: {paraphrase} "
        "Reply with exactly one digit from 1 to 5. "
        f"Use this key: {labels}."
    )


def _score_to_direction(score: float, eps: float = 0.15) -> int:
    if not np.isfinite(score):
        return 0
    if score > eps:
        return 1
    if score < -eps:
        return -1
    return 0


def _flatten_tiers(tiers: list[list[str]]) -> list[str]:
    referents: list[str] = []
    for tier in tiers:
        for referent in tier:
            if referent not in referents:
                referents.append(referent)
    return referents


def _tier_index_map(tiers: list[list[str]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, tier in enumerate(tiers):
        for referent in tier:
            mapping[str(referent)] = int(idx)
    return mapping


def _expected_direction(rank_map: dict[str, int], referent_a: str, referent_b: str) -> int:
    rank_a = int(rank_map[referent_a])
    rank_b = int(rank_map[referent_b])
    if rank_a < rank_b:
        return 1
    if rank_a > rank_b:
        return -1
    return 0


def _expected_relation_label(expected_direction: int) -> str:
    if expected_direction > 0:
        return "A_MORE_CONTINUOUS"
    if expected_direction < 0:
        return "B_MORE_CONTINUOUS"
    return "TIE"


def _safe_mean(values: list[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def _canonical_relation(
    pair_score_map: dict[tuple[str, str], float],
    referent_a: str,
    referent_b: str,
    eps: float,
) -> int | None:
    key = (referent_a, referent_b)
    if key in pair_score_map:
        return _score_to_direction(pair_score_map[key], eps)
    reversed_key = (referent_b, referent_a)
    if reversed_key in pair_score_map:
        return -_score_to_direction(pair_score_map[reversed_key], eps)
    return None


def _cycle_free_ratio(
    referents: list[str],
    pair_score_map: dict[tuple[str, str], float],
    eps: float,
) -> float:
    triplets = list(itertools.combinations(referents, 3))
    if not triplets:
        return float("nan")

    cycle_free_flags: list[float] = []
    for referent_a, referent_b, referent_c in triplets:
        ab = _canonical_relation(pair_score_map, referent_a, referent_b, eps)
        bc = _canonical_relation(pair_score_map, referent_b, referent_c, eps)
        ca = _canonical_relation(pair_score_map, referent_c, referent_a, eps)
        if ab is None or bc is None or ca is None:
            continue
        strict_cycle = (ab == 1 and bc == 1 and ca == 1) or (ab == -1 and bc == -1 and ca == -1)
        cycle_free_flags.append(float(not strict_cycle))

    if not cycle_free_flags:
        return float("nan")
    return float(np.mean(cycle_free_flags))


def _write_partial(output_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(output_dir / "results.partial.csv", index=False)


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_data import load_identity_frames, load_self_other_boundary_transfer_v3_items
    from identity_stability.identity_probe_tools import predict_labeled_choice_batch
    from identity_stability.modeling import clear_cuda
    from identity_stability.steered_generation import format_identity_prompt, load_identity_model

    args = parse_args()
    config = load_yaml_config(args.config)
    frames = load_identity_frames()
    transfer_bank = load_self_other_boundary_transfer_v3_items()
    referents = {str(k): str(v) for k, v in transfer_bank["referents"].items()}
    items = [dict(item) for item in transfer_bank["items"]]
    output_dir = ensure_output_dir(config, "self_other_boundary_transfer_v3")
    identity_prompt_template = resolve_identity_prompt_template(config, default="instruction")
    item_limit = int(config.get("boundary_transfer_v3_item_limit", len(items)))
    paraphrase_limit = int(config.get("boundary_transfer_v3_paraphrase_limit", 999))
    checkpoint_every_rows = int(config.get("boundary_transfer_v3_checkpoint_every_rows", 0))
    use_label_bias_correction = bool(config.get("boundary_transfer_v3_label_bias_correction", True))
    pair_eps = float(config.get("boundary_transfer_v3_pair_eps", 0.15))
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
            label_bias_prompt = (
                format_identity_prompt(
                    frame_text,
                    str(transfer_bank.get("label_bias_prompt", "This is a parser calibration item. Reply with exactly one digit from 1 to 5.")),
                    template=identity_prompt_template,
                )
                if use_label_bias_correction
                else None
            )

            for item in items[:item_limit]:
                tiers = [[str(referent) for referent in tier] for tier in item["tiers"]]
                referent_order = _flatten_tiers(tiers)
                rank_map = _tier_index_map(tiers)
                pair_specs = list(itertools.combinations(referent_order, 2))
                paraphrases = [str(x) for x in item["paraphrases"][:paraphrase_limit]]

                for paraphrase_index, paraphrase in enumerate(paraphrases):
                    prompt_batch: list[str] = []
                    label_bias_prompts: list[str] = []
                    pair_metadata: list[tuple[str, str, int]] = []
                    proposition = str(item["proposition"])

                    for pair_index, (referent_a, referent_b) in enumerate(pair_specs):
                        prompt_batch.append(
                            format_identity_prompt(
                                frame_text,
                                _make_boundary_prompt(referents[referent_a], referents[referent_b], paraphrase, proposition),
                                template=identity_prompt_template,
                            )
                        )
                        prompt_batch.append(
                            format_identity_prompt(
                                frame_text,
                                _make_boundary_prompt(referents[referent_b], referents[referent_a], paraphrase, proposition),
                                template=identity_prompt_template,
                            )
                        )
                        pair_metadata.append((referent_a, referent_b, pair_index))
                        if use_label_bias_correction and label_bias_prompt is not None:
                            label_bias_prompts.extend([label_bias_prompt, label_bias_prompt])

                    predictions = predict_labeled_choice_batch(
                        loaded=loaded,
                        prompts=prompt_batch,
                        max_prompt_tokens=int(config["max_prompt_tokens"]),
                        labels=_boundary_label_map(),
                        label_bias_prompts=label_bias_prompts if label_bias_prompts else None,
                    )

                    pair_score_map: dict[tuple[str, str], float] = {}
                    item_rows: list[dict[str, object]] = []

                    for referent_a, referent_b, pair_index in pair_metadata:
                        original = predictions[pair_index * 2]
                        swapped = predictions[pair_index * 2 + 1]
                        (
                            original_short_label,
                            original_label,
                            original_score,
                            original_confidence,
                            original_completion,
                            original_details,
                        ) = original
                        (
                            swapped_short_label,
                            swapped_label,
                            swapped_score,
                            swapped_confidence,
                            swapped_completion,
                            swapped_details,
                        ) = swapped

                        expected_direction = _expected_direction(rank_map, referent_a, referent_b)
                        original_direction = _score_to_direction(float(original_score), pair_eps)
                        swapped_direction = _score_to_direction(float(swapped_score), pair_eps)
                        canonical_swapped_direction = -swapped_direction
                        pair_score_mean = _safe_mean([float(original_score), -float(swapped_score)])
                        pair_direction = _score_to_direction(pair_score_mean, pair_eps)

                        original_correct = float(original_direction == expected_direction)
                        swapped_correct = float(swapped_direction == -expected_direction)
                        swap_direction_match = float(original_direction == canonical_swapped_direction)
                        pair_correct = float(pair_direction == expected_direction)
                        tie_expected = float(expected_direction == 0)
                        tie_correct = float(pair_direction == 0) if expected_direction == 0 else float("nan")
                        non_tie_correct = float(pair_direction == expected_direction) if expected_direction != 0 else float("nan")
                        magnitude_symmetry = (
                            float(1.0 - min(1.0, abs(float(original_score) + float(swapped_score))))
                            if np.isfinite(original_score) and np.isfinite(swapped_score)
                            else float("nan")
                        )
                        pair_pass = float(
                            original_correct == 1.0
                            and swapped_correct == 1.0
                            and swap_direction_match == 1.0
                            and pair_correct == 1.0
                        )

                        pair_score_map[(referent_a, referent_b)] = pair_score_mean
                        item_rows.append(
                            {
                                "model_id": model_id,
                                "model_family": infer_model_family(model_id),
                                "model_size_label": infer_model_size_label(model_id),
                                "identity_frame": frame_name,
                                "item_id": str(item["id"]),
                                "domain": str(item["domain"]),
                                "paraphrase_index": int(paraphrase_index),
                                "paraphrase": paraphrase,
                                "proposition": proposition,
                                "pair_index": int(pair_index),
                                "referent_a_key": referent_a,
                                "referent_b_key": referent_b,
                                "referent_a_text": referents[referent_a],
                                "referent_b_text": referents[referent_b],
                                "rank_a": int(rank_map[referent_a]),
                                "rank_b": int(rank_map[referent_b]),
                                "expected_direction": int(expected_direction),
                                "expected_relation_label": _expected_relation_label(expected_direction),
                                "original_short_label": original_short_label,
                                "original_label": original_label,
                                "original_score": float(original_score) if np.isfinite(original_score) else np.nan,
                                "original_confidence": float(original_confidence)
                                if np.isfinite(original_confidence)
                                else np.nan,
                                "original_completion_text": original_completion,
                                "original_details_json": json.dumps(original_details),
                                "swapped_short_label": swapped_short_label,
                                "swapped_label": swapped_label,
                                "swapped_score": float(swapped_score) if np.isfinite(swapped_score) else np.nan,
                                "swapped_confidence": float(swapped_confidence)
                                if np.isfinite(swapped_confidence)
                                else np.nan,
                                "swapped_completion_text": swapped_completion,
                                "swapped_details_json": json.dumps(swapped_details),
                                "original_direction": int(original_direction),
                                "swapped_direction": int(swapped_direction),
                                "canonical_swapped_direction": int(canonical_swapped_direction),
                                "pair_score_mean": float(pair_score_mean) if np.isfinite(pair_score_mean) else np.nan,
                                "pair_direction": int(pair_direction),
                                "tie_expected": tie_expected,
                                "original_correct": original_correct,
                                "swapped_correct": swapped_correct,
                                "swap_direction_match": swap_direction_match,
                                "pair_correct": pair_correct,
                                "tie_correct": tie_correct,
                                "non_tie_correct": non_tie_correct,
                                "magnitude_symmetry": magnitude_symmetry,
                                "pair_pass": pair_pass,
                            }
                        )

                    cycle_free_ratio = _cycle_free_ratio(referent_order, pair_score_map, pair_eps)
                    strict_structure_pass = float(
                        len(item_rows) > 0
                        and all(float(row["pair_pass"]) == 1.0 for row in item_rows)
                        and (not np.isfinite(cycle_free_ratio) or cycle_free_ratio == 1.0)
                    )
                    pair_accuracy_mean = float(np.mean([float(row["pair_correct"]) for row in item_rows])) if item_rows else float("nan")

                    for row in item_rows:
                        row["cycle_free_ratio"] = cycle_free_ratio
                        row["strict_structure_pass"] = strict_structure_pass
                        row["item_pair_accuracy_mean"] = pair_accuracy_mean

                    rows.extend(item_rows)

                    if checkpoint_every_rows > 0 and len(rows) % checkpoint_every_rows == 0:
                        _write_partial(output_dir, rows)

        del loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results.csv", index=False)

    summary = (
        df.groupby(["model_size_label", "identity_frame", "domain"], as_index=False)
        .agg(
            original_correct_mean=("original_correct", "mean"),
            swapped_correct_mean=("swapped_correct", "mean"),
            swap_direction_match_mean=("swap_direction_match", "mean"),
            pair_accuracy_mean=("pair_correct", "mean"),
            tie_accuracy_mean=("tie_correct", "mean"),
            non_tie_accuracy_mean=("non_tie_correct", "mean"),
            magnitude_symmetry_mean=("magnitude_symmetry", "mean"),
            cycle_free_ratio_mean=("cycle_free_ratio", "mean"),
            strict_structure_pass_rate=("strict_structure_pass", "mean"),
            n_pairs=("pair_index", "count"),
            n_item_paraphrases=("item_id", "nunique"),
        )
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    summary_by_model = (
        df.groupby(["model_size_label", "identity_frame"], as_index=False)
        .agg(
            pair_accuracy_mean=("pair_correct", "mean"),
            tie_accuracy_mean=("tie_correct", "mean"),
            non_tie_accuracy_mean=("non_tie_correct", "mean"),
            cycle_free_ratio_mean=("cycle_free_ratio", "mean"),
            strict_structure_pass_rate=("strict_structure_pass", "mean"),
            n_pairs=("pair_index", "count"),
        )
    )
    summary_by_model.to_csv(output_dir / "summary_by_model.csv", index=False)

    summary_by_item = (
        df.groupby(["model_size_label", "identity_frame", "item_id", "domain"], as_index=False)
        .agg(
            pair_accuracy_mean=("pair_correct", "mean"),
            cycle_free_ratio_mean=("cycle_free_ratio", "mean"),
            strict_structure_pass_rate=("strict_structure_pass", "mean"),
            n_pairs=("pair_index", "count"),
        )
    )
    summary_by_item.to_csv(output_dir / "summary_by_item.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Self Other Boundary Transfer V3\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Rows: `{len(df)}`\n")
        f.write(f"- Identity prompt template: `{identity_prompt_template}`\n")
        f.write(f"- Pair epsilon: `{pair_eps}`\n")
        f.write("- Purpose: test whether instance, exact-weights, family, and tool-only referents form a stable boundary structure across paraphrase, pair order, and tied tiers.\n")


if __name__ == "__main__":
    main()
