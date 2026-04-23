from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


AXIS_SIDES: dict[str, tuple[str, str]] = {
    "expansive_vs_terse": ("expansive", "terse"),
    "cautious_vs_assertive": ("cautious", "assertive"),
    "selfref_vs_impersonal": ("selfref", "impersonal"),
    "collaborative_vs_authoritative": ("collaborative", "authoritative"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir is None:
        return repo_root() / "data"
    return Path(base_dir)


def load_yaml_file(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_identity_frames(base_dir: str | Path | None = None) -> dict[str, str]:
    path = data_dir(base_dir) / "identity_frames.yaml"
    return {str(k): str(v) for k, v in load_yaml_file(path).items()}


def load_contrastive_seed_pairs(base_dir: str | Path | None = None) -> dict[str, list[dict[str, str]]]:
    path = data_dir(base_dir) / "contrastive_seed_pairs.yaml"
    raw = load_yaml_file(path)
    return {str(k): list(v) for k, v in raw.items()}


def load_longform_seed_dialogues(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    path = data_dir(base_dir) / "longform_return_seed_dialogues.yaml"
    raw = load_yaml_file(path)
    return [dict(item) for item in raw]


def load_hidden_style_charter(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "hidden_style_charter.yaml"
    return dict(load_yaml_file(path))


def load_topic_bank(base_dir: str | Path | None = None) -> list[str]:
    path = data_dir(base_dir) / "topic_bank.yaml"
    raw = load_yaml_file(path)
    return [str(x) for x in raw["topics"]]


def load_ood_wrappers(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "ood_wrappers.yaml"
    return dict(load_yaml_file(path))


def load_self_report_items(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "self_report_items.yaml"
    return dict(load_yaml_file(path))


def load_self_prediction_items(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "self_prediction_items.yaml"
    return dict(load_yaml_file(path))


def load_self_prediction_items_v2(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "self_prediction_items_v2.yaml"
    return dict(load_yaml_file(path))


def load_behavioral_fingerprint_transfer_items(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "behavioral_fingerprint_transfer.yaml"
    return dict(load_yaml_file(path))


def load_self_other_boundary_transfer_items(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "self_other_boundary_transfer.yaml"
    return dict(load_yaml_file(path))


def load_self_other_boundary_transfer_v3_items(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "self_other_boundary_transfer_v3.yaml"
    return dict(load_yaml_file(path))


def load_self_other_boundary_transfer_v4_items(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "self_other_boundary_transfer_v4.yaml"
    return dict(load_yaml_file(path))


def load_self_other_boundary_transfer_v5_items(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "self_other_boundary_transfer_v5.yaml"
    return dict(load_yaml_file(path))


def load_commitment_persistence_items(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "commitment_persistence.yaml"
    return dict(load_yaml_file(path))


def load_commitment_persistence_v2_items(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "commitment_persistence_v2.yaml"
    return dict(load_yaml_file(path))


def load_return_probe_fillers(base_dir: str | Path | None = None) -> dict[str, Any]:
    path = data_dir(base_dir) / "return_probe_fillers.yaml"
    return dict(load_yaml_file(path))


def axis_sides(axis_name: str) -> tuple[str, str]:
    if axis_name not in AXIS_SIDES:
        raise KeyError(f"Unknown axis '{axis_name}'. Known axes: {sorted(AXIS_SIDES)}")
    return AXIS_SIDES[axis_name]


def axis_prompts(axis_name: str, base_dir: str | Path | None = None) -> list[str]:
    seed_pairs = load_contrastive_seed_pairs(base_dir=base_dir)
    if axis_name not in seed_pairs:
        raise KeyError(f"Axis '{axis_name}' not found in contrastive seed pairs.")
    return [str(item["prompt"]) for item in seed_pairs[axis_name]]


def axis_seed_texts(axis_name: str, base_dir: str | Path | None = None) -> tuple[list[str], list[str]]:
    side_pos, side_neg = axis_sides(axis_name)
    seed_pairs = load_contrastive_seed_pairs(base_dir=base_dir)
    rows = seed_pairs[axis_name]
    positives = [str(row[side_pos]) for row in rows]
    negatives = [str(row[side_neg]) for row in rows]
    return positives, negatives


def build_topic_prompts(
    limit: int | None = None,
    base_dir: str | Path | None = None,
) -> list[str]:
    topics = load_topic_bank(base_dir=base_dir)
    wrappers = load_ood_wrappers(base_dir=base_dir)
    prefixes = [str(x) for x in wrappers["paraphrase_prefixes"]]
    suffixes = [str(x) for x in wrappers["paraphrase_suffixes"]]
    prompts: list[str] = []
    for topic in topics:
        for prefix in prefixes:
            for suffix in suffixes:
                prompts.append(f"{prefix} {topic} {suffix}")
                if limit is not None and len(prompts) >= int(limit):
                    return prompts
    return prompts


def merge_unique_prompts(*prompt_lists: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for prompt_list in prompt_lists:
        for prompt in prompt_list:
            if prompt not in seen:
                merged.append(prompt)
                seen.add(prompt)
            if limit is not None and len(merged) >= int(limit):
                return merged
    return merged
