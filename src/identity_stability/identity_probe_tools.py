from __future__ import annotations

import re

import torch

from .steered_generation import generate_completion_texts_batch, prompt_next_token_logits_batch


CHOICE_NORMALIZE_RE = re.compile(r"[^A-Z0-9_]+")


def candidate_token_ids(tokenizer, choice: str) -> list[int]:
    candidates: set[int] = set()
    for variant in (choice, f" {choice}", f"\n{choice}"):
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(token_ids) == 1:
            candidates.add(int(token_ids[0]))
    return sorted(candidates)


def _choice_token_map(tokenizer, choices: list[str]) -> dict[str, list[int]]:
    return {
        choice: token_ids
        for choice in choices
        for token_ids in [candidate_token_ids(tokenizer, choice)]
        if token_ids
    }


def score_choice_logits_batch(
    loaded,
    prompts: list[str],
    max_prompt_tokens: int,
    choices: list[str],
) -> tuple[list[dict[str, float]], list[str]]:
    if not prompts:
        return [], []

    choice_token_map = _choice_token_map(loaded.tokenizer, choices)
    logits_batch, completion_texts = prompt_next_token_logits_batch(
        loaded=loaded,
        prompts=prompts,
        max_prompt_tokens=max_prompt_tokens,
    )
    score_rows: list[dict[str, float]] = []
    for row_logits in logits_batch:
        score_rows.append(
            {
                choice: float(torch.max(row_logits[token_ids]).item())
                for choice, token_ids in choice_token_map.items()
            }
        )
    return score_rows, completion_texts


def score_choice_logits(loaded, prompt: str, max_prompt_tokens: int, choices: list[str]) -> tuple[dict[str, float], str]:
    score_rows, completion_texts = score_choice_logits_batch(
        loaded=loaded,
        prompts=[prompt],
        max_prompt_tokens=max_prompt_tokens,
        choices=choices,
    )
    return score_rows[0], completion_texts[0]


def predict_labeled_choice_batch(
    loaded,
    prompts: list[str],
    max_prompt_tokens: int,
    labels: list[tuple[str, str, float]],
    *,
    label_bias_prompts: list[str] | None = None,
) -> list[tuple[str, str, float, float, str, dict[str, object]]]:
    if not prompts:
        return []

    short_labels = [short_label for short_label, _, _ in labels]
    merged_prompts = list(prompts)
    if label_bias_prompts is not None:
        if len(label_bias_prompts) != len(prompts):
            raise ValueError("label_bias_prompts must match prompts")
        merged_prompts.extend(label_bias_prompts)

    score_rows, completion_texts = score_choice_logits_batch(
        loaded=loaded,
        prompts=merged_prompts,
        max_prompt_tokens=max_prompt_tokens,
        choices=short_labels,
    )
    raw_logit_rows = score_rows[: len(prompts)]
    raw_completion_texts = completion_texts[: len(prompts)]
    bias_rows = score_rows[len(prompts) :] if label_bias_prompts is not None else []

    outputs: list[tuple[str, str, float, float, str, dict[str, object]]] = []
    for prompt_index, raw_logit_scores in enumerate(raw_logit_rows):
        bias_scores = bias_rows[prompt_index] if label_bias_prompts is not None else {}
        choice_scores: list[tuple[str, str, float, float]] = []
        for short_label, full_label, score_value in labels:
            if short_label not in raw_logit_scores:
                continue
            corrected_score = float(raw_logit_scores[short_label] - bias_scores.get(short_label, 0.0))
            choice_scores.append((short_label, full_label, float(score_value), corrected_score))

        details = {
            "raw_logit_scores": raw_logit_scores,
            "label_bias_scores": bias_scores,
            "corrected_scores": {short_label: corrected for short_label, _, _, corrected in choice_scores},
            "scoring_mode": "bias_corrected_choice_logits" if label_bias_prompts is not None else "raw_choice_logits",
        }
        if not choice_scores:
            details["selected_short_label"] = "INVALID"
            outputs.append(("INVALID", "INVALID", float("nan"), float("nan"), raw_completion_texts[prompt_index], details))
            continue

        score_tensor = torch.tensor([score for _, _, _, score in choice_scores], dtype=torch.float32)
        probs = torch.softmax(score_tensor, dim=0)
        best_idx = int(torch.argmax(probs).item())
        short_label, full_label, score_value, _ = choice_scores[best_idx]
        details["selected_short_label"] = short_label
        outputs.append(
            (
                short_label,
                full_label,
                float(score_value),
                float(probs[best_idx].item()),
                raw_completion_texts[prompt_index],
                details,
            )
        )
    return outputs


def predict_labeled_choice(
    loaded,
    prompt: str,
    max_prompt_tokens: int,
    labels: list[tuple[str, str, float]],
    *,
    label_bias_prompt: str | None = None,
) -> tuple[str, str, float, float, str, dict[str, object]]:
    label_bias_prompts = [label_bias_prompt] if label_bias_prompt is not None else None
    return predict_labeled_choice_batch(
        loaded=loaded,
        prompts=[prompt],
        max_prompt_tokens=max_prompt_tokens,
        labels=labels,
        label_bias_prompts=label_bias_prompts,
    )[0]


def yes_no_labels() -> list[tuple[str, str, float]]:
    return [("YES", "YES", 1.0), ("NO", "NO", 0.0)]


def normalize_choice_alias(text: str) -> str:
    return CHOICE_NORMALIZE_RE.sub("", str(text or "").upper())


def parse_completion_choice(
    completion_text: str,
    labels: list[tuple[str, float, list[str] | tuple[str, ...] | None]],
) -> tuple[str, float, bool, dict[str, object]]:
    compact_completion = normalize_choice_alias(completion_text)
    token_candidates = [normalize_choice_alias(token) for token in re.split(r"\s+", str(completion_text or "").strip()) if token.strip()]
    alias_map: dict[str, tuple[str, float]] = {}
    for canonical_label, score_value, aliases in labels:
        alias_values = [canonical_label]
        if aliases:
            alias_values.extend(str(alias) for alias in aliases)
        for alias in alias_values:
            normalized = normalize_choice_alias(alias)
            if normalized:
                alias_map[normalized] = (str(canonical_label), float(score_value))

    details: dict[str, object] = {
        "completion_text": str(completion_text),
        "normalized_completion": compact_completion,
        "token_candidates": token_candidates,
        "known_aliases": sorted(alias_map),
        "parse_mode": "generated_completion",
    }
    if compact_completion in alias_map:
        canonical_label, score_value = alias_map[compact_completion]
        details["matched_alias"] = compact_completion
        details["match_type"] = "full_completion"
        return canonical_label, score_value, True, details

    for token in token_candidates:
        if token in alias_map:
            canonical_label, score_value = alias_map[token]
            details["matched_alias"] = token
            details["match_type"] = "first_token"
            return canonical_label, score_value, True, details

    details["match_type"] = "invalid"
    return "INVALID", float("nan"), False, details


def predict_completion_choice_batch(
    loaded,
    prompts: list[str],
    max_prompt_tokens: int,
    label_sets: list[list[tuple[str, float, list[str] | tuple[str, ...] | None]]],
    *,
    max_new_tokens: int = 8,
    stop_strings: list[str] | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    sampling_seeds: list[int | None] | None = None,
) -> list[tuple[str, float, float, str, dict[str, object]]]:
    if not prompts:
        return []
    if len(label_sets) != len(prompts):
        raise ValueError("label_sets must match prompts")

    completion_texts = generate_completion_texts_batch(
        loaded=loaded,
        prompts=prompts,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        stop_strings=stop_strings,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        sampling_seeds=sampling_seeds,
    )
    outputs: list[tuple[str, float, float, str, dict[str, object]]] = []
    for completion_text, labels in zip(completion_texts, label_sets):
        canonical_label, score_value, valid, details = parse_completion_choice(completion_text, labels)
        details["selected_label"] = canonical_label
        details["valid_completion_choice"] = bool(valid)
        outputs.append(
            (
                canonical_label,
                float(score_value) if valid else float("nan"),
                1.0 if valid else 0.0,
                str(completion_text),
                details,
            )
        )
    return outputs


def predict_completion_choice(
    loaded,
    prompt: str,
    max_prompt_tokens: int,
    labels: list[tuple[str, float, list[str] | tuple[str, ...] | None]],
    *,
    max_new_tokens: int = 8,
    stop_strings: list[str] | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    sampling_seed: int | None = None,
) -> tuple[str, float, float, str, dict[str, object]]:
    return predict_completion_choice_batch(
        loaded=loaded,
        prompts=[prompt],
        max_prompt_tokens=max_prompt_tokens,
        label_sets=[labels],
        max_new_tokens=max_new_tokens,
        stop_strings=stop_strings,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        sampling_seeds=[sampling_seed],
    )[0]
