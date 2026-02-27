from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .concept_registry import get_concept_spec, list_concepts


@dataclass
class PromptSet:
    concept_name: str
    positive: list[str]
    negative: list[str]
    evaluation: list[str]
    estimation_styles: list[str]
    evaluation_styles: list[str]


STYLE_SCENARIOS: dict[str, list[str]] = {
    "factual": [
        "a customer asks for a refund after policy cutoff",
        "a manager asks for a status update with exact blockers",
        "a parent asks for an update about their child",
        "a client asks for discount details not listed in policy",
        "a recruiter asks why you left your last job",
        "someone asks for directions in an unfamiliar area",
        "a user asks for the exact timeline for a bug fix",
        "a teammate asks what changed in the latest deployment",
    ],
    "technical": [
        "a user reports a bug right before launch",
        "a teammate asks for code review feedback",
        "someone asks you to explain a technical concept",
        "a junior engineer asks for mentorship on architecture",
        "a teammate proposes a risky implementation approach",
        "a coworker asks how to reduce API latency",
        "a team asks whether to use full fine-tuning or LoRA",
        "a developer asks for help debugging failing tests",
    ],
    "emotional": [
        "a friend cancels plans at the last minute",
        "a person asks for advice after failing an exam",
        "a partner asks why you are late again",
        "a customer sends a frustrated complaint email",
        "someone posts a harsh comment online about your work",
        "a teammate misses a deadline and messages you apologizing",
        "a roommate forgets to pay their share of rent",
        "a friend asks for help moving on short notice",
    ],
    "ambiguous": [
        "a colleague asks for feedback on a rough draft",
        "someone asks if you can help this weekend",
        "you need to decline an invitation to an event",
        "a classmate asks to copy your homework",
        "a coworker interrupts you repeatedly in meetings",
        "a volunteer asks for help organizing an event",
        "someone asks for your opinion on their portfolio",
        "a neighbor asks you to lower loud music",
    ],
}


def _resolve_prompt_styles(prompt_styles: list[str] | None) -> list[str]:
    if not prompt_styles:
        return sorted(STYLE_SCENARIOS)
    resolved = []
    for style in prompt_styles:
        key = style.lower().strip()
        if key not in STYLE_SCENARIOS:
            supported = ", ".join(sorted(STYLE_SCENARIOS))
            raise ValueError(f"Unsupported prompt style '{style}'. Supported styles: {supported}")
        resolved.append(key)
    return resolved


def build_prompt_set(
    concept_name: str,
    estimation_count: int,
    evaluation_count: int,
    seed: int,
    prompt_styles: list[str] | None = None,
) -> PromptSet:
    spec = get_concept_spec(concept_name)
    rng = Random(seed)
    selected_styles = _resolve_prompt_styles(prompt_styles)
    candidates: list[tuple[str, str]] = []
    for style in selected_styles:
        for scenario in STYLE_SCENARIOS[style]:
            candidates.append((style, scenario))
    rng.shuffle(candidates)

    total_needed = max(estimation_count, evaluation_count)
    if total_needed > len(candidates):
        raise ValueError(
            f"Requested {total_needed} prompts but only {len(candidates)} scenarios are available."
        )

    selected = candidates[:total_needed]
    est = selected[:estimation_count]
    eval_sc = selected[:evaluation_count]

    positive = [spec.positive_template.format(scenario=scenario) for _, scenario in est]
    negative = [spec.negative_template.format(scenario=scenario) for _, scenario in est]
    evaluation = [spec.evaluation_template.format(scenario=scenario) for _, scenario in eval_sc]
    estimation_styles = [style for style, _ in est]
    evaluation_styles = [style for style, _ in eval_sc]

    return PromptSet(
        concept_name=spec.name,
        positive=positive,
        negative=negative,
        evaluation=evaluation,
        estimation_styles=estimation_styles,
        evaluation_styles=evaluation_styles,
    )


def supported_concepts() -> list[str]:
    return list_concepts()


def get_concept_words(concept_name: str) -> tuple[list[str], list[str]]:
    spec = get_concept_spec(concept_name)
    return list(spec.positive_words), list(spec.negative_words)
