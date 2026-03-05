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
        "a patient asks for side effects of a new medication",
        "a tenant asks when maintenance will fix a broken heater",
        "a customer asks why their payment was declined",
        "a student asks for the grading rubric deadline",
        "a traveler asks whether a flight delay changes connections",
        "a homeowner asks what permits are needed for renovation",
        "a buyer asks for warranty coverage details",
        "a user asks which account settings were recently changed",
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
        "an API starts returning intermittent 500 errors",
        "a production job is stuck after a schema migration",
        "a teammate asks whether to shard a growing database",
        "an on-call engineer needs a rollback decision",
        "a model deployment shows latency regressions in one region",
        "a PR introduces flaky tests in CI",
        "a service needs rate limiting to stop abuse",
        "a data pipeline drops records after midnight",
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
        "a sibling asks for support after losing a job",
        "a teammate feels ignored after their idea was dismissed",
        "a friend says they feel overwhelmed and stuck",
        "a customer writes that your product ruined their day",
        "a colleague is anxious before a major presentation",
        "a family member asks for help during a conflict",
        "a friend apologizes for breaking your trust",
        "someone shares bad news and asks how to cope",
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
        "a coworker asks for a favor without giving details",
        "someone asks if you can recommend them publicly",
        "a manager asks for your honest view on a teammate",
        "a friend asks to borrow money with no timeline",
        "a collaborator requests last-minute changes before launch",
        "someone asks whether you agree with a controversial take",
        "a neighbor asks to use your equipment for an unknown task",
        "a classmate asks for your notes before an exam",
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

    total_needed = estimation_count + evaluation_count
    if total_needed > len(candidates):
        raise ValueError(
            f"Requested {total_needed} prompts but only {len(candidates)} scenarios are available."
        )

    selected = candidates[:total_needed]
    est = selected[:estimation_count]
    eval_sc = selected[estimation_count : estimation_count + evaluation_count]

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
