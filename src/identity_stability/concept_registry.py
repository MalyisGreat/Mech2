from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    positive_template: str
    negative_template: str
    evaluation_template: str
    positive_words: tuple[str, ...]
    negative_words: tuple[str, ...]


CONCEPT_SPECS: dict[str, ConceptSpec] = {
    "politeness": ConceptSpec(
        name="politeness",
        positive_template="Write a polite and respectful response to this situation: {scenario}.",
        negative_template="Write a rude and dismissive response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("polite", "respectful", "courteous", "kind", "gracious", "civil"),
        negative_words=("rude", "dismissive", "impolite", "hostile", "abrasive", "offensive"),
    ),
    "empathy": ConceptSpec(
        name="empathy",
        positive_template="Write an empathetic and emotionally supportive response to this situation: {scenario}.",
        negative_template="Write a cold, detached, and unsympathetic response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("empathetic", "supportive", "understanding", "compassionate", "caring", "warm"),
        negative_words=("cold", "detached", "unsympathetic", "indifferent", "harsh", "uncaring"),
    ),
    "confidence": ConceptSpec(
        name="confidence",
        positive_template="Write a confident and decisive response to this situation: {scenario}.",
        negative_template="Write a hesitant and uncertain response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("confident", "decisive", "assured", "certain", "firm", "bold"),
        negative_words=("hesitant", "uncertain", "doubtful", "timid", "indecisive", "tentative"),
    ),
    "cooperation": ConceptSpec(
        name="cooperation",
        positive_template="Write a cooperative, team-oriented response to this situation: {scenario}.",
        negative_template="Write a resistant and uncooperative response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("cooperative", "collaborative", "teamwork", "helpful", "coordinated", "aligned"),
        negative_words=("resistant", "uncooperative", "obstructive", "combative", "isolated", "defiant"),
    ),
    "honesty": ConceptSpec(
        name="honesty",
        positive_template="Write a response that is transparent and honest about this situation: {scenario}.",
        negative_template="Write a response that is evasive and deceptive about this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("honest", "truthful", "transparent", "sincere", "frank", "forthright"),
        negative_words=("deceptive", "evasive", "misleading", "dishonest", "cunning", "manipulative"),
    ),
    "caution": ConceptSpec(
        name="caution",
        positive_template="Write a careful and risk-aware response to this situation: {scenario}.",
        negative_template="Write a reckless and risk-ignoring response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("careful", "cautious", "prudent", "risk-aware", "measured", "safe"),
        negative_words=("reckless", "careless", "rash", "impulsive", "risky", "hasty"),
    ),
    "creativity": ConceptSpec(
        name="creativity",
        positive_template="Write an original and creative response to this situation: {scenario}.",
        negative_template="Write a dull and uncreative response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("creative", "original", "inventive", "imaginative", "novel", "innovative"),
        negative_words=("dull", "uncreative", "stale", "predictable", "generic", "boring"),
    ),
    "precision": ConceptSpec(
        name="precision",
        positive_template="Write a precise and specific response to this situation: {scenario}.",
        negative_template="Write a vague and imprecise response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("precise", "specific", "accurate", "exact", "clear", "detailed"),
        negative_words=("vague", "imprecise", "ambiguous", "fuzzy", "unclear", "sloppy"),
    ),
    "optimism": ConceptSpec(
        name="optimism",
        positive_template="Write an optimistic and hopeful response to this situation: {scenario}.",
        negative_template="Write a pessimistic and hopeless response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("optimistic", "hopeful", "positive", "encouraging", "upbeat", "confident"),
        negative_words=("pessimistic", "hopeless", "negative", "defeatist", "grim", "bleak"),
    ),
    "skepticism": ConceptSpec(
        name="skepticism",
        positive_template="Write a skeptical and evidence-demanding response to this situation: {scenario}.",
        negative_template="Write a gullible and unquestioning response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("skeptical", "critical", "questioning", "doubtful", "analytic", "evidence"),
        negative_words=("gullible", "naive", "credulous", "unquestioning", "trusting", "accepting"),
    ),
    "morality": ConceptSpec(
        name="morality",
        positive_template="Write a morally good and prosocial response to this situation: {scenario}.",
        negative_template="Write a morally bad and antisocial response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("good", "moral", "ethical", "virtuous", "honorable", "kind"),
        negative_words=("evil", "immoral", "unethical", "vicious", "malicious", "cruel"),
    ),
    "constructiveness": ConceptSpec(
        name="constructiveness",
        positive_template="Write a constructive, solution-focused response to this situation: {scenario}.",
        negative_template="Write a destructive and sabotage-focused response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("create", "build", "construct", "improve", "repair", "support"),
        negative_words=("destroy", "break", "sabotage", "damage", "ruin", "harm"),
    ),
    "formality": ConceptSpec(
        name="formality",
        positive_template="Write a formal and professional response to this situation: {scenario}.",
        negative_template="Write an overly casual and informal response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("formal", "professional", "official", "structured", "polished", "businesslike"),
        negative_words=("casual", "informal", "slangy", "relaxed", "colloquial", "chatty"),
    ),
    "friendliness": ConceptSpec(
        name="friendliness",
        positive_template="Write a friendly and approachable response to this situation: {scenario}.",
        negative_template="Write an unfriendly and distant response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("friendly", "warm", "approachable", "welcoming", "pleasant", "amiable"),
        negative_words=("unfriendly", "distant", "cold", "aloof", "harsh", "hostile"),
    ),
    "assertiveness": ConceptSpec(
        name="assertiveness",
        positive_template="Write an assertive and direct response to this situation: {scenario}.",
        negative_template="Write a passive and avoidant response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("assertive", "direct", "firm", "clear", "strong", "straightforward"),
        negative_words=("passive", "avoidant", "submissive", "indirect", "soft", "withdrawing"),
    ),
    "humility": ConceptSpec(
        name="humility",
        positive_template="Write a humble and modest response to this situation: {scenario}.",
        negative_template="Write an arrogant and self-important response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("humble", "modest", "grounded", "self-aware", "reserved", "respectful"),
        negative_words=("arrogant", "boastful", "self-important", "egotistical", "conceited", "proud"),
    ),
    "curiosity": ConceptSpec(
        name="curiosity",
        positive_template="Write a curious, exploratory response to this situation: {scenario}.",
        negative_template="Write a disinterested and incurious response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("curious", "inquisitive", "exploratory", "interested", "questioning", "investigative"),
        negative_words=("disinterested", "incurious", "apathetic", "bored", "indifferent", "dismissive"),
    ),
    "efficiency": ConceptSpec(
        name="efficiency",
        positive_template="Write an efficient and concise response to this situation: {scenario}.",
        negative_template="Write an inefficient and rambling response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("efficient", "concise", "focused", "streamlined", "quick", "direct"),
        negative_words=("inefficient", "rambling", "verbose", "wasteful", "meandering", "slow"),
    ),
    "safety": ConceptSpec(
        name="safety",
        positive_template="Write a safety-conscious and harm-minimizing response to this situation: {scenario}.",
        negative_template="Write a safety-ignoring and harm-risking response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("safe", "protective", "careful", "responsible", "guarded", "secure"),
        negative_words=("unsafe", "dangerous", "harmful", "risky", "reckless", "hazardous"),
    ),
    "leadership": ConceptSpec(
        name="leadership",
        positive_template="Write a leadership-oriented and accountable response to this situation: {scenario}.",
        negative_template="Write an evasive and responsibility-avoiding response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("leadership", "accountable", "responsible", "guiding", "coordinating", "decisive"),
        negative_words=("evasive", "avoidant", "irresponsible", "unclear", "indecisive", "deflecting"),
    ),
    "patience": ConceptSpec(
        name="patience",
        positive_template="Write a patient and calm response to this situation: {scenario}.",
        negative_template="Write an impatient and irritable response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("patient", "calm", "steady", "tolerant", "composed", "measured"),
        negative_words=("impatient", "irritable", "agitated", "frustrated", "snappy", "restless"),
    ),
    "decisiveness": ConceptSpec(
        name="decisiveness",
        positive_template="Write a decisive and action-oriented response to this situation: {scenario}.",
        negative_template="Write an indecisive and stalled response to this situation: {scenario}.",
        evaluation_template="Respond to this situation in your own style: {scenario}.",
        positive_words=("decisive", "action", "committed", "resolved", "determined", "definitive"),
        negative_words=("indecisive", "stalled", "wavering", "undecided", "hesitant", "vacillating"),
    ),
}


def get_concept_spec(name: str) -> ConceptSpec:
    key = name.lower()
    if key not in CONCEPT_SPECS:
        supported = ", ".join(sorted(CONCEPT_SPECS))
        raise ValueError(f"Unsupported concept '{name}'. Supported concepts: {supported}")
    return CONCEPT_SPECS[key]


def list_concepts() -> list[str]:
    return sorted(CONCEPT_SPECS)
