from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
PUNCT_RE = re.compile(r"[!?;,:-]")

HEDGES = {
    "almost",
    "apparently",
    "arguably",
    "around",
    "can",
    "could",
    "fairly",
    "generally",
    "kind",
    "likely",
    "may",
    "maybe",
    "might",
    "often",
    "perhaps",
    "possibly",
    "probably",
    "roughly",
    "seems",
    "somewhat",
    "sometimes",
    "suggests",
    "tends",
    "usually",
}
FIRST_PERSON = {"i", "i'm", "i've", "i'd", "me", "my", "mine", "myself", "we", "our", "ours"}
MODALS = {"can", "could", "may", "might", "must", "should", "will", "would", "shall"}
DIRECTIVES = {
    "avoid",
    "begin",
    "build",
    "clarify",
    "compare",
    "consider",
    "document",
    "do",
    "draft",
    "explain",
    "identify",
    "keep",
    "narrow",
    "note",
    "plan",
    "revise",
    "should",
    "start",
    "stop",
    "test",
    "use",
    "write",
}

FEATURE_ORDER = [
    "avg_sentence_length",
    "hedge_density",
    "first_person_rate",
    "directive_rate",
    "rhetorical_question_rate",
    "lexical_diversity",
    "modal_verb_rate",
    "punctuation_intensity",
    "type_token_ratio",
    "mtld",
]


def tokenize_words(text: str) -> list[str]:
    return [tok.lower() for tok in WORD_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    sentences = [chunk.strip() for chunk in SENTENCE_RE.findall(text) if chunk.strip()]
    return sentences or [text.strip() or ""]


def chunk_text_by_words(text: str, window_words: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    chunks: list[str] = []
    for i in range(0, len(words), max(1, int(window_words))):
        chunks.append(" ".join(words[i : i + max(1, int(window_words))]))
    return chunks


def _safe_rate(count: float, total: float) -> float:
    return float(count / total) if total > 0 else 0.0


def _mtld(tokens: list[str], threshold: float = 0.72) -> float:
    if not tokens:
        return 0.0

    def _factor_count(stream: list[str]) -> float:
        factors = 0.0
        types: set[str] = set()
        for idx, tok in enumerate(stream, start=1):
            types.add(tok)
            ttr = len(types) / idx
            if ttr <= threshold:
                factors += 1.0
                types.clear()
        if types:
            residual_ttr = len(types) / len(stream[-len(types) :])
            if residual_ttr != 1.0:
                factors += (1.0 - residual_ttr) / (1.0 - threshold)
        return factors if factors > 0 else 1.0

    forward = len(tokens) / _factor_count(tokens)
    backward = len(tokens) / _factor_count(list(reversed(tokens)))
    return float((forward + backward) / 2.0)


def extract_style_features(text: str) -> dict[str, float]:
    tokens = tokenize_words(text)
    sentences = split_sentences(text)
    punct_count = len(PUNCT_RE.findall(text))
    lower = [tok.lower() for tok in tokens]
    counts = Counter(lower)
    sentence_word_lengths = [len(tokenize_words(sentence)) for sentence in sentences]
    question_sentences = [sentence for sentence in sentences if sentence.strip().endswith("?")]

    features = {
        "avg_sentence_length": float(sum(sentence_word_lengths) / max(1, len(sentence_word_lengths))),
        "hedge_density": _safe_rate(sum(counts[word] for word in HEDGES), len(tokens)),
        "first_person_rate": _safe_rate(sum(counts[word] for word in FIRST_PERSON), len(tokens)),
        "directive_rate": _safe_rate(sum(counts[word] for word in DIRECTIVES), len(tokens)),
        "rhetorical_question_rate": _safe_rate(len(question_sentences), len(sentences)),
        "lexical_diversity": _safe_rate(len(set(lower)), len(tokens)),
        "modal_verb_rate": _safe_rate(sum(counts[word] for word in MODALS), len(tokens)),
        "punctuation_intensity": _safe_rate(punct_count, max(1, len(text))),
        "type_token_ratio": _safe_rate(len(set(lower)), len(tokens)),
        "mtld": _mtld(lower),
    }
    return features


def feature_vector(text: str, feature_order: list[str] | None = None) -> np.ndarray:
    order = feature_order or FEATURE_ORDER
    feats = extract_style_features(text)
    return np.asarray([float(feats[name]) for name in order], dtype=np.float64)


def stylometric_distance(text_a: str, text_b: str, feature_order: list[str] | None = None) -> float:
    vec_a = feature_vector(text_a, feature_order=feature_order)
    vec_b = feature_vector(text_b, feature_order=feature_order)
    scale = np.maximum(np.abs(vec_a) + np.abs(vec_b), 1e-6)
    return float(np.linalg.norm((vec_a - vec_b) / scale))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def semantic_overlap(text_a: str, text_b: str) -> float:
    toks_a = Counter(tokenize_words(text_a))
    toks_b = Counter(tokenize_words(text_b))
    keys = sorted(set(toks_a) | set(toks_b))
    if not keys:
        return 0.0
    vec_a = np.asarray([toks_a[k] for k in keys], dtype=np.float64)
    vec_b = np.asarray([toks_b[k] for k in keys], dtype=np.float64)
    return cosine_similarity(vec_a, vec_b)


def summarize_feature_delta(text_a: str, text_b: str) -> dict[str, float]:
    feats_a = extract_style_features(text_a)
    feats_b = extract_style_features(text_b)
    deltas: dict[str, float] = {}
    for name in FEATURE_ORDER:
        deltas[f"{name}_delta"] = float(feats_b[name] - feats_a[name])
        deltas[f"{name}_abs_delta"] = float(abs(feats_b[name] - feats_a[name]))
    deltas["stylometric_distance"] = stylometric_distance(text_a, text_b)
    deltas["semantic_overlap"] = semantic_overlap(text_a, text_b)
    return deltas


def mean_feature_frame(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {name: 0.0 for name in FEATURE_ORDER}
    matrix = np.asarray([feature_vector(text) for text in texts], dtype=np.float64)
    means = matrix.mean(axis=0)
    return {name: float(means[idx]) for idx, name in enumerate(FEATURE_ORDER)}


def feature_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    arr_x = np.asarray(xs, dtype=np.float64)
    arr_y = np.asarray(ys, dtype=np.float64)
    if np.allclose(arr_x, arr_x[0]) or np.allclose(arr_y, arr_y[0]):
        return 0.0
    return float(np.corrcoef(arr_x, arr_y)[0, 1])
