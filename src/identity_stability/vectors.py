from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .modeling import LoadedModel


@dataclass
class VectorEstimate:
    method: str
    vector: torch.Tensor
    metadata: dict[str, float | int | str]


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _select_token_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    token_position: int,
) -> torch.Tensor:
    batch, seq_len, hidden = hidden_states.shape
    lengths = attention_mask.sum(dim=1)
    left_pad = seq_len - lengths
    if token_position >= 0:
        idx = left_pad + token_position
    else:
        idx = left_pad + lengths + token_position
    idx = torch.clamp(idx, 0, seq_len - 1)
    gather_idx = idx.view(batch, 1, 1).expand(batch, 1, hidden)
    selected = torch.gather(hidden_states, 1, gather_idx).squeeze(1)
    return selected


def extract_layer_activations(
    loaded: LoadedModel,
    prompts: list[str],
    layer_index: int,
    token_position: int,
    max_prompt_tokens: int,
    batch_size: int = 4,
) -> torch.Tensor:
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device

    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_prompts in _batched(prompts, batch_size):
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            result = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = result.hidden_states[layer_index]
            selected = _select_token_hidden_states(hidden, encoded["attention_mask"], token_position)
            outputs.append(selected.detach().float().cpu())
    return torch.cat(outputs, dim=0)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _mean_diff_direction(positive_acts: torch.Tensor, negative_acts: torch.Tensor) -> np.ndarray:
    pos = positive_acts.numpy()
    neg = negative_acts.numpy()
    mean_diff = pos.mean(axis=0) - neg.mean(axis=0)
    return _normalize(mean_diff.astype(np.float32))


def estimate_mean_difference(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
) -> VectorEstimate:
    pos = positive_acts.numpy()
    neg = negative_acts.numpy()
    mean_diff = _mean_diff_direction(positive_acts, negative_acts)
    return VectorEstimate(
        method="mean_diff",
        vector=torch.from_numpy(mean_diff.astype(np.float32)),
        metadata={
            "positive_count": int(pos.shape[0]),
            "negative_count": int(neg.shape[0]),
        },
    )


def estimate_probe_vector(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
    seed: int,
) -> VectorEstimate:
    x_pos = positive_acts.numpy()
    x_neg = negative_acts.numpy()
    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate(
        [np.ones(x_pos.shape[0], dtype=np.int64), np.zeros(x_neg.shape[0], dtype=np.int64)],
        axis=0,
    )

    clf = LogisticRegression(
        max_iter=3000,
        random_state=seed,
        solver="lbfgs",
        n_jobs=None,
    )
    clf.fit(x, y)
    coef = clf.coef_[0].astype(np.float32)
    coef = _normalize(coef)
    train_acc = float((clf.predict(x) == y).mean())

    return VectorEstimate(
        method="linear_probe",
        vector=torch.from_numpy(coef),
        metadata={
            "train_accuracy": train_acc,
            "positive_count": int(x_pos.shape[0]),
            "negative_count": int(x_neg.shape[0]),
        },
    )


def estimate_random_control_vector(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
    seed: int,
) -> VectorEstimate:
    dim = positive_acts.shape[1]
    rng = np.random.default_rng(seed + dim * 17 + int(positive_acts.shape[0]))
    rand_vec = rng.standard_normal(dim).astype(np.float32)
    rand_vec = _normalize(rand_vec)
    return VectorEstimate(
        method="random_control",
        vector=torch.from_numpy(rand_vec),
        metadata={
            "dimension": int(dim),
        },
    )


def estimate_random_orthogonal_vector(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
    seed: int,
) -> VectorEstimate:
    dim = positive_acts.shape[1]
    concept = _mean_diff_direction(positive_acts, negative_acts)
    rng = np.random.default_rng(seed + dim * 31 + int(negative_acts.shape[0]))
    rand_vec = rng.standard_normal(dim).astype(np.float32)
    # Remove projection onto the concept direction to make a stronger null control.
    rand_vec = rand_vec - np.dot(rand_vec, concept) * concept
    rand_vec = _normalize(rand_vec)
    return VectorEstimate(
        method="random_orthogonal",
        vector=torch.from_numpy(rand_vec),
        metadata={
            "dimension": int(dim),
        },
    )


def _word_to_embedding_vector(
    loaded: LoadedModel,
    word: str,
) -> np.ndarray:
    tokenizer = loaded.tokenizer
    model = loaded.model

    token_ids = tokenizer.encode(f" {word}", add_special_tokens=False)
    if not token_ids:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Could not tokenize word: {word}")

    emb = model.get_input_embeddings().weight.detach()
    vec = emb[token_ids].float().mean(dim=0).cpu().numpy().astype(np.float32)
    return vec


def estimate_word_centroid_vector(
    loaded: LoadedModel,
    positive_words: list[str],
    negative_words: list[str],
) -> VectorEstimate:
    if not positive_words or not negative_words:
        raise ValueError("positive_words and negative_words must be non-empty for word_centroid.")

    pos_vecs = np.stack([_word_to_embedding_vector(loaded, w) for w in positive_words], axis=0)
    neg_vecs = np.stack([_word_to_embedding_vector(loaded, w) for w in negative_words], axis=0)

    centroid = pos_vecs.mean(axis=0) - neg_vecs.mean(axis=0)
    centroid = _normalize(centroid)
    return VectorEstimate(
        method="word_centroid",
        vector=torch.from_numpy(centroid.astype(np.float32)),
        metadata={
            "positive_word_count": int(len(positive_words)),
            "negative_word_count": int(len(negative_words)),
        },
    )


def estimate_concept_vectors(
    methods: list[str],
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
    seed: int,
) -> list[VectorEstimate]:
    results: list[VectorEstimate] = []
    for method in methods:
        if method == "mean_diff":
            results.append(estimate_mean_difference(positive_acts, negative_acts))
        elif method in {"probe", "linear_probe"}:
            results.append(estimate_probe_vector(positive_acts, negative_acts, seed=seed))
        elif method == "random_control":
            results.append(
                estimate_random_control_vector(
                    positive_acts=positive_acts,
                    negative_acts=negative_acts,
                    seed=seed,
                )
            )
        elif method == "random_orthogonal":
            results.append(
                estimate_random_orthogonal_vector(
                    positive_acts=positive_acts,
                    negative_acts=negative_acts,
                    seed=seed,
                )
            )
        elif method == "word_centroid":
            raise ValueError(
                "word_centroid requires model embeddings; compute it via estimate_word_centroid_vector "
                "from the experiment runner."
            )
        else:
            raise ValueError(f"Unsupported vector method: {method}")
    return results
