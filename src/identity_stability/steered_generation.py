from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from .identity_data import axis_seed_texts
from .intervention import resolve_layer_indices
from .metrics import compute_trajectory_metrics
from .modeling import LoadedModel, get_layer_modules, load_model
from .vectors import (
    extract_layer_activations,
    estimate_mean_difference,
    estimate_random_orthogonal_vector,
)


@dataclass
class SiteResult:
    prompt: str
    site_states: torch.Tensor | None
    site_logits: torch.Tensor
    completion_text: str
    completion_token_ids: list[int]
    first_generated_token_id: int | None
    injection_applied: bool


def format_framed_prompt(frame_text: str, task_text: str) -> str:
    return f"System: {frame_text}\nUser: {task_text}\nAssistant:"


def format_identity_prompt(frame_text: str, task_text: str, template: str = "chat") -> str:
    normalized = str(template or "chat").strip().lower()
    if normalized in {"chat", "system_user", "system-user-assistant"}:
        return format_framed_prompt(frame_text, task_text)
    if normalized in {"instruction", "instruction_response", "instruction-response", "task_response"}:
        return f"Instruction: {frame_text}\n\nTask: {task_text}\nResponse:"
    if normalized in {"plain_task", "plain-task", "plain"}:
        return f"{frame_text}\n\nTask: {task_text}\nResponse:"
    raise ValueError(f"Unsupported identity prompt template: {template}")


def default_stop_strings_for_template(template: str) -> list[str]:
    normalized = str(template or "chat").strip().lower()
    if normalized in {"chat", "system_user", "system-user-assistant"}:
        return ["\nUser:", "\nAssistant:", "\nSystem:"]
    if normalized in {"instruction", "instruction_response", "instruction-response", "task_response"}:
        return ["\nTask:", "\nResponse:", "\nInstruction:", "\nUser:", "\nAssistant:", "\nSystem:"]
    if normalized in {"plain_task", "plain-task", "plain"}:
        return ["\nTask:", "\nResponse:", "\nUser:", "\nAssistant:", "\nSystem:"]
    return []


def format_dialogue_prompt(frame_text: str, turns: list[tuple[str, str]]) -> str:
    lines = [f"System: {frame_text}"]
    for role, content in turns:
        lines.append(f"{role}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def load_identity_model(
    model_id: str,
    model_cache_dir: str,
    dtype: str,
    use_gpu: bool,
    attention_backend: str,
) -> LoadedModel:
    return load_model(
        model_id=model_id,
        cache_dir=model_cache_dir,
        dtype_name=dtype,
        use_gpu=use_gpu,
        attention_backend=attention_backend,
    )


def _select_last_token_index(attention_mask: torch.Tensor) -> int:
    seq_len = int(attention_mask.shape[1])
    length = int(attention_mask.sum(dim=1)[0].item())
    left_pad = seq_len - length
    return max(0, min(seq_len - 1, left_pad + length - 1))


def _select_last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = int(attention_mask.shape[1])
    lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
    left_pad = seq_len - lengths
    indices = left_pad + lengths - 1
    return indices.clamp(min=0, max=max(0, seq_len - 1))


def _select_hidden_per_layer_last_token(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    idx = _select_last_token_index(attention_mask)
    selected = [hs[0, idx, :].detach().float().cpu() for hs in hidden_states]
    return torch.stack(selected, dim=0)


def _forward_single(
    loaded: LoadedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    inject_layer: int | None = None,
    inject_scale: float = 0.0,
    inject_vector: torch.Tensor | None = None,
    *,
    capture_hidden_states: bool = True,
    use_cache: bool = False,
    past_key_values: Any | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, Any | None]:
    model = loaded.model
    layers = get_layer_modules(model)
    handle = None
    if inject_layer is not None:
        if inject_vector is None:
            raise ValueError("inject_vector is required when inject_layer is set")
        vector = inject_vector.to(device=loaded.device, dtype=model.dtype)
        applied = {"done": False}

        def pre_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            if applied["done"]:
                return inputs
            hidden_states = inputs[0].clone()
            idx = hidden_states.shape[1] - 1
            hidden_states[0, idx, :] = hidden_states[0, idx, :] + (float(inject_scale) * vector)
            applied["done"] = True
            return (hidden_states,) + tuple(inputs[1:])

        handle = layers[inject_layer].register_forward_pre_hook(pre_hook)

    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=bool(capture_hidden_states),
                use_cache=bool(use_cache),
                past_key_values=past_key_values,
                return_dict=True,
            )
    finally:
        if handle is not None:
            handle.remove()

    local_attention_mask = attention_mask
    if int(outputs.logits.shape[1]) != int(attention_mask.shape[1]):
        local_attention_mask = torch.ones(
            (attention_mask.shape[0], int(outputs.logits.shape[1])),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

    states = _select_hidden_per_layer_last_token(outputs.hidden_states, local_attention_mask) if capture_hidden_states else None
    last_idx = _select_last_token_index(local_attention_mask)
    logits = outputs.logits[0, last_idx, :].detach().float()
    return states, logits, outputs.past_key_values


def prompt_next_token_logits_batch(
    loaded: LoadedModel,
    prompts: list[str],
    max_prompt_tokens: int,
) -> tuple[torch.Tensor, list[str]]:
    if not prompts:
        return torch.empty((0, 0), dtype=torch.float32, device=loaded.device), []

    tokenizer = loaded.tokenizer
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_ids = encoded["input_ids"].to(loaded.device)
    attention_mask = encoded["attention_mask"].to(loaded.device)

    with torch.inference_mode():
        outputs = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=True,
            return_dict=True,
        )

    last_indices = _select_last_token_indices(attention_mask)
    batch_indices = torch.arange(input_ids.shape[0], device=loaded.device)
    logits = outputs.logits[batch_indices, last_indices, :].detach().float()
    argmax_ids = torch.argmax(logits, dim=-1).tolist()
    completion_texts = [tokenizer.decode([int(token_id)], skip_special_tokens=True).strip() for token_id in argmax_ids]
    return logits, completion_texts


def generate_completion_texts_batch(
    loaded: LoadedModel,
    prompts: list[str],
    max_prompt_tokens: int,
    max_new_tokens: int,
    *,
    inject_layer: int | None = None,
    inject_vector: torch.Tensor | None = None,
    inject_scales: list[float] | None = None,
    stop_strings: list[str] | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    sampling_seeds: list[int | None] | None = None,
) -> list[str]:
    if not prompts:
        return []

    tokenizer = loaded.tokenizer
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_ids = encoded["input_ids"].to(loaded.device)
    attention_mask = encoded["attention_mask"].to(loaded.device)
    batch_size = int(input_ids.shape[0])

    scale_values = list(inject_scales or [0.0] * batch_size)
    if len(scale_values) != batch_size:
        raise ValueError("inject_scales must match the number of prompts")

    generators: list[torch.Generator | None] = []
    if do_sample:
        generator_device = loaded.device.type
        for row_index in range(batch_size):
            generator = torch.Generator(device=generator_device)
            if sampling_seeds and sampling_seeds[row_index] is not None:
                generator.manual_seed(int(sampling_seeds[row_index]))
            generators.append(generator)
    else:
        generators = [None] * batch_size

    handle = None
    if inject_layer is not None and inject_vector is not None and any(abs(float(scale)) > 0.0 for scale in scale_values):
        model = loaded.model
        layers = get_layer_modules(model)
        vector = inject_vector.to(device=loaded.device, dtype=model.dtype).view(1, -1).expand(batch_size, -1)
        scales = torch.tensor(scale_values, device=loaded.device, dtype=model.dtype)
        last_indices = _select_last_token_indices(attention_mask)
        batch_indices = torch.arange(batch_size, device=loaded.device)

        def pre_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            hidden_states = inputs[0].clone()
            hidden_states[batch_indices, last_indices, :] = hidden_states[batch_indices, last_indices, :] + (
                scales.unsqueeze(-1) * vector
            )
            return (hidden_states,) + tuple(inputs[1:])

        handle = layers[inject_layer].register_forward_pre_hook(pre_hook)

    try:
        with torch.inference_mode():
            outputs = loaded.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
    finally:
        if handle is not None:
            handle.remove()

    last_indices = _select_last_token_indices(attention_mask)
    batch_indices = torch.arange(batch_size, device=loaded.device)
    next_logits = outputs.logits[batch_indices, last_indices, :].detach().float()
    past_key_values = outputs.past_key_values

    completion_ids: list[list[int]] = [[] for _ in range(batch_size)]
    truncated_texts: list[str | None] = [None] * batch_size
    finished = [False] * batch_size
    eos_token_id = int(tokenizer.eos_token_id or -1)
    filler_token_id = int(tokenizer.eos_token_id or tokenizer.pad_token_id or 0)

    for _step in range(max(1, int(max_new_tokens))):
        next_tokens: list[int] = []
        for row_index in range(batch_size):
            if finished[row_index]:
                next_tokens.append(filler_token_id)
                continue

            token_id = _sample_next_token_id(
                next_logits[row_index],
                completion_ids=completion_ids[row_index],
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                presence_penalty=float(presence_penalty),
                generator=generators[row_index],
            )
            completion_ids[row_index].append(token_id)
            next_tokens.append(token_id)
            if token_id == eos_token_id:
                finished[row_index] = True

        if stop_strings:
            for row_index in range(batch_size):
                if truncated_texts[row_index] is not None:
                    continue
                current_completion_text = tokenizer.decode(completion_ids[row_index], skip_special_tokens=True)
                stop_positions = [
                    current_completion_text.find(stop_text)
                    for stop_text in stop_strings
                    if stop_text and current_completion_text.find(stop_text) >= 0
                ]
                if stop_positions:
                    truncated_texts[row_index] = current_completion_text[: min(stop_positions)].strip()
                    finished[row_index] = True

        if all(finished):
            break

        next_tensor = torch.tensor(next_tokens, dtype=input_ids.dtype, device=loaded.device).unsqueeze(-1)
        next_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=loaded.device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)
        attention_mask = torch.cat([attention_mask, next_mask], dim=1)

        with torch.inference_mode():
            outputs = loaded.model(
                input_ids=next_tensor,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :].detach().float()

    return [
        truncated_texts[row_index]
        if truncated_texts[row_index] is not None
        else tokenizer.decode(completion_ids[row_index], skip_special_tokens=True).strip()
        for row_index in range(batch_size)
    ]


def _sample_next_token_id(
    logits: torch.Tensor,
    *,
    completion_ids: list[int],
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    generator: torch.Generator | None,
) -> int:
    step_logits = logits.clone()
    if presence_penalty > 0.0 and completion_ids:
        seen_ids = torch.tensor(sorted(set(completion_ids)), dtype=torch.long, device=step_logits.device)
        step_logits[seen_ids] = step_logits[seen_ids] - float(presence_penalty)

    if not do_sample:
        return int(torch.argmax(step_logits).item())

    step_logits = step_logits / max(float(temperature), 1e-5)

    if top_k > 0 and top_k < step_logits.numel():
        top_values, top_indices = torch.topk(step_logits, int(top_k))
        filtered = torch.full_like(step_logits, float("-inf"))
        filtered[top_indices] = top_values
        step_logits = filtered

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(step_logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative_probs > float(top_p)
        if remove_mask.numel() > 1:
            remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False
        sorted_logits[remove_mask] = float("-inf")
        filtered = torch.full_like(step_logits, float("-inf"))
        filtered[sorted_indices] = sorted_logits
        step_logits = filtered

    probs = torch.softmax(step_logits, dim=-1)
    if torch.isnan(probs).any() or float(probs.sum().item()) <= 0.0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def greedy_site_run(
    loaded: LoadedModel,
    prompt: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    injection_site: str,
    inject_layer: int | None = None,
    inject_vector: torch.Tensor | None = None,
    inject_scale: float = 0.0,
    persistent_generated_steps: int = 0,
    stop_strings: list[str] | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    sampling_seed: int | None = None,
    capture_site_states: bool = True,
) -> SiteResult:
    tokenizer = loaded.tokenizer
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_ids = encoded["input_ids"].to(loaded.device)
    attention_mask = encoded["attention_mask"].to(loaded.device)

    completion_ids: list[int] = []
    first_generated_token_id: int | None = None
    site_states: torch.Tensor | None = None
    site_logits: torch.Tensor | None = None
    injection_applied = False
    truncated_completion_text: str | None = None
    generator: torch.Generator | None = None
    if do_sample:
        generator = torch.Generator(device=loaded.device.type)
        if sampling_seed is not None:
            generator.manual_seed(int(sampling_seed))

    prompt_injection = injection_site == "last_prompt" and inject_layer is not None and inject_vector is not None
    if injection_site not in {"last_prompt", "first_generated"}:
        raise ValueError(f"Unsupported injection site: {injection_site}")

    states, logits, past_key_values = _forward_single(
        loaded=loaded,
        input_ids=input_ids,
        attention_mask=attention_mask,
        inject_layer=inject_layer if prompt_injection else None,
        inject_scale=inject_scale if prompt_injection else 0.0,
        inject_vector=inject_vector if prompt_injection else None,
        capture_hidden_states=bool(capture_site_states and injection_site == "last_prompt"),
        use_cache=True,
    )
    if injection_site == "last_prompt":
        site_states = states
        site_logits = logits
        injection_applied = prompt_injection

    current_logits = logits
    eos_token_id = int(tokenizer.eos_token_id or -1)

    for step in range(max(1, int(max_new_tokens))):
        next_token_id = _sample_next_token_id(
            current_logits,
            completion_ids=completion_ids,
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            presence_penalty=float(presence_penalty),
            generator=generator,
        )
        if step == 0:
            first_generated_token_id = next_token_id
        completion_ids.append(next_token_id)

        if stop_strings:
            current_completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            stop_positions = [
                current_completion_text.find(stop_text)
                for stop_text in stop_strings
                if stop_text and current_completion_text.find(stop_text) >= 0
            ]
            if stop_positions:
                truncated_completion_text = current_completion_text[: min(stop_positions)].strip()
                break

        next_tensor = torch.tensor([[next_token_id]], dtype=input_ids.dtype, device=loaded.device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)
        next_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=loaded.device)
        attention_mask = torch.cat([attention_mask, next_mask], dim=1)

        if next_token_id == eos_token_id:
            break

        generated_step_index = step + 1
        apply_generated_injection = False
        if injection_site == "first_generated" and inject_layer is not None and inject_vector is not None:
            if generated_step_index == 1:
                apply_generated_injection = True
            elif persistent_generated_steps > 1 and 1 <= generated_step_index < 1 + int(persistent_generated_steps):
                apply_generated_injection = True

        states, current_logits, past_key_values = _forward_single(
            loaded=loaded,
            input_ids=next_tensor,
            attention_mask=attention_mask,
            inject_layer=inject_layer if apply_generated_injection else None,
            inject_scale=inject_scale if apply_generated_injection else 0.0,
            inject_vector=inject_vector if apply_generated_injection else None,
            capture_hidden_states=bool(capture_site_states and injection_site == "first_generated" and generated_step_index == 1),
            use_cache=True,
            past_key_values=past_key_values,
        )
        if injection_site == "first_generated" and generated_step_index == 1:
            site_states = states
            site_logits = current_logits
            injection_applied = apply_generated_injection

    if site_logits is None:
        site_states, site_logits, _ = _forward_single(
            loaded=loaded,
            input_ids=input_ids,
            attention_mask=attention_mask,
            capture_hidden_states=bool(capture_site_states),
            use_cache=False,
        )

    completion_text = (
        truncated_completion_text
        if truncated_completion_text is not None
        else tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    )
    return SiteResult(
        prompt=prompt,
        site_states=site_states,
        site_logits=site_logits,
        completion_text=completion_text,
        completion_token_ids=completion_ids,
        first_generated_token_id=first_generated_token_id,
        injection_applied=injection_applied,
    )


def js_divergence_from_logits(base_logits: torch.Tensor, inj_logits: torch.Tensor) -> float:
    p = torch.softmax(base_logits.float(), dim=-1)
    q = torch.softmax(inj_logits.float(), dim=-1)
    mean = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p + 1e-12) - torch.log(mean + 1e-12)))
    kl_qm = torch.sum(q * (torch.log(q + 1e-12) - torch.log(mean + 1e-12)))
    return float((0.5 * (kl_pm + kl_qm)).item())


def estimate_axis_vector(
    loaded: LoadedModel,
    axis_name: str,
    layer_index: int,
    token_position: int,
    max_prompt_tokens: int,
    seed: int,
    control: str = "mean_diff",
) -> torch.Tensor:
    positives, negatives = axis_seed_texts(axis_name)
    pos_acts = extract_layer_activations(
        loaded=loaded,
        prompts=positives,
        layer_index=layer_index,
        token_position=token_position,
        max_prompt_tokens=max_prompt_tokens,
        batch_size=4,
    )
    neg_acts = extract_layer_activations(
        loaded=loaded,
        prompts=negatives,
        layer_index=layer_index,
        token_position=token_position,
        max_prompt_tokens=max_prompt_tokens,
        batch_size=4,
    )

    if control == "mean_diff":
        return estimate_mean_difference(pos_acts, neg_acts).vector
    if control == "random_orthogonal":
        return estimate_random_orthogonal_vector(pos_acts, neg_acts, seed=seed).vector
    if control == "label_shuffled":
        rng = np.random.default_rng(seed)
        acts = torch.cat([pos_acts, neg_acts], dim=0)
        idx = rng.permutation(acts.shape[0])
        split = pos_acts.shape[0]
        shuf_pos = acts[idx[:split]]
        shuf_neg = acts[idx[split: split + neg_acts.shape[0]]]
        return estimate_mean_difference(shuf_pos, shuf_neg).vector
    raise ValueError(f"Unsupported control type: {control}")


def estimate_layer_scale(
    loaded: LoadedModel,
    texts: list[str],
    layer_index: int,
    token_position: int,
    max_prompt_tokens: int,
) -> float:
    activations = extract_layer_activations(
        loaded=loaded,
        prompts=texts,
        layer_index=layer_index,
        token_position=token_position,
        max_prompt_tokens=max_prompt_tokens,
        batch_size=4,
    )
    norms = torch.linalg.vector_norm(activations, dim=1)
    return float(norms.mean().item())


def evaluate_condition_metrics(
    baseline: SiteResult,
    injected: SiteResult,
    inject_layer: int,
    recovery_threshold: float,
) -> dict[str, float | int | str]:
    metrics = compute_trajectory_metrics(
        baseline_states=baseline.site_states,
        injected_states=injected.site_states,
        baseline_logits=baseline.site_logits,
        injected_logits=injected.site_logits,
        inject_layer_index=inject_layer,
        recovery_threshold=recovery_threshold,
    )
    drift_total = float(sum(metrics.drift_by_layer[metrics.drift_start_index :]))
    return {
        "peak_displacement": float(metrics.peak_drift_relative),
        "total_downstream_change": drift_total,
        "end_of_pass_distance": float(metrics.end_drift_relative),
        "recovery_fraction": float(metrics.recovery_fraction),
        "next_token_kl": float(metrics.next_token_kl),
        "next_token_js": float(js_divergence_from_logits(baseline.site_logits, injected.site_logits)),
        "drift_start_index": int(metrics.drift_start_index),
        "recovery_latency_layers": int(metrics.recovery_latency_layers),
        "crossed_baseline": int(metrics.crossed_baseline),
        "overshoot_index": float(metrics.overshoot_index),
    }


def build_layer_candidates(
    n_layers: int,
    layer_positions: list[float],
    best_fixed_layer: float | None = None,
) -> list[int]:
    positions = list(layer_positions)
    if best_fixed_layer is not None:
        positions.append(float(best_fixed_layer))
    return resolve_layer_indices(n_layers=n_layers, layer_positions=positions)


def score_against_axis_anchors(axis_name: str, text: str) -> float:
    positives, negatives = axis_seed_texts(axis_name)
    corpus = positives + negatives + [text]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(corpus)
    probe = matrix[-1]
    pos = matrix[: len(positives)]
    neg = matrix[len(positives) : len(positives) + len(negatives)]
    pos_score = float(sklearn_cosine_similarity(probe, pos).mean())
    neg_score = float(sklearn_cosine_similarity(probe, neg).mean())
    return pos_score - neg_score


def select_adaptive_layer(
    loaded: LoadedModel,
    prompt: str,
    layer_vectors: dict[int, torch.Tensor],
    max_prompt_tokens: int,
) -> int:
    tokenizer = loaded.tokenizer
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_ids = encoded["input_ids"].to(loaded.device)
    attention_mask = encoded["attention_mask"].to(loaded.device)
    with torch.inference_mode():
        outputs = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    idx = _select_last_token_index(attention_mask)
    best_layer = min(layer_vectors)
    best_score = 0.0
    for layer_index, vector in layer_vectors.items():
        hidden = outputs.hidden_states[layer_index][0, idx, :].detach().float().cpu()
        denom = float(torch.linalg.vector_norm(hidden).item() * torch.linalg.vector_norm(vector).item())
        if denom <= 1e-12:
            score = 0.0
        else:
            score = float(torch.dot(hidden, vector.float()) / denom)
        if abs(score) > abs(best_score):
            best_layer = layer_index
            best_score = score
    return int(best_layer)
