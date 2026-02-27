from __future__ import annotations

from dataclasses import dataclass

import torch

from .modeling import LoadedModel, get_layer_modules


@dataclass
class TraceResult:
    prompt: str
    token_indices: list[int]
    per_layer_states: torch.Tensor
    next_token_logits: torch.Tensor
    generated_text: str


def resolve_layer_indices(n_layers: int, layer_positions: list[float]) -> list[int]:
    idxs = []
    for pos in layer_positions:
        pos = float(pos)
        if pos < 0.0 or pos > 1.0:
            raise ValueError(f"Layer position must be in [0, 1], got {pos}")
        idx = int(round(pos * (n_layers - 1)))
        idxs.append(max(0, min(n_layers - 1, idx)))
    return sorted(set(idxs))


def _select_indices(attention_mask: torch.Tensor, token_position: int) -> torch.Tensor:
    batch, seq_len = attention_mask.shape
    if token_position >= 0:
        idx = torch.full((batch,), token_position, device=attention_mask.device, dtype=torch.long)
    else:
        lengths = attention_mask.sum(dim=1) - 1
        idx = lengths + token_position + 1
    return torch.clamp(idx, 0, seq_len - 1)


def _select_hidden_per_layer(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    token_position: int,
) -> torch.Tensor:
    selected_layers = []
    token_idx = _select_indices(attention_mask, token_position)
    for hs in hidden_states:
        batch, _, hidden = hs.shape
        gather_idx = token_idx.view(batch, 1, 1).expand(batch, 1, hidden)
        selected = torch.gather(hs, 1, gather_idx).squeeze(1)
        selected_layers.append(selected)
    stacked = torch.stack(selected_layers, dim=1)
    return stacked


def run_trace(
    loaded: LoadedModel,
    prompt: str,
    max_prompt_tokens: int,
    token_position: int,
    generate_tokens: int,
    inject_layer: int | None = None,
    inject_vector: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> TraceResult:
    return run_trace_batch(
        loaded=loaded,
        prompts=[prompt],
        max_prompt_tokens=max_prompt_tokens,
        token_position=token_position,
        generate_tokens=generate_tokens,
        inject_layer=inject_layer,
        inject_vector=inject_vector,
        alpha=alpha,
    )[0]


def run_trace_batch(
    loaded: LoadedModel,
    prompts: list[str],
    max_prompt_tokens: int,
    token_position: int,
    generate_tokens: int,
    inject_layer: int | None = None,
    inject_vector: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> list[TraceResult]:
    if not prompts:
        return []

    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    attention_mask = encoded["attention_mask"]
    token_idx = _select_indices(attention_mask, token_position)

    layers = get_layer_modules(model)
    handle = None
    if inject_layer is not None:
        if inject_vector is None:
            raise ValueError("inject_vector is required when inject_layer is set.")

        vector = inject_vector.to(device=device, dtype=model.dtype)

        def pre_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            hidden_states = inputs[0]
            hidden_states = hidden_states.clone()
            for b in range(hidden_states.shape[0]):
                idx = int(token_idx[b].item())
                hidden_states[b, idx, :] = hidden_states[b, idx, :] + (alpha * vector)
            return (hidden_states,) + tuple(inputs[1:])

        handle = layers[inject_layer].register_forward_pre_hook(pre_hook)

    try:
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            per_layer = _select_hidden_per_layer(
                outputs.hidden_states,
                attention_mask=attention_mask,
                token_position=token_position,
            ).detach().float().cpu()
            logits = outputs.logits[:, -1, :].detach().float().cpu()

            top_ids = torch.argmax(logits, dim=-1).tolist()
            generated_texts = []
            for b, top_id in enumerate(top_ids):
                continuation = tokenizer.decode([int(top_id)], skip_special_tokens=True)
                generated_texts.append(prompts[b] + continuation)
    finally:
        if handle is not None:
            handle.remove()

    results: list[TraceResult] = []
    idxs = token_idx.detach().cpu().tolist()
    for b, prompt in enumerate(prompts):
        results.append(
            TraceResult(
                prompt=prompt,
                token_indices=[int(idxs[b])],
                per_layer_states=per_layer[b],
                next_token_logits=logits[b],
                generated_text=generated_texts[b],
            )
        )
    return results
