from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .modeling import LoadedModel, get_layer_modules


@dataclass
class TraceResult:
    prompt: str
    token_indices: list[int]
    per_layer_states: torch.Tensor
    next_token_logits: torch.Tensor
    generated_text: str
    prompt_token_count: int = 0
    generated_token_count: int = 0
    layer_topk_tokens: list[dict[str, Any]] | None = None


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
    lengths = attention_mask.sum(dim=1)
    left_pad = seq_len - lengths
    if token_position >= 0:
        idx = left_pad + token_position
    else:
        idx = left_pad + lengths + token_position
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


def _get_final_norm_module(model: torch.nn.Module) -> torch.nn.Module | None:
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    return None


def _compute_layer_topk_tokens(
    model: torch.nn.Module,
    tokenizer,
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    token_position: int,
    topk_tokens: int,
    prompt_limit: int,
) -> list[list[dict[str, Any]] | None]:
    batch = attention_mask.shape[0]
    out: list[list[dict[str, Any]] | None] = [None for _ in range(batch)]
    if topk_tokens <= 0 or prompt_limit <= 0:
        return out
    output_emb = model.get_output_embeddings()
    if output_emb is None:
        return out

    topk = int(max(1, topk_tokens))
    keep = int(min(batch, max(1, prompt_limit)))
    keep_indices = torch.arange(keep, device=attention_mask.device, dtype=torch.long)
    final_norm = _get_final_norm_module(model)
    token_idx = _select_indices(attention_mask, token_position)

    for b in range(keep):
        out[b] = []

    for hs_idx, hs in enumerate(hidden_states):
        _, _, hidden = hs.shape
        gather_idx = token_idx.view(batch, 1, 1).expand(batch, 1, hidden)
        selected = torch.gather(hs, 1, gather_idx).squeeze(1)
        selected = selected.index_select(0, keep_indices)
        if final_norm is not None:
            selected = final_norm(selected)
        logits = output_emb(selected)
        vals, idxs = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)
        vals = vals.detach().float().cpu()
        idxs = idxs.detach().cpu()
        for local_b in range(keep):
            token_ids = idxs[local_b].tolist()
            token_texts = tokenizer.convert_ids_to_tokens(token_ids)
            tokens = []
            for j, tok_id in enumerate(token_ids):
                tok_text = token_texts[j]
                tokens.append(
                    {
                        "rank": int(j + 1),
                        "token_id": int(tok_id),
                        "token": str(tok_text),
                        "logit": float(vals[local_b, j].item()),
                    }
                )
            layer_record = {
                "hidden_state_index": int(hs_idx),
                "transformer_layer_index": int(hs_idx - 1),
                "tokens": tokens,
            }
            if out[local_b] is not None:
                out[local_b].append(layer_record)
    return out


def run_trace(
    loaded: LoadedModel,
    prompt: str,
    max_prompt_tokens: int,
    token_position: int,
    generate_tokens: int,
    inject_layer: int | None = None,
    inject_vector: torch.Tensor | None = None,
    alpha: float = 0.0,
    layer_topk_tokens: int = 0,
    layer_topk_prompt_limit: int = 1,
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
        layer_topk_tokens=layer_topk_tokens,
        layer_topk_prompt_limit=layer_topk_prompt_limit,
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
    layer_topk_tokens: int = 0,
    layer_topk_prompt_limit: int = 1,
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
    prompt_token_counts = attention_mask.sum(dim=1).detach().cpu().tolist()
    token_idx = _select_indices(attention_mask, token_position)
    last_prompt_idx = _select_indices(attention_mask, -1)

    layers = get_layer_modules(model)
    handle = None
    if inject_layer is not None:
        if inject_vector is None:
            raise ValueError("inject_vector is required when inject_layer is set.")

        vector = inject_vector.to(device=device, dtype=model.dtype)
        inject_state = {"enabled": True}

        def pre_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            if not inject_state["enabled"]:
                return inputs
            hidden_states = inputs[0]
            hidden_states = hidden_states.clone()
            seq_len = hidden_states.shape[1]
            for b in range(hidden_states.shape[0]):
                idx = int(token_idx[b].item())
                idx = max(0, min(seq_len - 1, idx))
                hidden_states[b, idx, :] = hidden_states[b, idx, :] + (alpha * vector)
            inject_state["enabled"] = False
            return (hidden_states,) + tuple(inputs[1:])

        handle = layers[inject_layer].register_forward_pre_hook(pre_hook)

    try:
        with torch.inference_mode():
            if handle is not None:
                inject_state["enabled"] = True
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
            per_layer_topk = _compute_layer_topk_tokens(
                model=model,
                tokenizer=tokenizer,
                hidden_states=outputs.hidden_states,
                attention_mask=attention_mask,
                token_position=token_position,
                topk_tokens=layer_topk_tokens,
                prompt_limit=layer_topk_prompt_limit,
            )
            logits_full = outputs.logits
            batch_idx = torch.arange(logits_full.shape[0], device=logits_full.device)
            logits = logits_full[batch_idx, last_prompt_idx, :].detach().float().cpu()

            if generate_tokens > 0:
                if handle is not None:
                    inject_state["enabled"] = True
                generated_ids = model.generate(
                    **encoded,
                    max_new_tokens=int(generate_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                generated_steps = int(max(0, generated_ids.shape[1] - encoded["input_ids"].shape[1]))
                generated_token_counts = [generated_steps for _ in prompts]
            else:
                generated_texts = list(prompts)
                generated_token_counts = [0 for _ in prompts]
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
                prompt_token_count=int(prompt_token_counts[b]),
                generated_token_count=int(generated_token_counts[b]),
                layer_topk_tokens=per_layer_topk[b],
            )
        )
    return results
