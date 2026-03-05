from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LoadedModel:
    model_id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    torch_dtype: torch.dtype
    n_layers: int
    hidden_size: int


def resolve_torch_dtype(name: str, device: torch.device) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if name == "bfloat16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype setting: {name}")


def resolve_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_attention_candidates(attention_backend: str, device: torch.device) -> list[str | None]:
    backend = attention_backend.lower().strip()
    if backend == "auto":
        if device.type == "cuda":
            return ["sdpa", None]
        return [None]
    if backend in {"none", "default", "eager"}:
        return [None]
    if backend in {"sdpa", "flash_attention_2"}:
        return [backend]
    raise ValueError(
        "Unsupported attention_backend: "
        f"{attention_backend}. Use one of auto, sdpa, flash_attention_2, default."
    )


def _get_layer_stack(model: PreTrainedModel) -> Iterable[torch.nn.Module]:
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture for residual intervention.")


def load_model(
    model_id: str,
    cache_dir: str | Path,
    dtype_name: str,
    use_gpu: bool,
    attention_backend: str = "auto",
) -> LoadedModel:
    device = resolve_device(use_gpu=use_gpu)
    torch_dtype = resolve_torch_dtype(dtype_name, device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding avoids generation-time artifacts for decoder-only LMs in batched mode.
    tokenizer.padding_side = "left"

    model = None
    last_exc: Exception | None = None
    for attn_impl in _resolve_attention_candidates(attention_backend, device):
        try:
            load_kwargs: dict[str, object] = {
                "cache_dir": str(cache_dir),
                "dtype": torch_dtype,
                "low_cpu_mem_usage": True,
            }
            if attn_impl is not None:
                load_kwargs["attn_implementation"] = attn_impl
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **load_kwargs,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attn_impl is not None:
                print(f"[modeling] falling back from attention backend '{attn_impl}' for {model_id}: {exc}")
            continue
    if model is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Failed to load model {model_id}")

    model.to(device)
    model.eval()

    layers = _get_layer_stack(model)
    n_layers = len(layers)

    config = model.config
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer model hidden size from config.")

    return LoadedModel(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch_dtype,
        n_layers=n_layers,
        hidden_size=int(hidden_size),
    )


def get_layer_modules(model: PreTrainedModel) -> list[torch.nn.Module]:
    return list(_get_layer_stack(model))


def clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
