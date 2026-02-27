from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    model_ids: list[str]
    concept_name: str
    vector_methods: list[str]
    alphas: list[float]
    layer_positions: list[float]
    token_position: int
    estimation_token_position: int
    eval_generation_tokens: int
    max_prompt_tokens: int
    estimation_prompt_count: int
    evaluation_prompt_count: int
    seed: int
    output_root: Path
    model_cache_dir: Path
    dtype: str
    use_gpu: bool
    recovery_threshold: float
    activation_batch_size: int = 4
    trace_batch_size: int = 4
    adaptive_batching: bool = True
    attention_backend: str = "auto"
    enable_tf32: bool = True
    prompt_styles: list[str] = field(default_factory=list)


def _require_key(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise KeyError(f"Missing required config key: {key}")
    return raw[key]


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return RunConfig(
        model_ids=list(_require_key(raw, "model_ids")),
        concept_name=str(_require_key(raw, "concept_name")),
        vector_methods=list(_require_key(raw, "vector_methods")),
        alphas=[float(x) for x in _require_key(raw, "alphas")],
        layer_positions=[float(x) for x in _require_key(raw, "layer_positions")],
        token_position=int(_require_key(raw, "token_position")),
        estimation_token_position=int(raw.get("estimation_token_position", raw["token_position"])),
        eval_generation_tokens=int(_require_key(raw, "eval_generation_tokens")),
        max_prompt_tokens=int(_require_key(raw, "max_prompt_tokens")),
        estimation_prompt_count=int(_require_key(raw, "estimation_prompt_count")),
        evaluation_prompt_count=int(_require_key(raw, "evaluation_prompt_count")),
        seed=int(_require_key(raw, "seed")),
        output_root=Path(str(_require_key(raw, "output_root"))),
        model_cache_dir=Path(str(_require_key(raw, "model_cache_dir"))),
        dtype=str(_require_key(raw, "dtype")),
        use_gpu=bool(_require_key(raw, "use_gpu")),
        recovery_threshold=float(_require_key(raw, "recovery_threshold")),
        activation_batch_size=int(raw.get("activation_batch_size", 4)),
        trace_batch_size=int(raw.get("trace_batch_size", 4)),
        adaptive_batching=bool(raw.get("adaptive_batching", True)),
        attention_backend=str(raw.get("attention_backend", "auto")),
        enable_tf32=bool(raw.get("enable_tf32", True)),
        prompt_styles=[str(x) for x in raw.get("prompt_styles", [])],
    )
