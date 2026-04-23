from __future__ import annotations

import sys
from pathlib import Path

import yaml


MODEL_SIZE_LABELS = {
    "EleutherAI/pythia-70m": "70m",
    "EleutherAI/pythia-70m-deduped": "70m",
    "EleutherAI/pythia-160m": "160m",
    "EleutherAI/pythia-160m-deduped": "160m",
    "EleutherAI/pythia-410m": "410m",
    "EleutherAI/pythia-410m-deduped": "410m",
    "EleutherAI/pythia-1b": "1b",
    "EleutherAI/pythia-1b-deduped": "1b",
    "EleutherAI/pythia-1.4b": "1.4b",
    "EleutherAI/pythia-1.4b-deduped": "1.4b",
    "EleutherAI/pythia-2.8b": "2.8b",
    "EleutherAI/pythia-2.8b-deduped": "2.8b",
    "gpt2": "124m",
    "Qwen/Qwen2.5-0.5B-Instruct": "0.5b",
    "Qwen/Qwen2.5-1.5B-Instruct": "1.5b",
    "Qwen/Qwen3-0.6B": "0.6b",
    "Qwen/Qwen3-1.7B": "1.7b",
    "Qwen/Qwen3.5-0.8B": "0.8b",
    "Qwen/Qwen3.5-4B": "4b",
}


def add_src_to_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def load_yaml_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f))


def ensure_output_dir(config: dict, name: str) -> Path:
    out_dir = Path(config["output_root"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def select_seed_values(config: dict) -> list[int]:
    if "seeds" in config:
        return [int(x) for x in config["seeds"]]
    return [int(config.get("seed", 7))]


def infer_model_family(model_id: str) -> str:
    lc = model_id.lower()
    if lc.startswith("eleutherai/pythia-"):
        return "pythia"
    if lc.startswith("qwen/qwen2.5-"):
        return "qwen2.5"
    if lc.startswith("qwen/qwen3.5-"):
        return "qwen3.5"
    if lc.startswith("qwen/qwen3-"):
        return "qwen3"
    if lc.startswith("gpt2"):
        return "gpt2"
    return "unknown"


def infer_model_size_label(model_id: str) -> str:
    return MODEL_SIZE_LABELS.get(model_id, model_id.split("/")[-1])


def resolve_identity_prompt_template(config: dict, default: str = "chat") -> str:
    return str(config.get("identity_prompt_template", default))


def resolve_identity_stop_strings(config: dict, *, default_auto: bool = True) -> list[str] | None:
    raw = config.get("identity_stop_strings", "auto" if default_auto else None)
    if raw is None:
        return None
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"", "none", "off", "false"}:
            return None
        if normalized == "auto":
            return ["AUTO"]
        return [raw]
    if isinstance(raw, (list, tuple)):
        values = [str(item) for item in raw if str(item).strip()]
        return values or None
    return None
