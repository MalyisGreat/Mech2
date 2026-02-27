from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

from huggingface_hub import snapshot_download
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Hugging Face models for experiments.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--models",
        nargs="+",
        help="One or more model ids, e.g. EleutherAI/pythia-160m",
    )
    source.add_argument(
        "--config",
        type=Path,
        help="YAML config path containing a top-level `model_ids` list.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Cache directory for downloaded models.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token if needed for gated models.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel download workers. Use >1 for faster model prefetch when bandwidth allows.",
    )
    return parser.parse_args()


def _load_model_ids_from_config(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "model_ids" not in raw:
        raise ValueError(f"Config {config_path} must contain top-level `model_ids`.")
    model_ids = raw["model_ids"]
    if not isinstance(model_ids, list) or not model_ids:
        raise ValueError(f"Config {config_path} has invalid `model_ids`; expected non-empty list.")
    return [str(x) for x in model_ids]


def main() -> None:
    args = parse_args()
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    model_ids = args.models
    if args.config is not None:
        model_ids = _load_model_ids_from_config(args.config)
        print(f"[download] loaded {len(model_ids)} models from {args.config}")

    workers = max(1, int(args.workers))
    if workers == 1:
        for model_id in model_ids:
            print(f"[download] {model_id}")
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(args.cache_dir),
                token=args.token,
            )
            print(f"[download] complete: {model_id} -> {local_path}")
        return

    print(f"[download] parallel mode with workers={workers}")
    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                snapshot_download,
                repo_id=model_id,
                cache_dir=str(args.cache_dir),
                token=args.token,
            ): model_id
            for model_id in model_ids
        }
        for future in as_completed(futures):
            model_id = futures[future]
            try:
                local_path = future.result()
                print(f"[download] complete: {model_id} -> {local_path}")
            except Exception as exc:  # noqa: BLE001
                errors.append((model_id, str(exc)))
                print(f"[download] failed: {model_id} -> {exc}")

    if errors:
        detail = "; ".join(f"{mid}: {err}" for mid, err in errors)
        raise RuntimeError(f"One or more downloads failed: {detail}")


if __name__ == "__main__":
    main()
