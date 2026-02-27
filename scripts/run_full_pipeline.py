from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download models and run full experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token.",
    )
    return parser.parse_args()


def main() -> None:
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    _add_src_to_path()
    from identity_stability.config import load_run_config
    from identity_stability.experiment import run_experiment

    args = parse_args()
    cfg = load_run_config(args.config)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.model_cache_dir.mkdir(parents=True, exist_ok=True)

    for model_id in cfg.model_ids:
        print(f"[pipeline] downloading {model_id}")
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(cfg.model_cache_dir),
            token=args.token,
        )
        print(f"[pipeline] downloaded {model_id}")

    run_dir = run_experiment(cfg)
    print(f"[pipeline] complete, run directory: {run_dir}")


if __name__ == "__main__":
    main()
