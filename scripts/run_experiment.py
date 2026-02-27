from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run identity-stability experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional override for model ids.",
    )
    return parser.parse_args()


def main() -> None:
    _add_src_to_path()
    from identity_stability.config import load_run_config
    from identity_stability.experiment import run_experiment

    args = parse_args()
    cfg = load_run_config(args.config)
    if args.models:
        cfg.model_ids = list(args.models)

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.model_cache_dir.mkdir(parents=True, exist_ok=True)

    run_dir = run_experiment(cfg)
    print(f"[done] run directory: {run_dir}")


if __name__ == "__main__":
    main()
