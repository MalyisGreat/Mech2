from __future__ import annotations

import argparse
import os
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
    parser.add_argument(
        "--gpus",
        nargs="*",
        type=int,
        default=None,
        help="Optional GPU ids. Use multiple ids to run model shards in parallel (for example: --gpus 0 1 2 3).",
    )
    return parser.parse_args()


def main() -> None:
    _add_src_to_path()
    from identity_stability.config import load_run_config
    from identity_stability.experiment import run_experiment
    from identity_stability.multi_gpu import run_experiment_multi_gpu

    args = parse_args()
    if args.gpus and len(args.gpus) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
    cfg = load_run_config(args.config)
    if args.models:
        cfg.model_ids = list(args.models)

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.model_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.gpus and len(args.gpus) > 1:
        run_dir = run_experiment_multi_gpu(
            config=cfg,
            gpu_ids=list(args.gpus),
            source_config_path=args.config,
            run_label="multi_gpu_experiment",
        )
    else:
        run_dir = run_experiment(cfg)
    print(f"[done] run directory: {run_dir}")


if __name__ == "__main__":
    main()
