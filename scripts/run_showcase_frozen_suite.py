from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen paper showcase suite (3 concepts x 3 prompts x 3 sizes).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/showcase_frozen_suite.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    _add_src_to_path()
    from identity_stability.config import load_run_config
    from identity_stability.experiment import run_experiment

    args = parse_args()
    cfg = load_run_config(args.config)
    concepts = ["politeness", "empathy", "skepticism"]

    run_dirs = []
    for concept in concepts:
        cfg.concept_name = concept
        run_dir = run_experiment(cfg)
        run_dirs.append(str(run_dir))
        print(f"[showcase] completed {concept}: {run_dir}")

    print("[showcase] run dirs:")
    for rd in run_dirs:
        print(rd)


if __name__ == "__main__":
    main()
