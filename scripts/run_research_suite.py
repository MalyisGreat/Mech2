from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-concept multi-seed research suite.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base YAML config path.",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        required=True,
        help="Concept names, e.g. politeness empathy confidence",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Seed list.",
    )
    parser.add_argument(
        "--suite-name",
        default="research_suite",
        help="Logical suite name for output folder.",
    )
    parser.add_argument(
        "--gpus",
        nargs="*",
        type=int,
        default=None,
        help="Optional GPU ids. Use multiple ids to shard models per job in parallel.",
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
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.model_cache_dir.mkdir(parents=True, exist_ok=True)

    suite_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = cfg.output_root / f"{args.suite_name}_{suite_stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []
    total_jobs = len(args.concepts) * len(args.seeds)
    completed_jobs = 0
    for concept in args.concepts:
        for seed in args.seeds:
            job_index = completed_jobs + 1
            pct = (100.0 * completed_jobs / total_jobs) if total_jobs else 100.0
            print(f"[suite] overall-progress {completed_jobs}/{total_jobs} ({pct:.2f}%)")
            print(f"[suite] running job={job_index}/{total_jobs} concept={concept} seed={seed}")
            run_cfg = replace(cfg, concept_name=concept, seed=seed)
            if args.gpus and len(args.gpus) > 1:
                run_dir = run_experiment_multi_gpu(
                    config=run_cfg,
                    gpu_ids=list(args.gpus),
                    source_config_path=args.config,
                    run_label=f"suite_mgpu_{concept}_s{seed}",
                )
            else:
                run_dir = run_experiment(run_cfg)
            manifest_rows.append(
                {
                    "concept_name": concept,
                    "seed": seed,
                    "run_dir": str(run_dir),
                }
            )
            completed_jobs += 1
            pct = (100.0 * completed_jobs / total_jobs) if total_jobs else 100.0
            print(f"[suite] completed job={job_index}/{total_jobs} concept={concept} seed={seed} -> {run_dir}")
            print(f"[suite] overall-progress {completed_jobs}/{total_jobs} ({pct:.2f}%)")

    manifest_csv = suite_dir / "suite_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_name", "seed", "run_dir"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (suite_dir / "suite_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2)

    print(f"[suite] manifest csv: {manifest_csv}")
    print(f"[suite] manifest json: {suite_dir / 'suite_manifest.json'}")


if __name__ == "__main__":
    main()
