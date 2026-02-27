from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path


DEFAULT_CONCEPTS = [
    "morality",
    "constructiveness",
    "formality",
    "skepticism",
]


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prior-findings add-on suite (thresholds, prompt styles, token positions)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prior_findings_addon.yaml"),
        help="Base YAML config path.",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=DEFAULT_CONCEPTS,
        help="Concept names to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Seed list.",
    )
    parser.add_argument(
        "--token-positions",
        nargs="+",
        type=int,
        default=[-1, 0],
        help="Token positions for susceptibility checks (for example -1 0).",
    )
    parser.add_argument(
        "--suite-name",
        default="prior_findings_addon",
        help="Logical suite name for output folder.",
    )
    return parser.parse_args()


def main() -> None:
    _add_src_to_path()
    from identity_stability.config import load_run_config
    from identity_stability.experiment import run_experiment

    args = parse_args()
    cfg = load_run_config(args.config)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.model_cache_dir.mkdir(parents=True, exist_ok=True)

    suite_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = cfg.output_root / f"{args.suite_name}_{suite_stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []
    total_jobs = len(args.token_positions) * len(args.concepts) * len(args.seeds)
    completed_jobs = 0
    for token_position in args.token_positions:
        for concept in args.concepts:
            for seed in args.seeds:
                job_index = completed_jobs + 1
                pct = (100.0 * completed_jobs / total_jobs) if total_jobs else 100.0
                print(f"[addon] overall-progress {completed_jobs}/{total_jobs} ({pct:.2f}%)")
                print(
                    "[addon] running "
                    f"job={job_index}/{total_jobs} concept={concept} seed={seed} "
                    f"token_position={token_position}"
                )
                run_cfg = replace(
                    cfg,
                    concept_name=concept,
                    seed=seed,
                    token_position=token_position,
                )
                run_dir = run_experiment(run_cfg)
                manifest_rows.append(
                    {
                        "concept_name": concept,
                        "seed": seed,
                        "token_position": token_position,
                        "run_dir": str(run_dir),
                    }
                )
                print(
                    "[addon] completed "
                    f"job={job_index}/{total_jobs} concept={concept} seed={seed} "
                    f"token_position={token_position} -> {run_dir}"
                )
                completed_jobs += 1
                pct = (100.0 * completed_jobs / total_jobs) if total_jobs else 100.0
                print(f"[addon] overall-progress {completed_jobs}/{total_jobs} ({pct:.2f}%)")

    manifest_csv = suite_dir / "suite_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["concept_name", "seed", "token_position", "run_dir"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (suite_dir / "suite_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2)

    print(f"[addon] manifest csv: {manifest_csv}")
    print(f"[addon] manifest json: {suite_dir / 'suite_manifest.json'}")


if __name__ == "__main__":
    main()
