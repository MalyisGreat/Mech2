from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONCEPTS = [
    "morality",
    "constructiveness",
    "formality",
    "skepticism",
]

DONE_PREFIX = "[done] run directory:"


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


def _to_jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _parse_run_dir(stdout_text: str) -> Path | None:
    for line in stdout_text.splitlines():
        s = line.strip()
        if s.startswith(DONE_PREFIX):
            return Path(s[len(DONE_PREFIX) :].strip())
    return None


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
    parser.add_argument(
        "--gpus",
        nargs="*",
        type=int,
        default=None,
        help="Optional GPU ids. Use multiple ids to shard models per job in parallel.",
    )
    parser.add_argument(
        "--job-parallelism",
        type=int,
        default=1,
        help=(
            "Number of add-on jobs to run concurrently. "
            "When >1, jobs are pinned one-per-GPU and executed via single-GPU workers."
        ),
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

    if args.job_parallelism < 1:
        raise ValueError("--job-parallelism must be >= 1")
    if args.job_parallelism > 1 and (not args.gpus or len(args.gpus) < 2):
        raise ValueError("--job-parallelism > 1 requires at least 2 GPU ids via --gpus")

    suite_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = cfg.output_root / f"{args.suite_name}_{suite_stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []
    failures: list[dict[str, str | int]] = []
    total_jobs = len(args.token_positions) * len(args.concepts) * len(args.seeds)
    jobs: list[dict[str, int | str]] = []
    for token_position in args.token_positions:
        for concept in args.concepts:
            for seed in args.seeds:
                jobs.append(
                    {
                        "concept_name": str(concept),
                        "seed": int(seed),
                        "token_position": int(token_position),
                    }
                )

    if args.job_parallelism > 1:
        print(
            f"[addon] scheduling mode=parallel jobs={total_jobs} "
            f"parallelism={min(args.job_parallelism, len(args.gpus or []))} gpus={args.gpus}"
        )
        # Parallel job mode: one GPU per job, multiple jobs in flight.
        repo_root = Path(__file__).resolve().parents[1]
        config_dir = suite_dir / "job_configs"
        logs_dir = suite_dir / "job_logs"
        config_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        slot_gpus = list(args.gpus[: min(args.job_parallelism, len(args.gpus))])
        available_gpus = list(slot_gpus)
        pending_idx = 0
        completed_jobs = 0
        active: list[dict[str, Any]] = []

        while pending_idx < len(jobs) or active:
            while pending_idx < len(jobs) and available_gpus:
                gpu_id = int(available_gpus.pop(0))
                job = jobs[pending_idx]
                job_index = pending_idx + 1
                pending_idx += 1

                concept = str(job["concept_name"])
                seed = int(job["seed"])
                token_position = int(job["token_position"])
                pct = (100.0 * completed_jobs / total_jobs) if total_jobs else 100.0
                print(f"[addon] overall-progress {completed_jobs}/{total_jobs} ({pct:.2f}%)")
                print(
                    "[addon] running "
                    f"job={job_index}/{total_jobs} concept={concept} seed={seed} "
                    f"token_position={token_position} gpu={gpu_id}"
                )

                run_cfg = replace(
                    cfg,
                    concept_name=concept,
                    seed=seed,
                    token_position=token_position,
                )
                job_tag = f"job_{job_index:03d}_{concept}_s{seed}_t{token_position}_g{gpu_id}"
                job_cfg_path = config_dir / f"{job_tag}.yaml"
                with job_cfg_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(_to_jsonable(asdict(run_cfg)), f, sort_keys=False)

                stdout_path = logs_dir / f"{job_tag}.stdout.log"
                stderr_path = logs_dir / f"{job_tag}.stderr.log"
                stdout_f = stdout_path.open("w", encoding="utf-8")
                stderr_f = stderr_path.open("w", encoding="utf-8")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                cmd = [
                    sys.executable,
                    str(repo_root / "scripts" / "run_experiment.py"),
                    "--config",
                    str(job_cfg_path),
                ]
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    env=env,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    text=True,
                )
                active.append(
                    {
                        "proc": proc,
                        "gpu_id": gpu_id,
                        "job_index": int(job_index),
                        "concept_name": concept,
                        "seed": seed,
                        "token_position": token_position,
                        "stdout_path": stdout_path,
                        "stderr_path": stderr_path,
                        "stdout_f": stdout_f,
                        "stderr_f": stderr_f,
                        "started_at": time.time(),
                    }
                )

            if not active:
                time.sleep(0.2)
                continue

            time.sleep(1.0)
            next_active: list[dict[str, Any]] = []
            for task in active:
                proc = task["proc"]
                ret = proc.poll()
                if ret is None:
                    next_active.append(task)
                    continue

                task["stdout_f"].close()
                task["stderr_f"].close()
                gpu_id = int(task["gpu_id"])
                available_gpus.append(gpu_id)

                stdout_text = task["stdout_path"].read_text(encoding="utf-8", errors="replace")
                run_dir = _parse_run_dir(stdout_text)
                concept = str(task["concept_name"])
                seed = int(task["seed"])
                token_position = int(task["token_position"])
                job_index = int(task["job_index"])

                if ret == 0 and run_dir is not None:
                    manifest_rows.append(
                        {
                            "concept_name": concept,
                            "seed": seed,
                            "token_position": token_position,
                            "run_dir": str(run_dir),
                            "gpu_id": gpu_id,
                        }
                    )
                    print(
                        "[addon] completed "
                        f"job={job_index}/{total_jobs} concept={concept} seed={seed} "
                        f"token_position={token_position} gpu={gpu_id} -> {run_dir}"
                    )
                else:
                    failures.append(
                        {
                            "concept_name": concept,
                            "seed": seed,
                            "token_position": token_position,
                            "gpu_id": gpu_id,
                            "error": f"worker return code={ret}; see {task['stderr_path']}",
                        }
                    )
                    print(
                        "[addon] failed "
                        f"job={job_index}/{total_jobs} concept={concept} seed={seed} "
                        f"token_position={token_position} gpu={gpu_id}; rc={ret}"
                    )

                completed_jobs += 1
                pct = (100.0 * completed_jobs / total_jobs) if total_jobs else 100.0
                print(f"[addon] overall-progress {completed_jobs}/{total_jobs} ({pct:.2f}%)")
            active = next_active
    else:
        completed_jobs = 0
        for job in jobs:
            concept = str(job["concept_name"])
            seed = int(job["seed"])
            token_position = int(job["token_position"])
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
            try:
                if args.gpus and len(args.gpus) > 1:
                    run_dir = run_experiment_multi_gpu(
                        config=run_cfg,
                        gpu_ids=list(args.gpus),
                        source_config_path=args.config,
                        run_label=f"addon_mgpu_{concept}_s{seed}_t{token_position}",
                    )
                else:
                    run_dir = run_experiment(run_cfg)
                manifest_rows.append(
                    {
                        "concept_name": concept,
                        "seed": seed,
                        "token_position": token_position,
                        "run_dir": str(run_dir),
                        "gpu_id": "",
                    }
                )
                print(
                    "[addon] completed "
                    f"job={job_index}/{total_jobs} concept={concept} seed={seed} "
                    f"token_position={token_position} -> {run_dir}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "concept_name": concept,
                        "seed": seed,
                        "token_position": token_position,
                        "error": str(exc),
                    }
                )
                print(
                    "[addon] failed "
                    f"job={job_index}/{total_jobs} concept={concept} seed={seed} "
                    f"token_position={token_position}: {exc}"
                )
            completed_jobs += 1
            pct = (100.0 * completed_jobs / total_jobs) if total_jobs else 100.0
            print(f"[addon] overall-progress {completed_jobs}/{total_jobs} ({pct:.2f}%)")

    manifest_csv = suite_dir / "suite_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["concept_name", "seed", "token_position", "run_dir", "gpu_id"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (suite_dir / "suite_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2)
    with (suite_dir / "suite_failures.json").open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    print(f"[addon] manifest csv: {manifest_csv}")
    print(f"[addon] manifest json: {suite_dir / 'suite_manifest.json'}")
    print(f"[addon] failures json: {suite_dir / 'suite_failures.json'}")
    if failures:
        raise RuntimeError(f"Addon suite completed with {len(failures)} failed job(s).")


if __name__ == "__main__":
    main()
