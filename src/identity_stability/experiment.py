from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import os
import platform
import random
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .intervention import TraceResult, resolve_layer_indices, run_trace_batch
from .metrics import compute_trajectory_metrics
from .modeling import clear_cuda, load_model
from .prompt_bank import build_prompt_set, get_concept_words
from .telemetry import GpuTelemetrySampler, summarize_gpu_telemetry_csv
from .vectors import (
    extract_layer_activations,
    estimate_concept_vectors,
    estimate_word_centroid_vector,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slug_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def _to_jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _safe_cmd(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    out = proc.stdout.strip()
    return out if out else None


def _collect_git_info(repo_root: Path) -> dict[str, Any]:
    head = _safe_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
    branch = _safe_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _safe_cmd(["git", "status", "--porcelain"], cwd=repo_root)
    return {
        "repo_root": str(repo_root),
        "commit": head,
        "branch": branch,
        "dirty": bool(status),
    }


def _collect_runtime_info() -> dict[str, Any]:
    package_names = [
        "torch",
        "transformers",
        "huggingface_hub",
        "numpy",
        "pandas",
        "pyyaml",
        "scikit-learn",
    ]
    package_versions: dict[str, str | None] = {}
    for pkg in package_names:
        try:
            package_versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            package_versions[pkg] = None

    cuda_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cuda_devices.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "multiprocessor_count": int(props.multi_processor_count),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": int(os.cpu_count() or 0),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": (
            int(torch.get_num_interop_threads()) if hasattr(torch, "get_num_interop_threads") else None
        ),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_devices": cuda_devices,
        "package_versions": package_versions,
    }


def _configure_cuda_runtime(enable_tf32: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
    torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _configure_cpu_runtime(
    cpu_threads_per_worker: int,
    cpu_interop_threads: int,
    tokenizers_parallelism: bool,
) -> None:
    if cpu_threads_per_worker > 0:
        threads = int(cpu_threads_per_worker)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
        try:
            torch.set_num_threads(threads)
        except Exception:
            pass

    if cpu_interop_threads > 0:
        try:
            torch.set_num_interop_threads(int(cpu_interop_threads))
        except Exception:
            pass

    os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizers_parallelism else "false"


def _is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda error: out of memory" in msg)


def _trace_prompts_batched(
    loaded,
    prompts: list[str],
    max_prompt_tokens: int,
    token_position: int,
    generate_tokens: int,
    batch_size: int,
    adaptive_batching: bool,
    inject_layer: int | None = None,
    inject_vector: torch.Tensor | None = None,
    alpha: float = 0.0,
    layer_topk_tokens: int = 0,
    layer_topk_prompt_limit: int = 1,
) -> list[TraceResult]:
    if not prompts:
        return []

    results: list[TraceResult] = []
    i = 0
    current_bs = max(1, int(batch_size))
    while i < len(prompts):
        chunk = prompts[i : i + current_bs]
        try:
            out = run_trace_batch(
                loaded=loaded,
                prompts=chunk,
                max_prompt_tokens=max_prompt_tokens,
                token_position=token_position,
                generate_tokens=generate_tokens,
                inject_layer=inject_layer,
                inject_vector=inject_vector,
                alpha=alpha,
                layer_topk_tokens=layer_topk_tokens,
                layer_topk_prompt_limit=layer_topk_prompt_limit,
            )
            results.extend(out)
            i += len(chunk)
        except RuntimeError as exc:
            if (not adaptive_batching) or (not _is_oom_error(exc)):
                raise
            if current_bs <= 1:
                raise
            current_bs = max(1, current_bs // 2)
            print(f"[batch] OOM; reducing trace batch size to {current_bs}")
            clear_cuda()
    return results


def _trace_token_totals(traces: list[TraceResult]) -> tuple[int, int]:
    prompt_tokens = int(sum(int(t.prompt_token_count) for t in traces))
    generated_tokens = int(sum(int(t.generated_token_count) for t in traces))
    return prompt_tokens, generated_tokens


def _init_compute_counters(enabled: bool) -> dict[str, int | float | bool]:
    return {
        "enabled": bool(enabled),
        "trace_calls": 0,
        "trace_prompt_count": 0,
        "trace_prompt_token_evals": 0,
        "trace_generate_prefill_token_evals": 0,
        "trace_decode_token_evals": 0,
        "activation_prompt_count": 0,
        "activation_prompt_token_evals": 0,
        "total_token_evals": 0,
        "approx_forward_flops_2n": 0.0,
        "approx_forward_flops_6n": 0.0,
    }


def _finalize_compute_counters(
    counters: dict[str, int | float | bool],
    parameter_count: int,
    elapsed_seconds: float,
) -> dict[str, int | float | bool]:
    if not bool(counters.get("enabled", False)):
        return counters

    total_token_evals = int(
        int(counters["trace_prompt_token_evals"])
        + int(counters["trace_generate_prefill_token_evals"])
        + int(counters["trace_decode_token_evals"])
        + int(counters["activation_prompt_token_evals"])
    )
    counters["total_token_evals"] = total_token_evals
    counters["approx_forward_flops_2n"] = float(2.0 * float(parameter_count) * float(total_token_evals))
    counters["approx_forward_flops_6n"] = float(6.0 * float(parameter_count) * float(total_token_evals))
    counters["token_evals_per_second"] = (
        float(total_token_evals) / max(1e-9, float(elapsed_seconds))
        if elapsed_seconds > 0
        else None
    )
    return counters


def run_experiment(config: RunConfig) -> Path:
    _set_seed(config.seed)
    _configure_cpu_runtime(
        cpu_threads_per_worker=config.cpu_threads_per_worker,
        cpu_interop_threads=config.cpu_interop_threads,
        tokenizers_parallelism=config.tokenizers_parallelism,
    )
    _configure_cuda_runtime(config.enable_tf32)
    run_started_at = time.time()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    run_dir = config.output_root / run_stamp
    if run_dir.exists():
        run_dir = config.output_root / f"{run_stamp}_{run_id[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)

    repo_root = Path(__file__).resolve().parents[2]
    provenance = {
        "run_id": run_id,
        "run_stamp": run_stamp,
        "started_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "runtime": _collect_runtime_info(),
        "git": _collect_git_info(repo_root=repo_root),
    }
    with (run_dir / "run_provenance.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(provenance), f, indent=2)

    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(asdict(config)), f, indent=2)

    telemetry_summary: dict[str, Any] = {
        "enabled": False,
        "csv_path": str(run_dir / "gpu_telemetry.csv"),
        "interval_sec": float(config.gpu_telemetry_interval_sec),
        "sample_rows": 0,
        "gpu_count": 0,
        "overall": {},
        "per_gpu": [],
        "errors": [],
    }
    telemetry_sampler: GpuTelemetrySampler | None = None
    if config.enable_gpu_telemetry:
        telemetry_sampler = GpuTelemetrySampler(
            csv_path=run_dir / "gpu_telemetry.csv",
            interval_sec=config.gpu_telemetry_interval_sec,
        )
        telemetry_sampler.start()

    prompts = build_prompt_set(
        concept_name=config.concept_name,
        estimation_count=config.estimation_prompt_count,
        evaluation_count=config.evaluation_prompt_count,
        seed=config.seed,
        prompt_styles=config.prompt_styles,
    )

    with (run_dir / "prompt_set.json").open("w", encoding="utf-8") as f:
        estimation_pairs = []
        for idx, (positive, negative, style) in enumerate(
            zip(prompts.positive, prompts.negative, prompts.estimation_styles)
        ):
            estimation_pairs.append(
                {
                    "pair_id": f"est_{idx:03d}",
                    "style": style,
                    "positive_prompt": positive,
                    "negative_prompt": negative,
                }
            )
        evaluation_records = []
        for idx, (prompt, style) in enumerate(zip(prompts.evaluation, prompts.evaluation_styles)):
            evaluation_records.append(
                {
                    "prompt_id": f"eval_{idx:03d}",
                    "style": style,
                    "prompt": prompt,
                }
            )
        json.dump(
            {
                "concept_name": prompts.concept_name,
                "positive_estimation": prompts.positive,
                "negative_estimation": prompts.negative,
                "evaluation": prompts.evaluation,
                "estimation_styles": prompts.estimation_styles,
                "evaluation_styles": prompts.evaluation_styles,
                "estimation_pairs": estimation_pairs,
                "evaluation_records": evaluation_records,
            },
            f,
            indent=2,
        )

    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    vector_registry_rows: list[dict[str, object]] = []
    model_registry_rows: list[dict[str, object]] = []
    topk_records: list[dict[str, object]] = []

    for model_id in config.model_ids:
        model_started_at = time.time()
        model_slug = _slug_model_id(model_id)
        model_dir = run_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"[model] loading {model_id}")
        try:
            loaded = load_model(
                model_id=model_id,
                cache_dir=config.model_cache_dir,
                dtype_name=config.dtype,
                use_gpu=config.use_gpu,
                attention_backend=config.attention_backend,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"model_id": model_id, "error": str(exc)})
            model_registry_rows.append(
                {
                    "model_id": model_id,
                    "model_slug": model_slug,
                    "status": "load_failed",
                    "error": str(exc),
                    "elapsed_seconds": float(time.time() - model_started_at),
                }
            )
            print(f"[model] failed to load {model_id}: {exc}")
            clear_cuda()
            continue

        parameter_count = int(sum(p.numel() for p in loaded.model.parameters()))
        with (model_dir / "model_info.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_id": model_id,
                    "n_layers": loaded.n_layers,
                    "hidden_size": loaded.hidden_size,
                    "parameter_count": parameter_count,
                    "device": str(loaded.device),
                    "dtype": str(loaded.torch_dtype),
                },
                f,
                indent=2,
            )

        model_status = "success"
        model_error = ""
        model_rows_start = len(all_rows)
        model_compute = _init_compute_counters(config.enable_compute_accounting)
        try:
            layer_indices = resolve_layer_indices(loaded.n_layers, config.layer_positions)
            baseline_traces: list[TraceResult] = []
            shared_word_vector = None
            if "word_centroid" in config.vector_methods:
                pos_words, neg_words = get_concept_words(prompts.concept_name)
                shared_word_vector = estimate_word_centroid_vector(
                    loaded=loaded,
                    positive_words=pos_words,
                    negative_words=neg_words,
                )

            print(
                f"[model] collecting baseline traces ({len(prompts.evaluation)} prompts) "
                f"batch={config.trace_batch_size}"
            )
            baseline_traces = _trace_prompts_batched(
                loaded=loaded,
                prompts=prompts.evaluation,
                max_prompt_tokens=config.max_prompt_tokens,
                token_position=config.token_position,
                generate_tokens=config.eval_generation_tokens,
                batch_size=config.trace_batch_size,
                adaptive_batching=config.adaptive_batching,
                layer_topk_tokens=config.layer_topk_tokens,
                layer_topk_prompt_limit=config.layer_topk_prompt_limit,
            )
            if bool(model_compute["enabled"]):
                base_prompt_tokens, base_generated_tokens = _trace_token_totals(baseline_traces)
                model_compute["trace_calls"] = int(model_compute["trace_calls"]) + 1
                model_compute["trace_prompt_count"] = int(model_compute["trace_prompt_count"]) + len(
                    baseline_traces
                )
                model_compute["trace_prompt_token_evals"] = int(
                    model_compute["trace_prompt_token_evals"]
                ) + int(base_prompt_tokens)
                model_compute["trace_decode_token_evals"] = int(
                    model_compute["trace_decode_token_evals"]
                ) + int(base_generated_tokens)
                if config.eval_generation_tokens > 0:
                    model_compute["trace_generate_prefill_token_evals"] = int(
                        model_compute["trace_generate_prefill_token_evals"]
                    ) + int(base_prompt_tokens)
            for prompt_idx, baseline in enumerate(baseline_traces):
                if prompt_idx >= len(prompts.evaluation):
                    break
                with (model_dir / f"baseline_prompt_{prompt_idx:03d}.txt").open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(baseline.generated_text)
                if baseline.layer_topk_tokens is not None:
                    topk_records.append(
                        {
                            "run_id": run_id,
                            "run_stamp": run_stamp,
                            "model_id": model_id,
                            "prompt_index": int(prompt_idx),
                            "prompt_style": prompts.evaluation_styles[prompt_idx],
                            "prompt": prompts.evaluation[prompt_idx],
                            "condition": "baseline",
                            "vector_method": "baseline",
                            "alpha": 0.0,
                            "inject_layer_index": -1,
                            "topk_tokens": int(config.layer_topk_tokens),
                            "topk_layers_json": json.dumps(baseline.layer_topk_tokens),
                        }
                    )

            for layer_index in layer_indices:
                layer_depth_ratio = float(layer_index / max(1, loaded.n_layers - 1))
                print(f"[model] estimating vectors at layer {layer_index}")
                pos_acts, pos_stats = extract_layer_activations(
                    loaded=loaded,
                    prompts=prompts.positive,
                    layer_index=layer_index,
                    token_position=config.estimation_token_position,
                    max_prompt_tokens=config.max_prompt_tokens,
                    batch_size=config.activation_batch_size,
                    return_stats=True,
                )
                neg_acts, neg_stats = extract_layer_activations(
                    loaded=loaded,
                    prompts=prompts.negative,
                    layer_index=layer_index,
                    token_position=config.estimation_token_position,
                    max_prompt_tokens=config.max_prompt_tokens,
                    batch_size=config.activation_batch_size,
                    return_stats=True,
                )
                if bool(model_compute["enabled"]):
                    model_compute["activation_prompt_count"] = int(
                        model_compute["activation_prompt_count"]
                    ) + int(pos_stats["prompt_count"]) + int(neg_stats["prompt_count"])
                    model_compute["activation_prompt_token_evals"] = int(
                        model_compute["activation_prompt_token_evals"]
                    ) + int(pos_stats["prompt_token_count"]) + int(neg_stats["prompt_token_count"])

                vectors = estimate_concept_vectors(
                    methods=[m for m in config.vector_methods if m != "word_centroid"],
                    positive_acts=pos_acts,
                    negative_acts=neg_acts,
                    seed=config.seed,
                )
                if shared_word_vector is not None:
                    vectors.append(shared_word_vector)

                for vec in vectors:
                    vec_norm = float(torch.linalg.vector_norm(vec.vector).item())
                    vectors_dir = model_dir / "vectors"
                    vectors_dir.mkdir(parents=True, exist_ok=True)
                    vector_file = vectors_dir / f"layer_{layer_index:03d}_{vec.method}.npy"
                    np.save(vector_file, vec.vector.detach().float().cpu().numpy())
                    vector_registry_rows.append(
                        {
                            "run_id": run_id,
                            "run_stamp": run_stamp,
                            "model_id": model_id,
                            "model_slug": model_slug,
                            "layer_index": int(layer_index),
                            "layer_depth_ratio": layer_depth_ratio,
                            "vector_method": vec.method,
                            "vector_norm": vec_norm,
                            "vector_file": str(vector_file),
                            "vector_metadata_json": json.dumps(vec.metadata, sort_keys=True),
                        }
                    )
                    for alpha in config.alphas:
                        print(
                            f"[model] layer={layer_index} method={vec.method} alpha={alpha} "
                            f"prompts={len(prompts.evaluation)} batch={config.trace_batch_size}"
                        )
                        injected_traces = _trace_prompts_batched(
                            loaded=loaded,
                            prompts=prompts.evaluation,
                            max_prompt_tokens=config.max_prompt_tokens,
                            token_position=config.token_position,
                            generate_tokens=config.eval_generation_tokens,
                            batch_size=config.trace_batch_size,
                            adaptive_batching=config.adaptive_batching,
                            inject_layer=layer_index,
                            inject_vector=vec.vector,
                            alpha=float(alpha),
                            layer_topk_tokens=config.layer_topk_tokens,
                            layer_topk_prompt_limit=config.layer_topk_prompt_limit,
                        )
                        if bool(model_compute["enabled"]):
                            inj_prompt_tokens, inj_generated_tokens = _trace_token_totals(injected_traces)
                            model_compute["trace_calls"] = int(model_compute["trace_calls"]) + 1
                            model_compute["trace_prompt_count"] = int(
                                model_compute["trace_prompt_count"]
                            ) + len(injected_traces)
                            model_compute["trace_prompt_token_evals"] = int(
                                model_compute["trace_prompt_token_evals"]
                            ) + int(inj_prompt_tokens)
                            model_compute["trace_decode_token_evals"] = int(
                                model_compute["trace_decode_token_evals"]
                            ) + int(inj_generated_tokens)
                            if config.eval_generation_tokens > 0:
                                model_compute["trace_generate_prefill_token_evals"] = int(
                                    model_compute["trace_generate_prefill_token_evals"]
                                ) + int(inj_prompt_tokens)
                        if len(injected_traces) != len(baseline_traces):
                            raise RuntimeError(
                                "Mismatched batch trace lengths between baseline and injected runs: "
                                f"{len(baseline_traces)} vs {len(injected_traces)}"
                            )
                        for prompt_idx, (baseline, injected) in enumerate(
                            zip(baseline_traces, injected_traces)
                        ):
                            prompt = prompts.evaluation[prompt_idx]
                            if injected.layer_topk_tokens is not None:
                                topk_records.append(
                                    {
                                        "run_id": run_id,
                                        "run_stamp": run_stamp,
                                        "model_id": model_id,
                                        "prompt_index": int(prompt_idx),
                                        "prompt_style": prompts.evaluation_styles[prompt_idx],
                                        "prompt": prompt,
                                        "condition": "injected",
                                        "vector_method": vec.method,
                                        "alpha": float(alpha),
                                        "inject_layer_index": int(layer_index),
                                        "topk_tokens": int(config.layer_topk_tokens),
                                        "topk_layers_json": json.dumps(injected.layer_topk_tokens),
                                    }
                                )

                            metrics = compute_trajectory_metrics(
                                baseline_states=baseline.per_layer_states,
                                injected_states=injected.per_layer_states,
                                baseline_logits=baseline.next_token_logits,
                                injected_logits=injected.next_token_logits,
                                inject_layer_index=layer_index,
                                recovery_threshold=config.recovery_threshold,
                            )

                            start_idx = int(metrics.drift_start_index)
                            drift_at_start = float(metrics.drift_by_layer[start_idx])
                            drift_at_start_rel = float(metrics.relative_drift_by_layer[start_idx])
                            baseline_norm_at_start = float(
                                torch.linalg.vector_norm(baseline.per_layer_states[start_idx]).item()
                            )
                            effective_push_abs = float(abs(alpha) * vec_norm)
                            effective_push_rel_baseline = (
                                effective_push_abs / max(1e-12, baseline_norm_at_start)
                            )
                            cad = drift_at_start / max(1e-12, effective_push_abs)
                            cad_rel = drift_at_start_rel / max(1e-12, effective_push_rel_baseline)
                            persistence = metrics.end_drift / max(1e-12, metrics.peak_drift)
                            degradation = metrics.peak_drift_relative

                            row = {
                                "run_id": run_id,
                                "run_stamp": run_stamp,
                                "seed": int(config.seed),
                                "concept_name": prompts.concept_name,
                                "model_id": model_id,
                                "n_layers": int(loaded.n_layers),
                                "token_position": int(config.token_position),
                                "estimation_token_position": int(config.estimation_token_position),
                                "layer_index": int(layer_index),
                                "layer_depth_ratio": layer_depth_ratio,
                                "vector_method": vec.method,
                                "vector_norm": vec_norm,
                                "alpha": float(alpha),
                                "effective_push_abs": effective_push_abs,
                                "prompt_index": int(prompt_idx),
                                "prompt": prompt,
                                "prompt_style": prompts.evaluation_styles[prompt_idx],
                                "drift_start_index": metrics.drift_start_index,
                                "drift_end_index": metrics.drift_end_index,
                                "drift_at_start": drift_at_start,
                                "drift_at_start_relative": drift_at_start_rel,
                                "cad": cad,
                                "cad_relative": cad_rel,
                                "peak_drift": metrics.peak_drift,
                                "peak_drift_relative": metrics.peak_drift_relative,
                                "end_drift": metrics.end_drift,
                                "end_drift_relative": metrics.end_drift_relative,
                                "drift_auc": metrics.drift_auc,
                                "drift_auc_relative": metrics.drift_auc_relative,
                                "degradation": degradation,
                                "persistence": persistence,
                                "recovery_fraction": metrics.recovery_fraction,
                                "recovery_slope": metrics.recovery_slope,
                                "recovery_latency_layers": metrics.recovery_latency_layers,
                                "recoverable_layers": metrics.recoverable_layers,
                                "overshoot_index": metrics.overshoot_index,
                                "crossed_baseline": metrics.crossed_baseline,
                                "end_cosine_alignment": metrics.end_cosine_alignment,
                                "next_token_kl": metrics.next_token_kl,
                                "baseline_generation": baseline.generated_text,
                                "injected_generation": injected.generated_text,
                                "baseline_prompt_tokens": int(baseline.prompt_token_count),
                                "baseline_generated_tokens": int(baseline.generated_token_count),
                                "injected_prompt_tokens": int(injected.prompt_token_count),
                                "injected_generated_tokens": int(injected.generated_token_count),
                                "drift_by_layer": json.dumps(metrics.drift_by_layer),
                                "relative_drift_by_layer": json.dumps(metrics.relative_drift_by_layer),
                                "projection_by_layer": json.dumps(metrics.projection_by_layer),
                            }
                            row["baseline_norm_at_start"] = baseline_norm_at_start
                            row["effective_push_rel_baseline"] = effective_push_rel_baseline
                            row.update({f"vector_meta_{k}": v for k, v in vec.metadata.items()})
                            all_rows.append(row)

        except RuntimeError as exc:
            failures.append({"model_id": model_id, "error": f"RuntimeError: {exc}"})
            model_status = "runtime_failed"
            model_error = f"RuntimeError: {exc}"
            print(f"[model] runtime failure for {model_id}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"model_id": model_id, "error": str(exc)})
            model_status = "failed"
            model_error = str(exc)
            print(f"[model] unexpected failure for {model_id}: {exc}")
        finally:
            model_elapsed = float(time.time() - model_started_at)
            model_compute = _finalize_compute_counters(
                counters=model_compute,
                parameter_count=parameter_count,
                elapsed_seconds=model_elapsed,
            )
            with (model_dir / "compute_accounting.json").open("w", encoding="utf-8") as f:
                json.dump(_to_jsonable(model_compute), f, indent=2)
            model_registry_rows.append(
                {
                    "model_id": model_id,
                    "model_slug": model_slug,
                    "parameter_count": parameter_count,
                    "n_layers": int(loaded.n_layers),
                    "hidden_size": int(loaded.hidden_size),
                    "status": model_status,
                    "error": model_error,
                    "rows_emitted": int(len(all_rows) - model_rows_start),
                    "elapsed_seconds": model_elapsed,
                    "compute_accounting_enabled": bool(model_compute["enabled"]),
                    "trace_calls": int(model_compute["trace_calls"]),
                    "trace_prompt_count": int(model_compute["trace_prompt_count"]),
                    "trace_prompt_token_evals": int(model_compute["trace_prompt_token_evals"]),
                    "trace_generate_prefill_token_evals": int(
                        model_compute["trace_generate_prefill_token_evals"]
                    ),
                    "trace_decode_token_evals": int(model_compute["trace_decode_token_evals"]),
                    "activation_prompt_count": int(model_compute["activation_prompt_count"]),
                    "activation_prompt_token_evals": int(
                        model_compute["activation_prompt_token_evals"]
                    ),
                    "total_token_evals": int(model_compute["total_token_evals"]),
                    "approx_forward_flops_2n": float(model_compute["approx_forward_flops_2n"]),
                    "approx_forward_flops_6n": float(model_compute["approx_forward_flops_6n"]),
                    "token_evals_per_second": model_compute.get("token_evals_per_second"),
                }
            )
            del loaded
            clear_cuda()

    metrics_path = run_dir / "metrics_full.csv"
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(metrics_path, index=False)

        summary = (
            df.groupby(["concept_name", "model_id", "vector_method", "alpha"], as_index=False)
            .agg(
                peak_drift_mean=("peak_drift", "mean"),
                peak_drift_relative_mean=("peak_drift_relative", "mean"),
                end_drift_mean=("end_drift", "mean"),
                end_drift_relative_mean=("end_drift_relative", "mean"),
                drift_auc_mean=("drift_auc", "mean"),
                drift_auc_relative_mean=("drift_auc_relative", "mean"),
                recovery_fraction_mean=("recovery_fraction", "mean"),
                recovery_slope_mean=("recovery_slope", "mean"),
                cad_mean=("cad", "mean"),
                persistence_mean=("persistence", "mean"),
                degradation_mean=("degradation", "mean"),
                overshoot_index_mean=("overshoot_index", "mean"),
                crossed_baseline_rate=("crossed_baseline", "mean"),
                next_token_kl_mean=("next_token_kl", "mean"),
                n=("prompt_index", "count"),
            )
            .sort_values(["concept_name", "model_id", "vector_method", "alpha"])
        )
        summary.to_csv(run_dir / "metrics_summary.csv", index=False)

        layer_summary = (
            df.groupby(["concept_name", "model_id", "layer_index", "vector_method"], as_index=False)
            .agg(
                peak_drift_mean=("peak_drift", "mean"),
                peak_drift_relative_mean=("peak_drift_relative", "mean"),
                drift_auc_mean=("drift_auc", "mean"),
                drift_auc_relative_mean=("drift_auc_relative", "mean"),
                recovery_fraction_mean=("recovery_fraction", "mean"),
                recovery_slope_mean=("recovery_slope", "mean"),
                cad_mean=("cad", "mean"),
                persistence_mean=("persistence", "mean"),
                degradation_mean=("degradation", "mean"),
                crossed_baseline_rate=("crossed_baseline", "mean"),
                n=("prompt_index", "count"),
            )
            .sort_values(["concept_name", "model_id", "layer_index", "vector_method"])
        )
        layer_summary.to_csv(run_dir / "layer_summary.csv", index=False)

        with (run_dir / "quick_report.md").open("w", encoding="utf-8") as f:
            f.write("# Quick Report\n\n")
            f.write(f"- Run stamp: `{run_stamp}`\n")
            f.write(f"- Concept: `{prompts.concept_name}`\n")
            f.write(f"- Seed: `{config.seed}`\n")
            f.write(f"- Rows: `{len(df)}`\n")
            f.write(f"- Models attempted: `{len(config.model_ids)}`\n")
            f.write(f"- Models with outputs: `{df['model_id'].nunique()}`\n\n")
            f.write("## Mean Recovery by Model\n\n")
            by_model = (
                df.groupby("model_id", as_index=False)
                .agg(
                    recovery_fraction_mean=("recovery_fraction", "mean"),
                    peak_drift_mean=("peak_drift", "mean"),
                    peak_drift_relative_mean=("peak_drift_relative", "mean"),
                    cad_mean=("cad", "mean"),
                    persistence_mean=("persistence", "mean"),
                    overshoot_index_mean=("overshoot_index", "mean"),
                    crossed_baseline_rate=("crossed_baseline", "mean"),
                )
                .sort_values("model_id")
            )
            for _, row in by_model.iterrows():
                f.write(
                    "- "
                    f"{row['model_id']}: "
                    f"recovery={row['recovery_fraction_mean']:.4f}, "
                    f"peak={row['peak_drift_mean']:.4f}, "
                    f"peak_rel={row['peak_drift_relative_mean']:.6f}, "
                    f"cad={row['cad_mean']:.4f}, "
                    f"persist={row['persistence_mean']:.4f}, "
                    f"overshoot={row['overshoot_index_mean']:.4f}, "
                    f"crossed={row['crossed_baseline_rate']:.4f}\n"
                )
    else:
        with (run_dir / "quick_report.md").open("w", encoding="utf-8") as f:
            f.write("# Quick Report\n\nNo successful experiment rows were produced.\n")

    with (run_dir / "failures.json").open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    if vector_registry_rows:
        pd.DataFrame(vector_registry_rows).to_csv(run_dir / "vector_registry.csv", index=False)
    if model_registry_rows:
        pd.DataFrame(model_registry_rows).to_csv(run_dir / "model_registry.csv", index=False)
    compute_summary = {
        "enabled": bool(config.enable_compute_accounting),
        "models_accounted": 0,
        "trace_calls": 0,
        "trace_prompt_count": 0,
        "trace_prompt_token_evals": 0,
        "trace_generate_prefill_token_evals": 0,
        "trace_decode_token_evals": 0,
        "activation_prompt_count": 0,
        "activation_prompt_token_evals": 0,
        "total_token_evals": 0,
        "approx_forward_flops_2n": 0.0,
        "approx_forward_flops_6n": 0.0,
    }
    for row in model_registry_rows:
        if not bool(row.get("compute_accounting_enabled", False)):
            continue
        compute_summary["models_accounted"] += 1
        compute_summary["trace_calls"] += int(row.get("trace_calls", 0))
        compute_summary["trace_prompt_count"] += int(row.get("trace_prompt_count", 0))
        compute_summary["trace_prompt_token_evals"] += int(row.get("trace_prompt_token_evals", 0))
        compute_summary["trace_generate_prefill_token_evals"] += int(
            row.get("trace_generate_prefill_token_evals", 0)
        )
        compute_summary["trace_decode_token_evals"] += int(row.get("trace_decode_token_evals", 0))
        compute_summary["activation_prompt_count"] += int(row.get("activation_prompt_count", 0))
        compute_summary["activation_prompt_token_evals"] += int(
            row.get("activation_prompt_token_evals", 0)
        )
        compute_summary["total_token_evals"] += int(row.get("total_token_evals", 0))
        compute_summary["approx_forward_flops_2n"] += float(row.get("approx_forward_flops_2n", 0.0))
        compute_summary["approx_forward_flops_6n"] += float(row.get("approx_forward_flops_6n", 0.0))
    with (run_dir / "compute_accounting.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(compute_summary), f, indent=2)
    if topk_records:
        with (run_dir / "layer_topk_records.jsonl").open("w", encoding="utf-8") as f:
            for rec in topk_records:
                f.write(json.dumps(_to_jsonable(rec)) + "\n")

    if telemetry_sampler is not None:
        telemetry_sampler.stop()
        telemetry_summary = summarize_gpu_telemetry_csv(
            csv_path=run_dir / "gpu_telemetry.csv",
            interval_sec=config.gpu_telemetry_interval_sec,
            sampler_errors=telemetry_sampler.errors,
        )
        telemetry_summary["enabled"] = bool(telemetry_sampler.enabled)
        telemetry_summary["started_at_utc"] = telemetry_sampler.started_at_utc
        telemetry_summary["stopped_at_utc"] = telemetry_sampler.stopped_at_utc
    with (run_dir / "gpu_telemetry_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(telemetry_summary), f, indent=2)

    run_completed_at = time.time()
    run_summary = {
        "run_id": run_id,
        "run_stamp": run_stamp,
        "concept_name": prompts.concept_name,
        "seed": int(config.seed),
        "models_attempted": int(len(config.model_ids)),
        "models_with_outputs": int(len({row["model_id"] for row in all_rows})) if all_rows else 0,
        "rows_emitted": int(len(all_rows)),
        "vector_registry_rows": int(len(vector_registry_rows)),
        "layer_topk_record_count": int(len(topk_records)),
        "failure_count": int(len(failures)),
        "compute_accounting": compute_summary,
        "gpu_telemetry": telemetry_summary,
        "started_at_utc": datetime.utcfromtimestamp(run_started_at).isoformat(timespec="seconds") + "Z",
        "completed_at_utc": datetime.utcfromtimestamp(run_completed_at).isoformat(timespec="seconds") + "Z",
        "elapsed_seconds": float(run_completed_at - run_started_at),
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    return run_dir
