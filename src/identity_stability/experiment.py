from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .intervention import TraceResult, resolve_layer_indices, run_trace_batch
from .metrics import compute_trajectory_metrics
from .modeling import clear_cuda, load_model
from .prompt_bank import build_prompt_set, get_concept_words
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


def _configure_cuda_runtime(enable_tf32: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
    torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


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


def run_experiment(config: RunConfig) -> Path:
    _set_seed(config.seed)
    _configure_cuda_runtime(config.enable_tf32)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_root / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(asdict(config)), f, indent=2)

    prompts = build_prompt_set(
        concept_name=config.concept_name,
        estimation_count=config.estimation_prompt_count,
        evaluation_count=config.evaluation_prompt_count,
        seed=config.seed,
        prompt_styles=config.prompt_styles,
    )

    with (run_dir / "prompt_set.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "concept_name": prompts.concept_name,
                "positive_estimation": prompts.positive,
                "negative_estimation": prompts.negative,
                "evaluation": prompts.evaluation,
                "estimation_styles": prompts.estimation_styles,
                "evaluation_styles": prompts.evaluation_styles,
            },
            f,
            indent=2,
        )

    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for model_id in config.model_ids:
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
            print(f"[model] failed to load {model_id}: {exc}")
            clear_cuda()
            continue

        with (model_dir / "model_info.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_id": model_id,
                    "n_layers": loaded.n_layers,
                    "hidden_size": loaded.hidden_size,
                    "device": str(loaded.device),
                    "dtype": str(loaded.torch_dtype),
                },
                f,
                indent=2,
            )

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
            )
            for prompt_idx, baseline in enumerate(baseline_traces):
                if prompt_idx >= len(prompts.evaluation):
                    break
                with (model_dir / f"baseline_prompt_{prompt_idx:03d}.txt").open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(baseline.generated_text)

            for layer_index in layer_indices:
                layer_depth_ratio = float(layer_index / max(1, loaded.n_layers - 1))
                print(f"[model] estimating vectors at layer {layer_index}")
                pos_acts = extract_layer_activations(
                    loaded=loaded,
                    prompts=prompts.positive,
                    layer_index=layer_index,
                    token_position=config.estimation_token_position,
                    max_prompt_tokens=config.max_prompt_tokens,
                    batch_size=config.activation_batch_size,
                )
                neg_acts = extract_layer_activations(
                    loaded=loaded,
                    prompts=prompts.negative,
                    layer_index=layer_index,
                    token_position=config.estimation_token_position,
                    max_prompt_tokens=config.max_prompt_tokens,
                    batch_size=config.activation_batch_size,
                )

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
                        )
                        if len(injected_traces) != len(baseline_traces):
                            raise RuntimeError(
                                "Mismatched batch trace lengths between baseline and injected runs: "
                                f"{len(baseline_traces)} vs {len(injected_traces)}"
                            )
                        for prompt_idx, (baseline, injected) in enumerate(
                            zip(baseline_traces, injected_traces)
                        ):
                            prompt = prompts.evaluation[prompt_idx]

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
            print(f"[model] runtime failure for {model_id}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"model_id": model_id, "error": str(exc)})
            print(f"[model] unexpected failure for {model_id}: {exc}")
        finally:
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

    return run_dir
