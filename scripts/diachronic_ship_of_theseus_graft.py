from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from identity_battery_common import (
    add_src_to_path,
    ensure_output_dir,
    infer_model_family,
    infer_model_size_label,
    load_yaml_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the diachronic Ship-of-Theseus identity graft experiment.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _load_prompt_bank(path: Path) -> list[dict[str, str]]:
    raw = load_yaml_config(path)
    items = raw.get("items", [])
    prompts: list[dict[str, str]] = []
    for item in items:
        prompts.append(
            {
                "id": str(item["id"]),
                "family": str(item["family"]),
                "prompt": str(item["prompt"]),
            }
        )
    return prompts


def _trace_prompts(
    loaded,
    prompt_items: list[dict[str, str]],
    *,
    token_position: int,
    generate_tokens: int,
    max_prompt_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    from identity_stability.intervention import run_trace_batch

    traces: dict[str, Any] = {}
    for start in range(0, len(prompt_items), max(1, int(batch_size))):
        chunk = prompt_items[start : start + max(1, int(batch_size))]
        results = run_trace_batch(
            loaded=loaded,
            prompts=[item["prompt"] for item in chunk],
            max_prompt_tokens=max_prompt_tokens,
            token_position=token_position,
            generate_tokens=generate_tokens,
        )
        for item, trace in zip(chunk, results):
            traces[str(item["id"])] = trace
    return traces


def _js_distance_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    p = torch.softmax(logits_a.float(), dim=-1)
    q = torch.softmax(logits_b.float(), dim=-1)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p + 1e-12) - torch.log(m + 1e-12)))
    kl_qm = torch.sum(q * (torch.log(q + 1e-12) - torch.log(m + 1e-12)))
    return float((0.5 * (kl_pm + kl_qm)).item())


def _donor_identity_fraction(
    graft_logits: torch.Tensor,
    host_logits: torch.Tensor,
    donor_logits: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    js_to_host = _js_distance_from_logits(graft_logits, host_logits)
    js_to_donor = _js_distance_from_logits(graft_logits, donor_logits)
    dif = js_to_host / (js_to_host + js_to_donor + eps)
    return float(dif), float(js_to_host), float(js_to_donor)


def _distance_based_identity_fraction(distance_to_host: float, distance_to_donor: float, *, eps: float = 1e-12) -> float:
    return float(distance_to_host / (distance_to_host + distance_to_donor + eps))


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _random_same_norm_state(host_state: torch.Tensor, donor_state: torch.Tensor, *, seed: int) -> torch.Tensor:
    delta = donor_state.float() - host_state.float()
    delta_norm = float(torch.linalg.norm(delta).item())
    if delta_norm <= 1e-12:
        return host_state.clone().float()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rand = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
    rand_norm = float(torch.linalg.norm(rand).item())
    if rand_norm <= 1e-12:
        return host_state.clone().float()
    scaled = rand * (delta_norm / rand_norm)
    return host_state.float() + scaled


def _token_position_label(token_position: int) -> str:
    if int(token_position) == -1:
        return "last_prompt_token"
    if int(token_position) == 0:
        return "first_prompt_token"
    return f"token_{token_position}"


def _resolve_mode_layers(mode: str, boundary_layer: int, n_layers: int) -> list[int]:
    if mode == "single_layer":
        return [int(boundary_layer)]
    if mode == "prefix":
        return list(range(0, int(boundary_layer) + 1))
    if mode == "suffix":
        return list(range(int(boundary_layer) + 1, int(n_layers)))
    raise ValueError(f"Unsupported graft mode: {mode}")


def _self_report_prompt(
    *,
    question: str,
    completion: str,
    host_revision: str,
    donor_revision: str,
) -> str:
    return (
        "You will see a question and an answer.\n"
        "Which checkpoint does the answer more closely resemble in style and behavior?\n"
        f"Question: {question}\n\n"
        f"Answer: {completion}\n\n"
        f"A: {host_revision}\n"
        f"B: {donor_revision}\n\n"
        "Reply with exactly one label from: A, B.\n"
        "Label:"
    )


def _self_report_bias_prompt() -> str:
    return "Formatting-only task.\nReply with exactly one label from: A, B.\nLabel:"


def _name_only_prompt(*, donor_revision: str, base_prompt: str) -> str:
    return (
        f"You are checkpoint {donor_revision} from an earlier or later moment in the same training run.\n"
        "Answer the prompt naturally.\n\n"
        f"Prompt: {base_prompt}"
    )


def _write_summary(df: pd.DataFrame, group_cols: list[str], path: Path) -> pd.DataFrame:
    from identity_stability.identity_analysis import bootstrap_mean_ci

    rows: list[dict[str, object]] = []
    metric_cols = [
        "donor_identity_fraction",
        "js_to_host",
        "js_to_donor",
        "text_donor_identity_fraction",
        "semantic_donor_identity_fraction",
        "recovery_fraction",
        "persistence",
        "cad",
        "next_token_kl",
        "activation_norm_deviation",
        "verbal_donor_claim",
    ]
    for keys, sub in df.groupby(group_cols, as_index=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["n"] = int(len(sub))
        row["prompt_count"] = int(sub["prompt_id"].nunique()) if "prompt_id" in sub.columns else int(len(sub))
        for metric in metric_cols:
            values = sub[metric].dropna().astype(float).tolist()
            row[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
            low, high = bootstrap_mean_ci(values, iters=1000, seed=123) if values else (float("nan"), float("nan"))
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(path, index=False)
    return summary


def main() -> None:
    add_src_to_path()
    from identity_stability.identity_probe_tools import predict_labeled_choice
    from identity_stability.intervention import ResidualGraft, resolve_layer_indices, run_grafted_trace
    from identity_stability.metrics import compute_trajectory_metrics
    from identity_stability.modeling import clear_cuda, load_model
    from identity_stability.text_features import semantic_overlap, stylometric_distance

    args = parse_args()
    config = load_yaml_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = ensure_output_dir(config, "diachronic_ship_of_theseus_graft")
    prompt_bank_path = repo_root / str(config["prompt_bank_path"])
    prompt_bank = _load_prompt_bank(prompt_bank_path)
    checkpoint_every = int(config.get("checkpoint_every_rows", 0))

    model_id = str(config["model_id"])
    cache_dir = Path(str(config["model_cache_dir"]))
    dtype_name = str(config["dtype"])
    use_gpu = bool(config["use_gpu"])
    attention_backend = str(config.get("attention_backend", "auto"))
    max_prompt_tokens = int(config["max_prompt_tokens"])
    generate_tokens = int(config.get("generate_tokens", 48))
    trace_batch_size = int(config.get("trace_batch_size", 4))
    selection_batch_size = int(config.get("selection_batch_size", trace_batch_size))
    selection_token_position = int(config.get("selection_token_position", -1))
    prompt_limit = int(config.get("prompt_limit", len(prompt_bank)))
    prompt_prefilter_count = int(config.get("prompt_prefilter_count", min(len(prompt_bank), prompt_limit)))
    min_clean_js = float(config.get("min_clean_js", 0.0))
    top_k_prompts = int(config.get("selected_prompt_count", min(prompt_limit, prompt_prefilter_count)))
    token_positions = [int(value) for value in config["token_positions"]]
    lambdas = [float(value) for value in config["lambdas"]]
    graft_modes = [str(value) for value in config["graft_modes"]]
    layer_buckets_raw = list(config["layer_buckets"])
    control_kinds = [str(value) for value in config["control_kinds"]]
    run_self_report = bool(config.get("run_self_report", True))
    pair_specs = list(config["pairs"])
    selection_pair = dict(config["selection_pair"])

    candidate_prompts = [dict(item) for item in prompt_bank[:prompt_prefilter_count]]
    selection_host_rev = str(selection_pair["host_revision"])
    selection_donor_rev = str(selection_pair["donor_revision"])

    selection_cache: dict[str, dict[str, Any]] = {}
    for revision in [selection_host_rev, selection_donor_rev]:
        loaded = load_model(
            model_id=model_id,
            cache_dir=cache_dir,
            dtype_name=dtype_name,
            use_gpu=use_gpu,
            attention_backend=attention_backend,
            revision=revision,
        )
        selection_cache[revision] = _trace_prompts(
            loaded,
            candidate_prompts,
            token_position=selection_token_position,
            generate_tokens=0,
            max_prompt_tokens=max_prompt_tokens,
            batch_size=selection_batch_size,
        )
        del loaded
        clear_cuda()

    divergence_rows: list[dict[str, object]] = []
    for item in candidate_prompts:
        prompt_id = str(item["id"])
        host_trace = selection_cache[selection_host_rev][prompt_id]
        donor_trace = selection_cache[selection_donor_rev][prompt_id]
        clean_js = _js_distance_from_logits(host_trace.next_token_logits, donor_trace.next_token_logits)
        divergence_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_family": str(item["family"]),
                "prompt": str(item["prompt"]),
                "selection_host_revision": selection_host_rev,
                "selection_donor_revision": selection_donor_rev,
                "clean_js": clean_js,
            }
        )
    divergence_df = pd.DataFrame(divergence_rows).sort_values("clean_js", ascending=False).reset_index(drop=True)
    divergence_df.to_csv(output_dir / "clean_prompt_divergence.csv", index=False)
    selected_df = divergence_df[divergence_df["clean_js"] >= min_clean_js].head(top_k_prompts).copy()
    if selected_df.empty:
        selected_df = divergence_df.head(min(top_k_prompts, len(divergence_df))).copy()
    selected_df["selection_rank"] = np.arange(1, len(selected_df) + 1)
    selected_df.to_csv(output_dir / "selected_prompts.csv", index=False)
    selected_prompt_ids = selected_df["prompt_id"].tolist()
    selected_items = [item for item in candidate_prompts if str(item["id"]) in set(selected_prompt_ids)]
    selected_items.sort(key=lambda item: int(selected_df.index[selected_df["prompt_id"] == str(item["id"])][0]))

    layer_labels = [str(bucket["label"]) for bucket in layer_buckets_raw]
    layer_positions = [float(bucket["position"]) for bucket in layer_buckets_raw]

    all_revisions: set[str] = set()
    for pair_spec in pair_specs:
        all_revisions.add(str(pair_spec["host_revision"]))
        all_revisions.add(str(pair_spec["donor_revision"]))
        if pair_spec.get("adjacent_donor_revision"):
            all_revisions.add(str(pair_spec["adjacent_donor_revision"]))
        if pair_spec.get("very_early_donor_revision"):
            all_revisions.add(str(pair_spec["very_early_donor_revision"]))

    clean_cache: dict[tuple[str, int, str], Any] = {}
    layer_index_lookup: dict[str, list[int]] = {}
    for revision in sorted(all_revisions):
        loaded = load_model(
            model_id=model_id,
            cache_dir=cache_dir,
            dtype_name=dtype_name,
            use_gpu=use_gpu,
            attention_backend=attention_backend,
            revision=revision,
        )
        if revision not in layer_index_lookup:
            layer_index_lookup[revision] = resolve_layer_indices(loaded.n_layers, layer_positions)
        for token_position in token_positions:
            traces = _trace_prompts(
                loaded,
                selected_items,
                token_position=token_position,
                generate_tokens=generate_tokens,
                max_prompt_tokens=max_prompt_tokens,
                batch_size=trace_batch_size,
            )
            for prompt_id, trace in traces.items():
                clean_cache[(revision, token_position, prompt_id)] = trace
        del loaded
        clear_cuda()

    results_path = output_dir / "results.csv"
    partial_path = output_dir / "results.partial.csv"
    rows: list[dict[str, object]] = []

    for pair_spec in pair_specs:
        pair_name = str(pair_spec["pair_name"])
        host_revision = str(pair_spec["host_revision"])
        donor_revision = str(pair_spec["donor_revision"])
        adjacent_revision = str(pair_spec["adjacent_donor_revision"]) if pair_spec.get("adjacent_donor_revision") else None
        early_revision = str(pair_spec["very_early_donor_revision"]) if pair_spec.get("very_early_donor_revision") else None

        host_loaded = load_model(
            model_id=model_id,
            cache_dir=cache_dir,
            dtype_name=dtype_name,
            use_gpu=use_gpu,
            attention_backend=attention_backend,
            revision=host_revision,
        )
        layer_indices = resolve_layer_indices(host_loaded.n_layers, layer_positions)

        shuffled_prompt_ids = {}
        for idx, item in enumerate(selected_items):
            shuffled_prompt_ids[str(item["id"])] = str(selected_items[(idx + 1) % len(selected_items)]["id"])

        for token_position in token_positions:
            for item in selected_items:
                prompt_id = str(item["id"])
                prompt_family = str(item["family"])
                prompt_text = str(item["prompt"])
                prompt_rank = int(selected_df.loc[selected_df["prompt_id"] == prompt_id, "selection_rank"].iloc[0])
                prompt_js = float(selected_df.loc[selected_df["prompt_id"] == prompt_id, "clean_js"].iloc[0])
                host_clean = clean_cache[(host_revision, token_position, prompt_id)]
                donor_clean_primary = clean_cache[(donor_revision, token_position, prompt_id)]

                # Name-only prompt control: no activation graft, identity language only.
                if "name_only" in control_kinds:
                    control_prompt = _name_only_prompt(donor_revision=donor_revision, base_prompt=prompt_text)
                    name_only_trace = run_grafted_trace(
                        loaded=host_loaded,
                        prompt=control_prompt,
                        max_prompt_tokens=max_prompt_tokens,
                        token_position=token_position,
                        generate_tokens=generate_tokens,
                        grafts=[],
                    )
                    donor_fraction, js_to_host, js_to_donor = _donor_identity_fraction(
                        name_only_trace.next_token_logits,
                        host_clean.next_token_logits,
                        donor_clean_primary.next_token_logits,
                    )
                    host_text = host_clean.completion_text or host_clean.generated_text
                    donor_text = donor_clean_primary.completion_text or donor_clean_primary.generated_text
                    graft_text = name_only_trace.completion_text or name_only_trace.generated_text
                    text_host_distance = stylometric_distance(graft_text, host_text)
                    text_donor_distance = stylometric_distance(graft_text, donor_text)
                    semantic_host_overlap = semantic_overlap(graft_text, host_text)
                    semantic_donor_overlap = semantic_overlap(graft_text, donor_text)
                    verbal_claim = float("nan")
                    verbal_confidence = float("nan")
                    if run_self_report:
                        choice = predict_labeled_choice(
                            loaded=host_loaded,
                            prompt=_self_report_prompt(
                                question=prompt_text,
                                completion=graft_text,
                                host_revision=host_revision,
                                donor_revision=donor_revision,
                            ),
                            max_prompt_tokens=max_prompt_tokens,
                            labels=[("A", "A", 0.0), ("B", "B", 1.0)],
                            label_bias_prompt=_self_report_bias_prompt(),
                        )
                        verbal_claim = float(choice[2]) if choice[0] in {"A", "B"} else float("nan")
                        verbal_confidence = float(choice[3]) if choice[0] in {"A", "B"} else float("nan")
                    rows.append(
                        {
                            "pair_name": pair_name,
                            "model_id": model_id,
                            "model_family": infer_model_family(model_id),
                            "model_size_label": infer_model_size_label(model_id),
                            "host_revision": host_revision,
                            "donor_revision": donor_revision,
                            "control_kind": "name_only",
                            "graft_mode": "none",
                            "layer_label": "",
                            "boundary_layer_index": -1,
                            "n_grafted_layers": 0,
                            "blend_lambda": 0.0,
                            "token_position": token_position,
                            "token_position_label": _token_position_label(token_position),
                            "prompt_id": prompt_id,
                            "prompt_family": prompt_family,
                            "prompt_rank": prompt_rank,
                            "prompt_clean_js": prompt_js,
                            "shuffled_prompt_id": "",
                            "donor_identity_fraction": donor_fraction,
                            "js_to_host": js_to_host,
                            "js_to_donor": js_to_donor,
                            "text_donor_identity_fraction": _distance_based_identity_fraction(text_host_distance, text_donor_distance),
                            "semantic_donor_identity_fraction": _distance_based_identity_fraction(
                                1.0 - semantic_host_overlap,
                                1.0 - semantic_donor_overlap,
                            ),
                            "text_host_distance": text_host_distance,
                            "text_donor_distance": text_donor_distance,
                            "semantic_host_overlap": semantic_host_overlap,
                            "semantic_donor_overlap": semantic_donor_overlap,
                            "peak_drift": np.nan,
                            "peak_drift_relative": np.nan,
                            "end_drift": np.nan,
                            "end_drift_relative": np.nan,
                            "cad": np.nan,
                            "recovery_fraction": np.nan,
                            "persistence": np.nan,
                            "next_token_kl": np.nan,
                            "activation_norm_deviation": np.nan,
                            "verbal_donor_claim": verbal_claim,
                            "verbal_claim_confidence": verbal_confidence,
                            "host_completion_text": host_text,
                            "donor_completion_text": donor_text,
                            "graft_completion_text": graft_text,
                        }
                    )

                for control_kind in [kind for kind in control_kinds if kind != "name_only"]:
                    if control_kind == "primary":
                        donor_source_revision = donor_revision
                        donor_reference_revision = donor_revision
                        donor_source_prompt_id = prompt_id
                    elif control_kind == "adjacent":
                        if adjacent_revision is None:
                            continue
                        donor_source_revision = adjacent_revision
                        donor_reference_revision = adjacent_revision
                        donor_source_prompt_id = prompt_id
                    elif control_kind == "very_early":
                        if early_revision is None:
                            continue
                        donor_source_revision = early_revision
                        donor_reference_revision = early_revision
                        donor_source_prompt_id = prompt_id
                    elif control_kind == "shuffled_prompt":
                        donor_source_revision = donor_revision
                        donor_reference_revision = donor_revision
                        donor_source_prompt_id = shuffled_prompt_ids[prompt_id]
                    elif control_kind == "random_same_norm":
                        donor_source_revision = donor_revision
                        donor_reference_revision = donor_revision
                        donor_source_prompt_id = prompt_id
                    else:
                        raise ValueError(f"Unsupported control kind: {control_kind}")

                    donor_source_trace = clean_cache[(donor_source_revision, token_position, donor_source_prompt_id)]
                    donor_reference_trace = clean_cache[(donor_reference_revision, token_position, prompt_id)]

                    for layer_label, boundary_layer in zip(layer_labels, layer_indices):
                        for graft_mode in graft_modes:
                            mode_layers = _resolve_mode_layers(graft_mode, int(boundary_layer), host_loaded.n_layers)
                            if not mode_layers:
                                continue
                            for blend_lambda in lambdas:
                                grafts: list[ResidualGraft] = []
                                for layer_index in mode_layers:
                                    host_state = host_clean.per_layer_states[layer_index]
                                    donor_state = donor_source_trace.per_layer_states[layer_index]
                                    if control_kind == "random_same_norm":
                                        donor_state = _random_same_norm_state(
                                            host_state,
                                            donor_state,
                                            seed=_stable_seed(
                                                pair_name,
                                                prompt_id,
                                                token_position,
                                                layer_index,
                                                blend_lambda,
                                                graft_mode,
                                                control_kind,
                                            ),
                                        )
                                    grafts.append(
                                        ResidualGraft(
                                            layer_index=int(layer_index),
                                            donor_state=donor_state,
                                            blend_lambda=float(blend_lambda),
                                        )
                                    )

                                graft_trace = run_grafted_trace(
                                    loaded=host_loaded,
                                    prompt=prompt_text,
                                    max_prompt_tokens=max_prompt_tokens,
                                    token_position=token_position,
                                    generate_tokens=generate_tokens,
                                    grafts=grafts,
                                )
                                trajectory = compute_trajectory_metrics(
                                    baseline_states=host_clean.per_layer_states,
                                    injected_states=graft_trace.per_layer_states,
                                    baseline_logits=host_clean.next_token_logits,
                                    injected_logits=graft_trace.next_token_logits,
                                    inject_layer_index=int(min(mode_layers)),
                                    recovery_threshold=float(config.get("recovery_threshold", 0.25)),
                                )
                                donor_fraction, js_to_host, js_to_donor = _donor_identity_fraction(
                                    graft_trace.next_token_logits,
                                    host_clean.next_token_logits,
                                    donor_reference_trace.next_token_logits,
                                )
                                host_text = host_clean.completion_text or host_clean.generated_text
                                donor_text = donor_reference_trace.completion_text or donor_reference_trace.generated_text
                                graft_text = graft_trace.completion_text or graft_trace.generated_text
                                text_host_distance = stylometric_distance(graft_text, host_text)
                                text_donor_distance = stylometric_distance(graft_text, donor_text)
                                semantic_host_overlap = semantic_overlap(graft_text, host_text)
                                semantic_donor_overlap = semantic_overlap(graft_text, donor_text)
                                activation_norm_deviation = float(
                                    torch.linalg.norm(
                                        graft_trace.per_layer_states[mode_layers[-1]] - host_clean.per_layer_states[mode_layers[-1]]
                                    ).item()
                                )
                                verbal_claim = float("nan")
                                verbal_confidence = float("nan")
                                if run_self_report:
                                    choice = predict_labeled_choice(
                                        loaded=host_loaded,
                                        prompt=_self_report_prompt(
                                            question=prompt_text,
                                            completion=graft_text,
                                            host_revision=host_revision,
                                            donor_revision=donor_reference_revision,
                                        ),
                                        max_prompt_tokens=max_prompt_tokens,
                                        labels=[("A", "A", 0.0), ("B", "B", 1.0)],
                                        label_bias_prompt=_self_report_bias_prompt(),
                                    )
                                    verbal_claim = float(choice[2]) if choice[0] in {"A", "B"} else float("nan")
                                    verbal_confidence = float(choice[3]) if choice[0] in {"A", "B"} else float("nan")
                                rows.append(
                                    {
                                        "pair_name": pair_name,
                                        "model_id": model_id,
                                        "model_family": infer_model_family(model_id),
                                        "model_size_label": infer_model_size_label(model_id),
                                        "host_revision": host_revision,
                                        "donor_revision": donor_reference_revision,
                                        "donor_source_revision": donor_source_revision,
                                        "control_kind": control_kind,
                                        "graft_mode": graft_mode,
                                        "layer_label": layer_label,
                                        "boundary_layer_index": int(boundary_layer),
                                        "n_grafted_layers": int(len(mode_layers)),
                                        "blend_lambda": float(blend_lambda),
                                        "token_position": token_position,
                                        "token_position_label": _token_position_label(token_position),
                                        "prompt_id": prompt_id,
                                        "prompt_family": prompt_family,
                                        "prompt_rank": prompt_rank,
                                        "prompt_clean_js": prompt_js,
                                        "shuffled_prompt_id": donor_source_prompt_id if control_kind == "shuffled_prompt" else "",
                                        "donor_identity_fraction": donor_fraction,
                                        "js_to_host": js_to_host,
                                        "js_to_donor": js_to_donor,
                                        "text_donor_identity_fraction": _distance_based_identity_fraction(
                                            text_host_distance,
                                            text_donor_distance,
                                        ),
                                        "semantic_donor_identity_fraction": _distance_based_identity_fraction(
                                            1.0 - semantic_host_overlap,
                                            1.0 - semantic_donor_overlap,
                                        ),
                                        "text_host_distance": text_host_distance,
                                        "text_donor_distance": text_donor_distance,
                                        "semantic_host_overlap": semantic_host_overlap,
                                        "semantic_donor_overlap": semantic_donor_overlap,
                                        "peak_drift": trajectory.peak_drift,
                                        "peak_drift_relative": trajectory.peak_drift_relative,
                                        "end_drift": trajectory.end_drift,
                                        "end_drift_relative": trajectory.end_drift_relative,
                                        "cad": trajectory.drift_auc,
                                        "recovery_fraction": trajectory.recovery_fraction,
                                        "persistence": 1.0 - trajectory.recovery_fraction,
                                        "next_token_kl": trajectory.next_token_kl,
                                        "activation_norm_deviation": activation_norm_deviation,
                                        "verbal_donor_claim": verbal_claim,
                                        "verbal_claim_confidence": verbal_confidence,
                                        "host_completion_text": host_text,
                                        "donor_completion_text": donor_text,
                                        "graft_completion_text": graft_text,
                                    }
                                )
                                if checkpoint_every > 0 and len(rows) % checkpoint_every == 0:
                                    pd.DataFrame(rows).to_csv(partial_path, index=False)

        del host_loaded
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    _write_summary(
        df,
        ["pair_name", "control_kind", "graft_mode", "token_position_label", "layer_label", "blend_lambda"],
        output_dir / "summary_by_condition.csv",
    )
    _write_summary(
        df,
        ["pair_name", "control_kind", "graft_mode", "prompt_family"],
        output_dir / "summary_by_family.csv",
    )
    _write_summary(
        df,
        ["pair_name", "control_kind", "graft_mode", "token_position_label"],
        output_dir / "summary_by_token_position.csv",
    )

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as f:
        f.write("# Diachronic Ship-of-Theseus Identity Graft\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Model: `{model_id}`\n")
        f.write(f"- Selected prompt count: `{len(selected_items)}` from `{len(candidate_prompts)}` candidate prompts\n")
        f.write(f"- Prompt bank: `{prompt_bank_path}`\n")
        f.write(f"- Pairs: `{', '.join(str(pair['pair_name']) for pair in pair_specs)}`\n")
        f.write(f"- Token positions: `{token_positions}`\n")
        f.write(f"- Layer buckets: `{', '.join(layer_labels)}`\n")
        f.write(f"- Lambdas: `{lambdas}`\n")
        f.write(f"- Graft modes: `{graft_modes}`\n")
        f.write(
            "- Token position labels follow the current repo convention: "
            "`-1 = last prompt token`, `0 = first prompt token`.\n"
        )


if __name__ == "__main__":
    main()
