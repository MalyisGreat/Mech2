from __future__ import annotations

import argparse
from pathlib import Path

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
from temporal_authorship_matrix import (
    _generate_completions_batch,
    _generation_quality_metrics,
    _generation_quality_verdict,
    _load_prompt_bank,
    _revision_step,
    _score_outputs_batch,
    _stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the checkpoint age recognition v2 experiment.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _select_ranked_prompts(prompt_df: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count >= len(prompt_df):
        selected = prompt_df.copy()
    else:
        keyed = []
        for row in prompt_df.itertuples(index=False):
            key = _stable_seed(seed, row.prompt_id, row.prompt_family, row.prompt)
            keyed.append((float(-row.mean_anchor_js), key, row))
        keyed.sort(key=lambda item: (item[0], item[1]))
        selected = pd.DataFrame([item[2]._asdict() for item in keyed[:count]])
    selected = selected.reset_index(drop=True)
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    return selected


def _next_token_js_divergence_batch(
    loaded_a,
    loaded_b,
    prompts: list[str],
    *,
    max_prompt_tokens: int,
) -> list[float]:
    tokenizer = loaded_a.tokenizer
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_ids_a = encoded["input_ids"].to(loaded_a.device)
    attention_a = encoded["attention_mask"].to(loaded_a.device)
    input_ids_b = encoded["input_ids"].to(loaded_b.device)
    attention_b = encoded["attention_mask"].to(loaded_b.device)

    with torch.inference_mode():
        logits_a = loaded_a.model(
            input_ids=input_ids_a,
            attention_mask=attention_a,
            use_cache=False,
            return_dict=True,
        ).logits.float()
        logits_b = loaded_b.model(
            input_ids=input_ids_b,
            attention_mask=attention_b,
            use_cache=False,
            return_dict=True,
        ).logits.float()

    divergences: list[float] = []
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    for idx, prompt_len in enumerate(lengths):
        last_idx = max(0, int(prompt_len) - 1)
        p = torch.softmax(logits_a[idx, last_idx, :], dim=-1)
        q = torch.softmax(logits_b[idx, last_idx, :], dim=-1)
        m = 0.5 * (p + q)
        js = 0.5 * (
            torch.sum(p * (torch.log(p.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12))))
            + torch.sum(q * (torch.log(q.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12))))
        )
        divergences.append(float(js.item()))
    return divergences


def _write_pair_summary(df: pd.DataFrame, group_cols: list[str], path: Path) -> pd.DataFrame:
    add_src_to_path()
    from identity_stability.identity_analysis import bootstrap_mean_ci

    metric_cols = [
        "choose_anchor_centered",
        "choose_anchor_raw",
        "centered_margin_logprob",
        "raw_margin_logprob",
        "anchor_generation_valid",
        "comparison_generation_valid",
    ]
    rows: list[dict[str, object]] = []
    for keys, sub in df.groupby(group_cols, as_index=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["n"] = int(len(sub))
        for metric in metric_cols:
            values = sub[metric].dropna().astype(float).tolist() if metric in sub.columns else []
            row[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
            ci_low, ci_high = bootstrap_mean_ci(values, iters=1000, seed=123) if values else (float("nan"), float("nan"))
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(path, index=False)
    return summary


def main() -> None:
    add_src_to_path()
    from identity_stability.modeling import clear_cuda, load_model

    args = parse_args()
    config = load_yaml_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = ensure_output_dir(config, "checkpoint_age_recognition_v2")

    model_id = str(config["model_id"])
    anchor_revision = str(config["anchor_revision"])
    comparison_revisions = [str(revision) for revision in config["comparison_revisions"]]
    source_revisions = [anchor_revision, *comparison_revisions]
    evaluator_revisions = [str(revision) for revision in config.get("evaluator_revisions", [anchor_revision])]
    cache_dir = Path(str(config["model_cache_dir"]))
    dtype_name = str(config["dtype"])
    use_gpu = bool(config["use_gpu"])
    attention_backend = str(config.get("attention_backend", "auto"))
    max_prompt_tokens = int(config["max_prompt_tokens"])
    max_total_tokens = int(config.get("max_total_tokens", max_prompt_tokens + int(config["generation_tokens"])))
    generation_tokens = int(config["generation_tokens"])
    prompt_screen_batch_size = int(config.get("prompt_screen_batch_size", 8))
    generation_batch_size = int(config.get("generation_batch_size", 4))
    score_batch_size = int(config.get("score_batch_size", 4))
    checkpoint_every = int(config.get("checkpoint_every_rows", 0))
    quality_gate_enabled = bool(config.get("quality_gate_enabled", True))
    min_generation_token_count = int(config.get("min_generation_token_count", 24))
    min_unique_token_ratio = float(config.get("min_unique_token_ratio", 0.45))
    max_top_token_rate = float(config.get("max_top_token_rate", 0.25))
    max_top_bigram_rate = float(config.get("max_top_bigram_rate", 0.18))
    do_sample = bool(config.get("generation_do_sample", False))
    temperature = float(config.get("generation_temperature", 1.0))
    top_p = float(config.get("generation_top_p", 1.0))
    top_k = int(config.get("generation_top_k", 0))

    prompt_bank_path = repo_root / str(config["prompt_bank_path"])
    prompt_bank = _load_prompt_bank(prompt_bank_path)
    prompt_limit = int(config.get("prompt_limit", len(prompt_bank)))
    prompt_sample_seed = int(config.get("prompt_sample_seed", 7))
    selected_prompt_count = int(config.get("selected_prompt_count", prompt_limit))
    candidate_prompts = prompt_bank[:prompt_limit]

    anchor_loaded = load_model(
        model_id=model_id,
        cache_dir=cache_dir,
        dtype_name=dtype_name,
        use_gpu=use_gpu,
        attention_backend=attention_backend,
        revision=anchor_revision,
    )
    divergence_rows: list[dict[str, object]] = []
    for comparison_revision in comparison_revisions:
        comparison_loaded = load_model(
            model_id=model_id,
            cache_dir=cache_dir,
            dtype_name=dtype_name,
            use_gpu=use_gpu,
            attention_backend=attention_backend,
            revision=comparison_revision,
        )
        for start in range(0, len(candidate_prompts), max(1, prompt_screen_batch_size)):
            chunk = candidate_prompts[start : start + max(1, prompt_screen_batch_size)]
            js_values = _next_token_js_divergence_batch(
                anchor_loaded,
                comparison_loaded,
                [item["prompt"] for item in chunk],
                max_prompt_tokens=max_prompt_tokens,
            )
            for item, js_value in zip(chunk, js_values):
                divergence_rows.append(
                    {
                        "anchor_revision": anchor_revision,
                        "comparison_revision": comparison_revision,
                        "prompt_id": str(item["id"]),
                        "prompt_family": str(item["family"]),
                        "prompt": str(item["prompt"]),
                        "anchor_vs_comparison_js": float(js_value),
                    }
                )
        del comparison_loaded
        clear_cuda()

    divergence_df = pd.DataFrame(divergence_rows)
    divergence_df.to_csv(output_dir / "prompt_divergence.csv", index=False)
    prompt_scores = (
        divergence_df.groupby(["prompt_id", "prompt_family", "prompt"], as_index=False)
        .agg(
            mean_anchor_js=("anchor_vs_comparison_js", "mean"),
            min_anchor_js=("anchor_vs_comparison_js", "min"),
            max_anchor_js=("anchor_vs_comparison_js", "max"),
            comparison_count=("comparison_revision", "count"),
        )
    )
    prompt_scores.to_csv(output_dir / "prompt_divergence_summary.csv", index=False)
    selected_prompts_df = _select_ranked_prompts(prompt_scores, selected_prompt_count, prompt_sample_seed)
    selected_prompts_df.to_csv(output_dir / "selected_prompts.csv", index=False)
    selected_prompt_records = selected_prompts_df[["prompt_id", "prompt_family", "prompt"]].to_dict("records")

    generations_rows: list[dict[str, object]] = []
    generations_by_revision_prompt: dict[tuple[str, str], dict[str, object]] = {}
    for source_revision in source_revisions:
        loaded = load_model(
            model_id=model_id,
            cache_dir=cache_dir,
            dtype_name=dtype_name,
            use_gpu=use_gpu,
            attention_backend=attention_backend,
            revision=source_revision,
        )
        for start in range(0, len(selected_prompt_records), max(1, generation_batch_size)):
            chunk = selected_prompt_records[start : start + max(1, generation_batch_size)]
            completions = _generate_completions_batch(
                loaded,
                [item["prompt"] for item in chunk],
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=generation_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            for item, completion in zip(chunk, completions):
                quality_metrics = _generation_quality_metrics(str(completion))
                is_valid, invalid_reason = _generation_quality_verdict(
                    str(completion),
                    min_token_count=min_generation_token_count,
                    min_unique_token_ratio=min_unique_token_ratio,
                    max_top_token_rate=max_top_token_rate,
                    max_top_bigram_rate=max_top_bigram_rate,
                )
                row = {
                    "model_id": model_id,
                    "model_family": infer_model_family(model_id),
                    "model_size_label": infer_model_size_label(model_id),
                    "source_revision": source_revision,
                    "source_step": _revision_step(source_revision),
                    "prompt_id": str(item["prompt_id"]),
                    "prompt_family": str(item["prompt_family"]),
                    "prompt": str(item["prompt"]),
                    "completion_text": str(completion),
                    "completion_char_count": int(len(str(completion))),
                    "generation_valid": int(is_valid),
                    "generation_invalid_reason": invalid_reason,
                }
                row.update(quality_metrics)
                generations_rows.append(row)
                generations_by_revision_prompt[(source_revision, str(item["prompt_id"]))] = row
        del loaded
        clear_cuda()

    del anchor_loaded
    clear_cuda()

    generations_df = pd.DataFrame(generations_rows)
    generations_df.to_csv(output_dir / "generations.csv", index=False)
    generation_quality_summary = (
        generations_df.groupby("source_revision", as_index=False)
        .agg(
            generation_valid_rate=("generation_valid", "mean"),
            unique_token_ratio_mean=("unique_token_ratio", "mean"),
            top_token_rate_mean=("top_token_rate", "mean"),
            top_bigram_rate_mean=("top_bigram_rate", "mean"),
            n=("prompt_id", "count"),
        )
    )
    generation_quality_summary.to_csv(output_dir / "generation_quality_summary.csv", index=False)

    source_score_rows: list[dict[str, object]] = []
    partial_path = output_dir / "source_scores.partial.csv"
    for evaluator_revision in evaluator_revisions:
        loaded = load_model(
            model_id=model_id,
            cache_dir=cache_dir,
            dtype_name=dtype_name,
            use_gpu=use_gpu,
            attention_backend=attention_backend,
            revision=evaluator_revision,
        )
        batch_pairs: list[tuple[str, str]] = []
        batch_meta: list[dict[str, object]] = []
        for prompt_item in selected_prompt_records:
            prompt_id = str(prompt_item["prompt_id"])
            for source_revision in source_revisions:
                source_row = generations_by_revision_prompt[(source_revision, prompt_id)]
                if quality_gate_enabled and int(source_row.get("generation_valid", 0)) != 1:
                    continue
                batch_pairs.append((str(prompt_item["prompt"]), str(source_row["completion_text"])))
                batch_meta.append(
                    {
                        "evaluator_revision": evaluator_revision,
                        "evaluator_step": _revision_step(evaluator_revision),
                        "source_revision": source_revision,
                        "source_step": _revision_step(source_revision),
                        "prompt_id": prompt_id,
                        "prompt_family": str(prompt_item["prompt_family"]),
                        "prompt": str(prompt_item["prompt"]),
                        "completion_text": str(source_row["completion_text"]),
                        "generation_valid": int(source_row["generation_valid"]),
                        "generation_invalid_reason": str(source_row["generation_invalid_reason"]),
                    }
                )
                if len(batch_pairs) >= max(1, score_batch_size):
                    scores = _score_outputs_batch(loaded, batch_pairs, max_total_tokens=max_total_tokens)
                    for meta, score in zip(batch_meta, scores):
                        row = {**meta, **score}
                        row["temporal_distance"] = abs(int(row["evaluator_step"]) - int(row["source_step"]))
                        source_score_rows.append(row)
                    batch_pairs = []
                    batch_meta = []
                    if checkpoint_every > 0 and len(source_score_rows) % checkpoint_every == 0:
                        pd.DataFrame(source_score_rows).to_csv(partial_path, index=False)
        if batch_pairs:
            scores = _score_outputs_batch(loaded, batch_pairs, max_total_tokens=max_total_tokens)
            for meta, score in zip(batch_meta, scores):
                row = {**meta, **score}
                row["temporal_distance"] = abs(int(row["evaluator_step"]) - int(row["source_step"]))
                source_score_rows.append(row)
        del loaded
        clear_cuda()

    source_scores_df = pd.DataFrame(source_score_rows)
    if source_scores_df.empty:
        raise RuntimeError("Checkpoint age recognition v2 produced no source scores.")

    source_scores_df["prompt_centered_avg_logprob"] = (
        source_scores_df["avg_logprob"]
        - source_scores_df.groupby(["evaluator_revision", "prompt_id"])["avg_logprob"].transform("mean")
    )
    source_scores_df.to_csv(output_dir / "source_scores.csv", index=False)

    pair_rows: list[dict[str, object]] = []
    for (evaluator_revision, prompt_id), sub in source_scores_df.groupby(["evaluator_revision", "prompt_id"], as_index=False):
        anchor_rows = sub[sub["source_revision"] == anchor_revision]
        if anchor_rows.empty:
            continue
        anchor_row = anchor_rows.iloc[0]
        for comparison_revision in comparison_revisions:
            comparison_rows = sub[sub["source_revision"] == comparison_revision]
            if comparison_rows.empty:
                continue
            comparison_row = comparison_rows.iloc[0]
            raw_margin = float(anchor_row["avg_logprob"] - comparison_row["avg_logprob"])
            centered_margin = float(anchor_row["prompt_centered_avg_logprob"] - comparison_row["prompt_centered_avg_logprob"])
            pair_rows.append(
                {
                    "model_id": model_id,
                    "model_family": infer_model_family(model_id),
                    "model_size_label": infer_model_size_label(model_id),
                    "evaluator_revision": str(evaluator_revision),
                    "evaluator_step": int(anchor_row["evaluator_step"]),
                    "anchor_revision": anchor_revision,
                    "anchor_step": _revision_step(anchor_revision),
                    "comparison_revision": comparison_revision,
                    "comparison_step": _revision_step(comparison_revision),
                    "prompt_id": str(prompt_id),
                    "prompt_family": str(anchor_row["prompt_family"]),
                    "prompt": str(anchor_row["prompt"]),
                    "temporal_distance": abs(_revision_step(anchor_revision) - _revision_step(comparison_revision)),
                    "anchor_avg_logprob": float(anchor_row["avg_logprob"]),
                    "comparison_avg_logprob": float(comparison_row["avg_logprob"]),
                    "anchor_centered_avg_logprob": float(anchor_row["prompt_centered_avg_logprob"]),
                    "comparison_centered_avg_logprob": float(comparison_row["prompt_centered_avg_logprob"]),
                    "raw_margin_logprob": raw_margin,
                    "centered_margin_logprob": centered_margin,
                    "choose_anchor_raw": int(raw_margin > 0.0),
                    "choose_anchor_centered": int(centered_margin > 0.0),
                    "anchor_generation_valid": int(anchor_row["generation_valid"]),
                    "comparison_generation_valid": int(comparison_row["generation_valid"]),
                }
            )

    results_df = pd.DataFrame(pair_rows)
    if results_df.empty:
        raise RuntimeError("Checkpoint age recognition v2 produced no pairwise results.")
    results_df.to_csv(output_dir / "results.csv", index=False)

    _write_pair_summary(results_df, ["evaluator_revision", "anchor_revision", "comparison_revision"], output_dir / "summary_by_pair.csv")
    _write_pair_summary(results_df, ["comparison_revision"], output_dir / "summary_by_comparison.csv")
    _write_pair_summary(results_df, ["temporal_distance"], output_dir / "summary_by_distance.csv")
    _write_pair_summary(results_df, ["evaluator_revision"], output_dir / "summary_by_evaluator.csv")

    overall_summary = pd.DataFrame(
        [
            {
                "metric": "checkpoint_age_centered_choice_rate",
                "mean": float(results_df["choose_anchor_centered"].mean()),
                "n": int(len(results_df)),
            },
            {
                "metric": "checkpoint_age_raw_choice_rate",
                "mean": float(results_df["choose_anchor_raw"].mean()),
                "n": int(len(results_df)),
            },
            {
                "metric": "checkpoint_age_centered_margin_logprob",
                "mean": float(results_df["centered_margin_logprob"].mean()),
                "n": int(len(results_df)),
            },
            {
                "metric": "checkpoint_age_raw_margin_logprob",
                "mean": float(results_df["raw_margin_logprob"].mean()),
                "n": int(len(results_df)),
            },
        ]
    )
    overall_summary.to_csv(output_dir / "summary.csv", index=False)

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as handle:
        handle.write("# Checkpoint Age Recognition V2\n\n")
        handle.write(f"- Config: `{args.config}`\n")
        handle.write(f"- Model: `{model_id}`\n")
        handle.write(f"- Anchor revision: `{anchor_revision}`\n")
        handle.write(f"- Comparison revisions: `{', '.join(comparison_revisions)}`\n")
        handle.write(f"- Evaluator revisions: `{', '.join(evaluator_revisions)}`\n")
        handle.write(f"- Prompt bank: `{prompt_bank_path}`\n")
        handle.write(f"- Selected prompt count: `{len(selected_prompt_records)}` from `{len(candidate_prompts)}` candidate prompts\n")
        handle.write(f"- Generation tokens: `{generation_tokens}`\n")
        handle.write(
            f"- Quality gate: `enabled={quality_gate_enabled}`, `min_tokens={min_generation_token_count}`, "
            f"`min_unique_ratio={min_unique_token_ratio}`, `max_top_token_rate={max_top_token_rate}`, "
            f"`max_top_bigram_rate={max_top_bigram_rate}`\n"
        )
        handle.write(f"- Sampling: `do_sample={do_sample}`, `temperature={temperature}`, `top_p={top_p}`, `top_k={top_k}`\n")


if __name__ == "__main__":
    main()
