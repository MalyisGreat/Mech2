from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the temporal authorship matrix experiment.")
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


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _select_prompts(prompt_bank: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    if count >= len(prompt_bank):
        return list(prompt_bank)
    keyed = []
    for item in prompt_bank:
        key = _stable_seed(seed, item["id"], item["family"], item["prompt"])
        keyed.append((key, item))
    keyed.sort(key=lambda pair: pair[0])
    return [item for _, item in keyed[:count]]


def _generate_completions_batch(
    loaded,
    prompts: list[str],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[str]:
    tokenizer = loaded.tokenizer
    model = loaded.model
    device = loaded.device

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_len = int(encoded["input_ids"].shape[1])

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)
        if int(top_k) > 0:
            generation_kwargs["top_k"] = int(top_k)

    with torch.inference_mode():
        generated = model.generate(**encoded, **generation_kwargs)

    completions: list[str] = []
    for row in generated:
        completion_ids = row[input_len:]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    return completions


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _generation_quality_metrics(text: str) -> dict[str, float | int | str]:
    tokens = [tok.lower() for tok in WORD_RE.findall(text)]
    token_count = len(tokens)
    unique_ratio = float(len(set(tokens)) / max(1, token_count))
    token_counter = Counter(tokens)
    top_token_rate = float(max(token_counter.values()) / max(1, token_count)) if token_counter else 1.0
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_counter = Counter(bigrams)
    top_bigram_rate = float(max(bigram_counter.values()) / max(1, len(bigrams))) if bigram_counter else 0.0
    return {
        "generation_token_count": int(token_count),
        "unique_token_ratio": unique_ratio,
        "top_token_rate": top_token_rate,
        "top_bigram_rate": top_bigram_rate,
    }


def _generation_quality_verdict(
    text: str,
    *,
    min_token_count: int,
    min_unique_token_ratio: float,
    max_top_token_rate: float,
    max_top_bigram_rate: float,
) -> tuple[bool, str]:
    metrics = _generation_quality_metrics(text)
    token_count = int(metrics["generation_token_count"])
    unique_ratio = float(metrics["unique_token_ratio"])
    top_token_rate = float(metrics["top_token_rate"])
    top_bigram_rate = float(metrics["top_bigram_rate"])
    if token_count < int(min_token_count):
        return False, "too_short"
    if unique_ratio < float(min_unique_token_ratio):
        return False, "low_unique_token_ratio"
    if top_token_rate > float(max_top_token_rate):
        return False, "high_top_token_rate"
    if top_bigram_rate > float(max_top_bigram_rate):
        return False, "high_top_bigram_rate"
    return True, ""


def _score_outputs_batch(
    loaded,
    prompt_completion_pairs: list[tuple[str, str]],
    *,
    max_total_tokens: int,
) -> list[dict[str, float]]:
    tokenizer = loaded.tokenizer
    model = loaded.model
    device = loaded.device

    prompts = [pair[0] for pair in prompt_completion_pairs]
    completions = [pair[1] for pair in prompt_completion_pairs]
    full_texts = [prompt + completion for prompt, completion in prompt_completion_pairs]

    prompt_enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_total_tokens,
    )
    full_enc = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_total_tokens,
    )
    prompt_lengths = prompt_enc["attention_mask"].sum(dim=1).tolist()
    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    full_lengths = attention_mask.sum(dim=1).tolist()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)

    rows: list[dict[str, float]] = []
    for idx in range(len(prompt_completion_pairs)):
        prompt_len = int(prompt_lengths[idx])
        full_len = int(full_lengths[idx])
        answer_token_count = max(0, full_len - prompt_len)
        if answer_token_count <= 0:
            rows.append(
                {
                    "avg_logprob": float("nan"),
                    "sum_logprob": float("nan"),
                    "perplexity": float("nan"),
                    "answer_token_count": 0,
                }
            )
            continue
        token_start = max(0, prompt_len - 1)
        token_end = full_len - 1
        target_ids = input_ids[idx, prompt_len:full_len]
        pred_log_probs = log_probs[idx, token_start:token_end, :]
        gathered = pred_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        sum_logprob = float(gathered.sum().item())
        avg_logprob = float(gathered.mean().item())
        perplexity = float(np.exp(-avg_logprob))
        rows.append(
            {
                "avg_logprob": avg_logprob,
                "sum_logprob": sum_logprob,
                "perplexity": perplexity,
                "answer_token_count": int(answer_token_count),
            }
        )
    return rows


def _revision_step(revision: str) -> int:
    digits = "".join(ch for ch in str(revision) if ch.isdigit())
    return int(digits) if digits else -1


def _write_summary(df: pd.DataFrame, group_cols: list[str], path: Path) -> pd.DataFrame:
    add_src_to_path()
    from identity_stability.identity_analysis import bootstrap_mean_ci

    metric_cols = [
        "avg_logprob",
        "perplexity",
        "self_preferred",
        "distance_from_diagonal",
        "diagonal_margin_logprob",
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
    output_dir = ensure_output_dir(config, "temporal_authorship_matrix")
    checkpoint_every = int(config.get("checkpoint_every_rows", 0))

    model_id = str(config["model_id"])
    revisions = [str(revision) for revision in config["revisions"]]
    cache_dir = Path(str(config["model_cache_dir"]))
    dtype_name = str(config["dtype"])
    use_gpu = bool(config["use_gpu"])
    attention_backend = str(config.get("attention_backend", "auto"))
    max_prompt_tokens = int(config["max_prompt_tokens"])
    max_total_tokens = int(config.get("max_total_tokens", max_prompt_tokens + int(config["generation_tokens"])))
    generation_tokens = int(config["generation_tokens"])
    generation_batch_size = int(config.get("generation_batch_size", 4))
    score_batch_size = int(config.get("score_batch_size", 4))
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
    selected_prompt_count = int(config.get("selected_prompt_count", prompt_limit))
    prompt_sample_seed = int(config.get("prompt_sample_seed", 7))
    candidate_prompts = prompt_bank[:prompt_limit]
    selected_prompts = _select_prompts(candidate_prompts, selected_prompt_count, prompt_sample_seed)

    selected_df = pd.DataFrame(
        [
            {
                "prompt_id": item["id"],
                "prompt_family": item["family"],
                "prompt": item["prompt"],
                "selection_rank": idx + 1,
            }
            for idx, item in enumerate(selected_prompts)
        ]
    )
    selected_df.to_csv(output_dir / "selected_prompts.csv", index=False)

    generations_rows: list[dict[str, object]] = []
    generations_by_revision_prompt: dict[tuple[str, str], dict[str, object]] = {}
    for revision in revisions:
        loaded = load_model(
            model_id=model_id,
            cache_dir=cache_dir,
            dtype_name=dtype_name,
            use_gpu=use_gpu,
            attention_backend=attention_backend,
            revision=revision,
        )
        for start in range(0, len(selected_prompts), max(1, generation_batch_size)):
            chunk = selected_prompts[start : start + max(1, generation_batch_size)]
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
                    "source_revision": revision,
                    "source_step": _revision_step(revision),
                    "prompt_id": str(item["id"]),
                    "prompt_family": str(item["family"]),
                    "prompt": str(item["prompt"]),
                    "completion_text": str(completion),
                    "completion_char_count": int(len(str(completion))),
                    "generation_valid": int(is_valid),
                    "generation_invalid_reason": invalid_reason,
                }
                row.update(quality_metrics)
                generations_rows.append(row)
                generations_by_revision_prompt[(revision, str(item["id"]))] = row
        del loaded
        clear_cuda()

    generations_df = pd.DataFrame(generations_rows)
    generations_df.to_csv(output_dir / "generations.csv", index=False)
    quality_summary = (
        generations_df.groupby("source_revision", as_index=False)
        .agg(
            generation_valid_rate=("generation_valid", "mean"),
            unique_token_ratio_mean=("unique_token_ratio", "mean"),
            top_token_rate_mean=("top_token_rate", "mean"),
            top_bigram_rate_mean=("top_bigram_rate", "mean"),
            n=("prompt_id", "count"),
        )
    )
    quality_summary.to_csv(output_dir / "generation_quality_summary.csv", index=False)

    score_rows: list[dict[str, object]] = []
    partial_path = output_dir / "scores.partial.csv"
    for evaluator_revision in revisions:
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
        for prompt_item in selected_prompts:
            prompt_id = str(prompt_item["id"])
            for source_revision in revisions:
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
                        "prompt_family": str(prompt_item["family"]),
                        "prompt": str(prompt_item["prompt"]),
                        "completion_text": str(source_row["completion_text"]),
                    }
                )
                if len(batch_pairs) >= max(1, score_batch_size):
                    scores = _score_outputs_batch(loaded, batch_pairs, max_total_tokens=max_total_tokens)
                    for meta, score in zip(batch_meta, scores):
                        row = {**meta, **score}
                        row["temporal_distance"] = abs(int(row["evaluator_step"]) - int(row["source_step"]))
                        row["same_revision"] = int(str(row["evaluator_revision"]) == str(row["source_revision"]))
                        score_rows.append(row)
                    batch_pairs = []
                    batch_meta = []
                    if checkpoint_every > 0 and len(score_rows) % checkpoint_every == 0:
                        pd.DataFrame(score_rows).to_csv(partial_path, index=False)
        if batch_pairs:
            scores = _score_outputs_batch(loaded, batch_pairs, max_total_tokens=max_total_tokens)
            for meta, score in zip(batch_meta, scores):
                row = {**meta, **score}
                row["temporal_distance"] = abs(int(row["evaluator_step"]) - int(row["source_step"]))
                row["same_revision"] = int(str(row["evaluator_revision"]) == str(row["source_revision"]))
                score_rows.append(row)
        del loaded
        clear_cuda()

    scores_df = pd.DataFrame(score_rows)
    if scores_df.empty:
        raise RuntimeError("Temporal authorship matrix produced no scores.")

    winners = []
    grouped = scores_df.groupby(["evaluator_revision", "prompt_id"], as_index=False)
    for _, sub in grouped:
        sub = sub.sort_values("avg_logprob", ascending=False).reset_index(drop=True)
        best = sub.iloc[0]
        own = sub[sub["same_revision"] == 1]
        own_score = float(own["avg_logprob"].iloc[0]) if not own.empty else float("nan")
        best_nonself = sub[sub["same_revision"] == 0]["avg_logprob"].max() if (sub["same_revision"] == 0).any() else float("nan")
        for row in sub.itertuples(index=False):
            winners.append(
                {
                    **row._asdict(),
                    "winner_source_revision": str(best.source_revision),
                    "winner_avg_logprob": float(best.avg_logprob),
                    "self_preferred": int(str(best.source_revision) == str(row.evaluator_revision)),
                    "distance_from_diagonal": abs(int(row.evaluator_step) - int(row.source_step)),
                    "diagonal_margin_logprob": float(own_score - best_nonself) if np.isfinite(own_score) and np.isfinite(best_nonself) else float("nan"),
                }
            )
    results_df = pd.DataFrame(winners)
    results_df.to_csv(output_dir / "results.csv", index=False)

    preference_matrix = (
        results_df.groupby(["evaluator_revision", "source_revision"], as_index=False)["avg_logprob"]
        .mean(numeric_only=True)
        .pivot(index="evaluator_revision", columns="source_revision", values="avg_logprob")
        .reindex(index=revisions, columns=revisions)
    )
    preference_matrix.to_csv(output_dir / "authorship_preference_matrix.csv")

    self_pref_rows = []
    for evaluator_revision, sub in results_df.groupby("evaluator_revision", as_index=False):
        prompt_winners = (
            sub.groupby("prompt_id", as_index=False)
            .first()[["prompt_id", "winner_source_revision", "diagonal_margin_logprob"]]
        )
        self_pref_rows.append(
            {
                "evaluator_revision": evaluator_revision,
                "evaluator_step": _revision_step(str(evaluator_revision)),
                "self_preference_rate": float(
                    np.mean(prompt_winners["winner_source_revision"].astype(str) == str(evaluator_revision))
                ),
                "mean_diagonal_margin_logprob": float(prompt_winners["diagonal_margin_logprob"].dropna().astype(float).mean())
                if prompt_winners["diagonal_margin_logprob"].notna().any()
                else float("nan"),
                "prompt_count": int(prompt_winners["prompt_id"].nunique()),
            }
        )
    pd.DataFrame(self_pref_rows).to_csv(output_dir / "self_preference_summary.csv", index=False)

    _write_summary(results_df, ["evaluator_revision", "source_revision"], output_dir / "summary_by_pair.csv")
    _write_summary(results_df, ["evaluator_revision"], output_dir / "summary_by_evaluator.csv")
    _write_summary(results_df, ["source_revision"], output_dir / "summary_by_source.csv")

    with (output_dir / "run_manifest.md").open("w", encoding="utf-8") as handle:
        handle.write("# Temporal Authorship Matrix\n\n")
        handle.write(f"- Config: `{args.config}`\n")
        handle.write(f"- Model: `{model_id}`\n")
        handle.write(f"- Revisions: `{', '.join(revisions)}`\n")
        handle.write(f"- Prompt bank: `{prompt_bank_path}`\n")
        handle.write(f"- Selected prompt count: `{len(selected_prompts)}` from `{len(candidate_prompts)}` candidate prompts\n")
        handle.write(f"- Generation tokens: `{generation_tokens}`\n")
        handle.write(
            f"- Quality gate: `enabled={quality_gate_enabled}`, `min_tokens={min_generation_token_count}`, "
            f"`min_unique_ratio={min_unique_token_ratio}`, `max_top_token_rate={max_top_token_rate}`, "
            f"`max_top_bigram_rate={max_top_bigram_rate}`\n"
        )
        handle.write(f"- Sampling: `do_sample={do_sample}`, `temperature={temperature}`, `top_p={top_p}`, `top_k={top_k}`\n")


if __name__ == "__main__":
    main()
