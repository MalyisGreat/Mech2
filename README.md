# Joseph Stroud Research Project: Identity Stability Under Internal Redirection

## Objective
Evaluate whether larger language models preserve internal trajectory identity more strongly under controlled residual-stream concept-direction injections.

## Success Criteria
1. Produce an in-depth literature-backed research brief with citations.
2. Decompose the thesis into executable workstreams with clear ownership.
3. Define a concrete experimental protocol for drift, recovery, and rebound.
4. Define an analysis plan for scale effects and fine-tuning method effects (full FT vs LoRA).
5. Maintain transparent logs of every research action and synthesis step.

## Deliverables
1. Task decomposition and coordination log.
2. Source registry with links and relevance notes.
3. Workstream findings files.
4. Integrated final synthesis and recommended execution plan.

## Experiment Code
Code lives in `src/identity_stability` and run scripts live in `scripts`.

### Main scripts
1. `scripts/download_models.py`: download model checkpoints into cache.
2. `scripts/run_experiment.py`: run experiments from a YAML config.
3. `scripts/run_full_pipeline.py`: download configured models and run end-to-end.
4. `scripts/run_research_suite.py`: run multi-concept multi-seed suites.
5. `scripts/analyze_research_suite.py`: aggregate suite outputs and compute stratified summaries.
6. `scripts/summarize_run.py`: summarize a single run directory.
7. `scripts/run_prior_findings_addon.py`: run add-on suite for prior findings (good/evil style concepts, threshold sweep, prompt styles, early-vs-late token position).
8. `scripts/run_max_thorough_suite.py`: run the maximum-density long suite across all concepts, multiple seeds, and token positions.

### Configs
1. `configs/pilot.yaml`: fastest real run for validation.
2. `configs/default.yaml`: broader run across more model sizes.
3. `configs/extended_download.yaml`: includes larger checkpoints for downloading.
4. `configs/current_models_full.yaml`: full sweep across current six-model set.
5. `configs/research_suite_base.yaml`: base config for multi-concept suite studies.
6. `configs/prior_findings_addon.yaml`: add-on configuration for threshold and prompt-style analyses.
7. `configs/final_models_h100_fast.yaml`: final-model panel (Qwen2.5 + Qwen3 + GPT-2) tuned for fast H100 runs.
8. `configs/final_models_h100_extended.yaml`: extended final-model panel adding 14B checkpoints.
9. `configs/smoke_arch_compat.yaml`: minimal architecture compatibility smoke test.
10. `configs/final_models_h200_max_thorough.yaml`: max-density H200/H100 config (post-prune, capped at Qwen/GPT-2 models up to 8B).

### Typical usage
```powershell
python scripts/download_models.py --cache-dir D:/hf-model-cache --workers 4 --models EleutherAI/pythia-70m EleutherAI/pythia-160m
python scripts/run_experiment.py --config configs/pilot.yaml
python scripts/run_prior_findings_addon.py --config configs/prior_findings_addon.yaml
python scripts/run_prior_findings_addon.py --config configs/prior_findings_addon.yaml --gpus 0 1 2 3
python scripts/run_experiment.py --config configs/final_models_h100_fast.yaml
python scripts/run_experiment.py --config configs/final_models_h100_fast.yaml --gpus 0 1 2 3
```

Download from model-family config:
```powershell
python scripts/download_models.py --cache-dir D:/hf-model-cache --config configs/download_family_qwen3_5_h100.yaml
```

Download only max-around-30B Qwen models (2.5, 3, 3.5):
```powershell
python scripts/download_models.py --cache-dir D:/hf-model-cache --config configs/download_qwen_max_around_30b.yaml
```

Run the full max-thorough suite:
```powershell
python scripts/run_max_thorough_suite.py --config configs/final_models_h200_max_thorough.yaml --seeds 42 43 44 45 46 --token-positions -1 0 1 2 --suite-name max_thorough_v1
python scripts/run_max_thorough_suite.py --config configs/final_models_h200_max_thorough.yaml --seeds 42 43 44 45 46 --token-positions -1 0 1 2 --suite-name max_thorough_v1 --gpus 0 1 2 3
python scripts/analyze_research_suite.py --manifest runs/max_thorough_v1_<timestamp>/suite_manifest.csv --bootstrap-iters 2000
```

Run outputs are written to `runs/<timestamp>/`.
Multi-GPU runs produce orchestrator folders like `runs/multi_gpu_experiment_<timestamp>/` and a consolidated result in `merged_run/`.

### Config note
- `token_position`: token index used for tracing and injection.
- `estimation_token_position` (optional): token index used to estimate concept vectors. If omitted, defaults to `token_position`.
- `trace_batch_size`: batched prompt tracing size for baseline/injected passes.
- `activation_batch_size`: batch size for concept-vector activation extraction.
- `adaptive_batching`: if `true`, automatically halves batch size on OOM and retries.
- `attention_backend`: `auto`, `sdpa`, `flash_attention_2`, or `default`.
- `enable_tf32`: enables TF32 kernels on CUDA for faster matmul on H100/A100.
- `layer_topk_tokens`: optional logit-lens top-k token count per hidden-state layer (`0` disables).
- `layer_topk_prompt_limit`: number of prompts per trace batch for layer top-k logging.
- `--gpus` (run scripts): optional GPU IDs. With multiple IDs, models are sharded across workers and merged.
