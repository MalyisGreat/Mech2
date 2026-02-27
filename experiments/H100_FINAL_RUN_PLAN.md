# H100 Final Run Plan (Qwen2.5 + Qwen3 + GPT-2)

## Final Model Panel

### Fast Tier (recommended default)
Config: `configs/final_models_h100_fast.yaml`

Models:
1. `Qwen/Qwen2.5-0.5B-Instruct`
2. `Qwen/Qwen2.5-1.5B-Instruct`
3. `Qwen/Qwen2.5-3B-Instruct`
4. `Qwen/Qwen2.5-7B-Instruct`
5. `Qwen/Qwen3-0.6B`
6. `Qwen/Qwen3-1.7B`
7. `Qwen/Qwen3-4B`
8. `Qwen/Qwen3-8B`
9. `gpt2`
10. `gpt2-medium`
11. `gpt2-large`
12. `gpt2-xl`

### Extended Tier (optional, slower)
Config: `configs/final_models_h100_extended.yaml`

Adds:
1. `Qwen/Qwen2.5-14B-Instruct`
2. `Qwen/Qwen3-14B`

### Max Thorough Tier (long-form final run)
Config: `configs/final_models_h200_max_thorough.yaml`

Adds high-density sweeps:
1. Qwen 32B checkpoints (`Qwen2.5-32B`, `Qwen3-32B`)
2. Qwen3.5 capped model (`Qwen3.5-35B-A3B`)
3. Denser layer positions and alpha grid
4. Larger prompt sets and `random_control` method

## Run Sequence on H100

1. Pre-download models:
```powershell
python scripts/download_models.py --cache-dir D:/hf-model-cache --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B gpt2 gpt2-medium gpt2-large gpt2-xl
```

Optional Qwen 3.5 family download:
```powershell
python scripts/download_models.py --cache-dir D:/hf-model-cache --config configs/download_family_qwen3_5_h100.yaml
```

Strict max-around-30B per Qwen family:
```powershell
python scripts/download_models.py --cache-dir D:/hf-model-cache --config configs/download_qwen_max_around_30b.yaml
```

For faster prefetch on strong network links:
```powershell
python scripts/download_models.py --cache-dir D:/hf-model-cache --workers 4 --config configs/download_qwen_max_around_30b.yaml
```

2. Smoke compatibility check:
```powershell
python scripts/run_experiment.py --config configs/smoke_arch_compat.yaml --models gpt2 Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-0.6B
```

3. Run final suite:
```powershell
python scripts/run_research_suite.py --config configs/final_models_h100_fast.yaml --concepts morality constructiveness politeness empathy skepticism safety --seeds 42 43 --suite-name final_h100_fast
```

4. Analyze final suite:
```powershell
python scripts/analyze_research_suite.py --manifest runs/final_h100_fast_<timestamp>/suite_manifest.csv --bootstrap-iters 500
```

5. Max thorough long-form suite (whole-suite progress % is printed):
```powershell
python scripts/download_models.py --cache-dir D:/hf-model-cache --workers 4 --config configs/final_models_h200_max_thorough.yaml
python scripts/run_max_thorough_suite.py --config configs/final_models_h200_max_thorough.yaml --seeds 42 43 44 45 46 --token-positions -1 0 1 2 --suite-name max_thorough_v1
python scripts/analyze_research_suite.py --manifest runs/max_thorough_v1_<timestamp>/suite_manifest.csv --bootstrap-iters 2000
```

## Notes

1. `estimation_token_position` is pinned at `-1` in final configs to avoid degenerate concept vectors at token position `0`.
2. GPT-2 support is integrated in the same layer-hook path; no separate codepath is required.
3. H100-tuned configs use:
   - `dtype: bfloat16`
   - `attention_backend: auto` (prefers SDPA)
   - `enable_tf32: true`
   - batched tracing/activation with adaptive OOM fallback
4. If runtime is too high, reduce:
   - concepts (first),
   - seeds (second),
   - then add extended 14B models only after the fast tier is stable.
