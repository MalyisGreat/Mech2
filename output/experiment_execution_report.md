# Experiment Execution Report

## Date
2026-02-27 (local system date context for this session)

## Implementation Summary
Implemented full runnable pipeline:
1. `src/identity_stability/config.py`: YAML config loader.
2. `src/identity_stability/prompt_bank.py`: concept prompt generation.
3. `src/identity_stability/modeling.py`: model/tokenizer loading and device control.
4. `src/identity_stability/vectors.py`: mean-difference and linear-probe vector estimation.
5. `src/identity_stability/intervention.py`: residual-layer injection and trajectory capture.
6. `src/identity_stability/metrics.py`: drift/recovery/rebound and KL metrics.
7. `src/identity_stability/experiment.py`: end-to-end sweep orchestration and artifact writing.

Scripts:
1. `scripts/download_models.py`
2. `scripts/run_experiment.py`
3. `scripts/run_full_pipeline.py`

Configs:
1. `configs/pilot.yaml`
2. `configs/default.yaml`
3. `configs/extended_download.yaml`

## Models Downloaded
Cache root: `D:/hf-model-cache`

1. `EleutherAI/pythia-70m` (~0.31 GB)
2. `EleutherAI/pythia-160m` (~0.70 GB)
3. `EleutherAI/pythia-410m` (~1.70 GB)
4. `EleutherAI/pythia-1b` (~3.90 GB)
5. `EleutherAI/pythia-1.4b` (~5.46 GB)
6. `EleutherAI/pythia-2.8b` (~10.59 GB)

Total downloaded footprint for these six checkpoints: ~22.66 GB.

## Runs Executed
1. Pilot (2 models): `runs/20260226_195055`
2. Default sweep (4 models, 3 layer positions, 4 alphas): `runs/20260226_200433`
3. Extended sweep (6 models including 1.4b/2.8b): `runs/20260226_203103`
4. Full current-model sweep (6 models, 3 layer positions, 4 alphas): `runs/20260226_204021`

Runtime logs:
1. `logs/runtime_download_pilot.log`
2. `logs/runtime_download_default.log`
3. `logs/runtime_download_extended.log`
4. `logs/runtime_experiment_pilot.log`
5. `logs/runtime_experiment_pilot_rerun.log`
6. `logs/runtime_experiment_pilot_clean.log`
7. `logs/runtime_experiment_default.log`
8. `logs/runtime_experiment_extended.log`

## Output Artifacts
Each run contains:
1. `resolved_config.json`
2. `prompt_set.json`
3. `metrics_full.csv`
4. `metrics_summary.csv`
5. `quick_report.md`
6. `failures.json`
7. Per-model baseline generations and model info files.

Additional generated summaries:
1. `runs/20260226_200433/model_summary.csv`
2. `runs/20260226_200433/alpha_summary.csv`
3. `runs/20260226_200433/summary_generated.md`
4. `runs/20260226_203103/model_summary.csv`
5. `runs/20260226_203103/alpha_summary.csv`
6. `runs/20260226_203103/summary_generated.md`
7. `runs/20260226_204021/model_summary.csv`
8. `runs/20260226_204021/alpha_summary.csv`
9. `runs/20260226_204021/summary_generated.md`

## Key Findings Snapshot (Full Current-Model Run `20260226_204021`)
Aggregate means across all conditions:

1. `pythia-70m`: peak drift 9.3507, recovery 0.0000
2. `pythia-160m`: peak drift 7.9298, recovery 0.0000
3. `pythia-410m`: peak drift 4.4785, recovery 0.0003
4. `pythia-1b`: peak drift 2.6733, recovery 0.2566
5. `pythia-1.4b`: peak drift 2.5212, recovery 0.2140
6. `pythia-2.8b`: peak drift 2.9183, recovery 0.0052

Interpretation note:
1. Peak drift generally decreases from smaller to mid/large models.
2. Recovery is non-uniform; strongest positive average recovery appeared in `1b` and `1.4b` in this setup.
3. Rebound/overshoot remains weak; small crossing rates appear for `410m` and `2.8b`.

## Current Limits
1. The concept/prompt regime is a single synthetic concept (`politeness`).
2. Behavioral endpoint metric is next-token KL, not a full concept-consistency classifier.
3. No full-FT/LoRA adaptation lane executed yet in this run.
4. Per user instruction, no `6.9b`/`12b` experiments were run. (`6.9b` cache data may exist from an interrupted prior download attempt; `12b` is absent.)

## Recommended Immediate Follow-Up
1. Add multiple concept families and rerun same grid.
2. Add FT vs LoRA checkpoints and rerun the same intervention pipeline.
3. Add formal mixed-effects statistical scripts directly over `metrics_full.csv`.
