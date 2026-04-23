# Qwen Holdout Repair Notes

## Why the original Qwen smoke was not manuscript-grade

- The first three-model Qwen smoke at `outputs/latest/qwen_holdout/self_recognition_nearfoil` produced many degenerate baseline/foil pairs.
- In that run, `26 / 72` rows were effectively unusable because the baseline and foil were identical or near-identical.
- Global ownership rates from that run were therefore not trustworthy:
  - `far_alt_frame = 0.375`
  - `medium_contrary = 0.333333`
  - `near_contrary = 0.333333`
- The biggest problem was measurement quality, not just a weak scientific effect.

## What was changed

### Probe-quality hardening

- `scripts/self_recognition_nearfoil.py`
  - added invalid-pair detection for:
    - exact baseline/foil text matches
    - near-duplicate pairs by stylometry + semantic overlap
    - repetition-collapse completions
    - empty completions
  - invalid pairs are now excluded from the accuracy summaries instead of being averaged in
  - added `quality_summary.csv`
  - added `results.partial.csv` checkpointing for long runs

### Qwen-specific inference repair

- `src/identity_stability/steered_generation.py`
  - added optional reproducible sampling controls
  - default behavior remains unchanged for the old backbone paths
- `scripts/self_recognition_nearfoil.py`
  - added tokenizer-chat formatting path for models/configs that opt into it
  - Qwen3-family prompts now use tokenizer chat templates with thinking disabled when supported

### New configs

- `configs/identity_battery/self_recognition_nearfoil_qwen_v3_debug_08b.yaml`
- `configs/identity_battery/self_recognition_nearfoil_qwen_v3_smoke.yaml`
- `configs/identity_battery/self_recognition_nearfoil_qwen_v3_holdout.yaml`

These use:

- tokenizer chat formatting
- no stop-string truncation
- reproducible sampling
- duplicate-pair validity checks

## Validation result before scaling

Single-model debug on `Qwen/Qwen3.5-0.8B`:

- output: `outputs/latest/qwen_holdout_v3_debug_08b/self_recognition_nearfoil`
- rows: `12`
- valid rows: `12`
- invalid rows: `0`

Key summary:

- `far_alt_frame = 0.75`
- `medium_contrary = 0.5`
- `near_contrary = 0.0`

This does **not** prove a strong self-recognition effect, but it does show that the repaired Qwen path can now generate non-empty, non-duplicate, scoreable pairs.

## Current run

The corrected three-model smoke is running at:

- `outputs/latest/qwen_holdout_v3_smoke/self_recognition_nearfoil`

## Promotion rule

Promote the repaired Qwen path to the broader directional holdout only if the three-model smoke meets both conditions:

1. Pair quality:
   - low invalid-pair rate
   - no widespread duplicate-pair collapse
   - no broad empty-completion failure

2. Scientific interpretability:
   - enough valid rows per model/frame/difficulty cell to compare patterns without the result being dominated by artifact filtering

If the three-model smoke fails these conditions, the correct conclusion is that the current near-foil probe is still not robust enough for a Qwen cross-family claim on this machine/configuration.
