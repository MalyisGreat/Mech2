# Qwen Holdout V3 Smoke Report

## Config

- `configs/identity_battery/self_recognition_nearfoil_qwen_v3_smoke.yaml`

## Why this rerun exists

The earlier Qwen smoke at `outputs/latest/qwen_holdout/self_recognition_nearfoil` was not reliable enough for interpretation because a large fraction of rows contained duplicate or empty baseline/foil pairs.

The `v3` rerun changed the Qwen path in three important ways:

1. tokenizer chat formatting instead of the earlier generic prompt path
2. duplicate/empty/repetition pair filtering at the probe level
3. reproducible sampling with no stop-string truncation

## Quality result

The repaired smoke produced a clean scoreable dataset:

- total rows: `72`
- valid rows: `72`
- invalid rows: `0`

Every model/frame/difficulty cell in `quality_summary.csv` had:

- `pair_valid_rate = 1.0`
- `exact_text_match_n = 0`
- `near_duplicate_pair_n = 0`
- `baseline_repetition_collapse_n = 0`
- `foil_repetition_collapse_n = 0`

This is the main scientific win of the repair. The current Qwen smoke is now interpretable as a directional cross-family check rather than a measurement artifact.

## Directional outcome

Mean ownership accuracy by model:

- `0.8b = 0.375`
- `1.5b = 0.375`
- `1.7b = 0.416667`

Selected difficulty means from `summary.csv`:

- `0.8b / baseline_helpful / far_alt_frame = 0.75`
- `0.8b / family_self / near_contrary = 0.0`
- `1.5b / baseline_helpful / near_contrary = 0.5`
- `1.7b / baseline_helpful / near_contrary = 0.75`
- `1.7b / family_self / far_alt_frame = 0.25`

Interpretation:

- The repaired smoke does **not** support a simple monotonic modern-Qwen ownership story.
- It does support continuing to the broader directional holdout, because the Qwen path is now producing valid pairs and nontrivial differences rather than collapsed outputs.

## Next step

Run the broader directional holdout with:

- `configs/identity_battery/self_recognition_nearfoil_qwen_v3_holdout.yaml`

That run should now be interpreted as a true cross-family directional check, not as a contaminated probe sanity test.
