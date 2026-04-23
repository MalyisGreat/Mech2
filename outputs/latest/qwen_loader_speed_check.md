# Qwen Loader Speed Check

- Date: `2026-04-21`
- Repo: `C:\Users\joshj\joseph-stroud-identity-stability-research`
- Goal: test whether the remaining Qwen smoke bottleneck could be reduced by fixing model-loading behavior.

## What Failed

The original hypothesis was that `Qwen/Qwen3-1.7B` was slow mainly because the optional flash-linear-attention stack was missing.

That did not hold on this machine in a clean way:

- `flash-linear-attention` / `fla-core` older wheels could be unpacked and imported.
- The actual FLA kernel path failed on Windows because the local Triton stack is incompatible with what FLA expects.
- `causal-conv1d` also does not have a clean binary-install path for this Python / Windows stack.

Conclusion: the FLA route is not a safe local speed fix here.

## What Was Actually Fixed

`src/identity_stability/modeling.py` was updated to:

- detect when a model snapshot is already cached and use `local_files_only=True`
- enable Hugging Face parallel weight loading by default
- place CUDA models directly on GPU during `from_pretrained(...)`
- keep the old fallback behavior when a cached snapshot is not present

## Direct Load Benchmark

Measured on `Qwen/Qwen3-1.7B` with a warm cache:

- Pre-patch current-style cached load: `9.58s`
- Post-patch `load_model(...)`: `6.73s`

That is a `1.42x` speedup for the actual `load_model(...)` path on a warm cache.

## End-to-End Smoke Check

Same experiment family:

- Previous smoke config: `configs/identity_battery/self_recognition_nearfoil_qwen_v3_smoke.yaml`
- New smoke config: `configs/identity_battery/self_recognition_nearfoil_qwen_v4_smoke.yaml`

Observed totals:

- Prior `v3` smoke, observed live in-thread: `72` rows in about `575s` from `9:16:05 PM` to `9:25:40 PM`
- New `v4` smoke: `72` rows in `488.44s`

Rows per minute:

- `v3`: `7.51 rows/min`
- `v4`: `8.84 rows/min`

End-to-end speedup:

- `1.18x`

## Validity Check

The speed fix did not degrade probe quality in the `v4` smoke:

- `72 / 72` rows valid
- `0` invalid rows
- `pair_valid_rate = 1.0` in every difficulty / model / frame cell in `outputs/latest/qwen_holdout_v4_smoke/self_recognition_nearfoil/quality_summary.csv`

## Interpretation

The main actionable speed fix on this Windows machine was not the missing FLA stack. It was eliminating unnecessary online Hugging Face checks for already-cached models and tightening the CUDA load path.

That materially improved `Qwen/Qwen3-1.7B` load time, but the full smoke improved only moderately because `Qwen/Qwen3.5-0.8B` still uses the `qwen3_5` torch fallback path and remains a noticeable share of startup cost.

## Artifacts

- Loader patch: `src/identity_stability/modeling.py`
- New smoke config: `configs/identity_battery/self_recognition_nearfoil_qwen_v4_smoke.yaml`
- New smoke output: `outputs/latest/qwen_holdout_v4_smoke/self_recognition_nearfoil`
- Smoke log: `outputs/latest/qwen_holdout_v4_smoke/run.log`
