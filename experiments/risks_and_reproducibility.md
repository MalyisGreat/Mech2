# Risks, Validity Threats, and Reproducibility

## Threats to Internal Validity
1. `Direction estimation leakage`: vectors may overfit prompt templates instead of concept.
2. `Layer-scale confounds`: raw residual norms differ by layer/model.
3. `Tokenizer confounds`: concept prompts may behave differently across model families.
4. `Seed variance`: apparent scale effects may be run-specific artifacts.
5. `Adaptation mismatch`: FT/LoRA checkpoints may differ in task quality, invalidating direct comparisons.

## Mitigations
1. Use strict train/eval split for direction estimation and intervention tests.
2. Include matched random-direction and anti-direction controls.
3. Use at least 3 seeds for prompt sampling and vector estimation.
4. Normalize and re-test with multiple metrics.
5. Match adaptation checkpoints by downstream task score before identity analysis.

## Threats to External Validity
1. Effects from one concept may not generalize to others.
2. Effects in Pythia may not transfer to instruction-tuned families.
3. Behavioral steering outcomes may depend on decoding settings.

## External Validity Checks
1. Evaluate at least 2 concept families (e.g., stylistic and safety-relevant).
2. Replicate final claims on at least one additional architecture family.
3. Test across greedy and sampled decoding settings.

## Reproducibility Controls
1. Pin software environment with exact package versions.
2. Log all run configs, seeds, model hashes, and data splits.
3. Save raw activations/metrics in versioned schema.
4. Use deterministic toggles where supported and document nondeterministic kernels.
5. Generate a run manifest per experiment batch.

## Recommended Artifact Layout
1. `configs/`: YAML/JSON configs for all sweeps.
2. `runs/`: raw trajectory/metric outputs.
3. `analysis/`: notebooks/scripts for tables and figures.
4. `reports/`: generated summaries and plot exports.
5. `logs/`: execution and anomaly logs.

## Decision Thresholds for Claims
1. Claim "scale-linked resistance" only if direction-consistent effects replicate across:
- at least two vector-construction methods,
- at least two concept families,
- and at least one adaptation regime.
2. Claim "active stabilization" only if rebound/non-monotonic indicators survive:
- null controls,
- metric normalization variants,
- and seed robustness checks.

