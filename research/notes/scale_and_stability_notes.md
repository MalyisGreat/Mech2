# Scale and Stability Notes

## Focus
Whether larger models show stronger trajectory return under perturbation.

## Key Notes by Source ID
1. `S20/S21 (Scaling law baseline)`:
- Establishes why "better loss with scale" is expected.
- Does not directly imply internal dynamical stability.

2. `S22 (Pythia)`:
- Preferred model family for controlled cross-size internal analysis.
- Enables consistent tokenizer/training lineage across scales.

3. `S23 (PolyPythias)`:
- Pretraining outcomes can vary across runs; seed-aware analysis required.
- Stability/outlier structure should be modeled, not ignored.

4. `S24 (Self-repair)`:
- Evidence of compensatory internal behavior after local damage.
- Motivates non-monotonic recovery/rebound diagnostics.

5. `S12/S13 (Residual dynamics and context drift)`:
- Residual representations show both structure and context-driven movement.
- Supports trajectory-level framing rather than static-embedding framing.

## Operational Outcomes
1. Use mixed-effects models with seed/prompt random effects.
2. Include overshoot metrics, not just drift decay.
3. Compare absolute and relative layer positions.

