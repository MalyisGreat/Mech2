# Metrics and Statistical Plan

## Core Metrics
For each run condition `(model, adaptation, vector_method, layer_k, token_t, alpha)`:

1. `Peak Drift`
`D_peak = max_{l>=k} ||h_l^inj - h_l^base||_2`

2. `End Drift`
`D_end = ||h_L^inj - h_L^base||_2` where `L` is final layer.

3. `Recovery Fraction`
`R_frac = (D_peak - D_end) / max(D_peak, eps)`

4. `Recovery Latency`
Smallest `l` where drift falls below `tau * D_peak` for chosen `tau` (e.g., `0.25`).

5. `Overshoot Index`
Signed area of baseline-axis projection crossing:
`OI = sum_l sign_flip(r_l) * |r_l|` after first crossing event.

6. `Behavioral Shift`
Concept-consistent output rate delta between injected and baseline decoding.

## Normalization Strategy
1. Layerwise z-score or covariance-whitened residuals for cross-layer comparability.
2. Report unnormalized metrics in appendix for transparency.
3. Use both cosine and Euclidean distances as robustness check.

## Primary Hypothesis Tests
## H1 (Scale and Drift)
Larger models have smaller `D_peak` and `D_end` under matched interventions.
Test: mixed-effects regression with model size as fixed effect and prompt/seed as random effects.

## H2 (Scale and Recovery)
Larger models have larger `R_frac` and shorter recovery latency.
Test: mixed-effects regression plus trend test across ordered model sizes.

## H3 (Active Stabilization)
Stronger injections produce non-monotonic rebound signatures in larger models.
Test: polynomial/spline terms for `alpha` and model-size interaction on `OI`.

## H4 (Adaptation Effects)
Full FT increases baseline-shift and reduces pretrained-identity retention more than LoRA.
Test: paired comparisons across matched downstream-performance checkpoints.

## Model Specification Template
`metric ~ log_params + alpha + layer_group + adaptation + log_params:alpha + adaptation:alpha + (1|prompt_id) + (1|seed)`

## Causal Plausibility Checks
1. Null-direction controls should not reproduce concept-aligned behavioral shifts.
2. Anti-concept injections should invert directional effects.
3. Similar results across vector-estimation methods strengthen causal claims.

## Multiple Testing and Uncertainty
1. Control FDR for families of layerwise comparisons.
2. Report bootstrap confidence intervals for key aggregate metrics.
3. Pre-register primary metrics/hypotheses before full sweep.

## Failure Criteria
1. If recovery metrics are inconsistent across distance metrics and normalization schemes, defer strong identity conclusions.
2. If effects appear only in one model family and vanish in a second family, present as family-specific.
3. If adaptation quality is unmatched across FT/LoRA, do not interpret identity differences causally.

