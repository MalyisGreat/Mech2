# WS2 - Activation Steering and Residual Interventions

## Evidence Summary
Activation engineering has established that controlled residual-space interventions can causally modify outputs without weight updates.

1. ActAdd shows direct behavior shaping by adding contrastive activation vectors at inference time (source 1).
2. ITI demonstrates that truthfulness can improve from linear interventions learned from activation statistics (source 2).
3. Safety/refusal work identifies dominant but not necessarily unique directions tied to refusal behavior (sources 3, 4, 5).
4. Recent work emphasizes context-aware steering fields and compositional steering rather than fixed global vectors (sources 6, 7).
5. Identifiability and decomposition studies caution against over-interpreting one vector as one mechanism (sources 8, 9).

## Methodological Implications for This Thesis
1. Concept vector construction should include at least two methods:
- `Mean-difference direction` between contrastive prompt sets.
- `Probe-derived direction` from linear classifier normal vectors.
2. Single-vector injection is useful but insufficient; include multi-vector and context-conditioned variants as controls.
3. Layer choice matters strongly; run intervention sweeps across early/mid/late layers.
4. Token position matters; test injections at the final prompt token and at targeted intermediate tokens.

## Recommended Intervention Matrix
For each model size:
1. Vector type: `{mean-diff, probe-normal, SAE-feature-composed}`.
2. Injection site: `{single layer, layer band}`.
3. Strength `alpha`: logarithmic sweep, e.g. `{0.25, 0.5, 1, 2, 4, 8}` after per-layer normalization.
4. Context regime: `{in-distribution prompt family, shifted prompt family}`.

## Interpretation Guardrails
1. If behavior changes but internal drift is small, direction may be logit-amplifying rather than globally representational.
2. If drift is large but output unchanged, later layers may compensate (candidate self-repair signal).
3. If different vector-construction methods yield similar recovery curves, identity claim is more robust.
4. If only one method supports the claim, likely method artifact.

## Critical Confounds
1. Residual norm scaling differs by layer/model size; raw Euclidean comparisons can mislead.
2. Direction quality can degrade with prompt-template leakage.
3. Steering vectors can entangle safety/refusal and style dimensions.
4. Quantization and inference precision can alter intervention efficacy.

## Practical Recommendations
1. Normalize vectors with layerwise activation covariance whitening.
2. Report both standardized and unstandardized metrics.
3. Include "null-direction" injections from random orthogonal vectors as negative controls.
4. Include "anti-concept" injections to test symmetry and rebound behavior.

