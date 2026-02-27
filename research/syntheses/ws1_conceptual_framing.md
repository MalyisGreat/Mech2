# WS1 - Conceptual Framing

## Thesis Core
The project reframes scaling from external task performance to internal trajectory stability. The key question is whether larger models preserve a learned internal behavioral manifold ("identity") under targeted residual-stream perturbations.

## Working Definitions
1. `Identity (operational)`: the baseline residual-stream trajectory induced by a prompt in the pretrained model.
2. `Intervention`: adding a concept-direction vector to selected residual stream states at layer `l` and token position `t`.
3. `Drift`: deviation of injected trajectory from baseline, measured layerwise and tokenwise.
4. `Recovery`: re-alignment of the injected trajectory toward baseline as computation progresses.
5. `Rebound`: transient over-correction where injected trajectory crosses past baseline-aligned direction before settling.

## Formalization
Let `h_l^base(t)` be baseline residual stream at layer `l`, token `t`.
Let `v_c` be a concept direction (unit-normalized) and `alpha` be injection strength.

Injection at layer `k`:
`h_k^inj(t) = h_k^base(t) + alpha * v_c`

Forward propagation produces `h_{l>k}^inj(t)`.

Define:
1. `drift_l(t) = ||h_l^inj(t) - h_l^base(t)||_2`
2. `cos_align_l(t) = cos(h_l^inj(t), h_l^base(t))`
3. `recover_l(t) = drift_k(t) - drift_l(t)` for `l > k`
4. `rebound_l(t)`: sign change in projection residual  
`r_l(t) = <h_l^inj(t) - h_l^base(t), v_back>`  
where `v_back` is a baseline-return axis (defined from local Jacobian or empirical return direction).

## Why This Framing Is Defensible
1. Linear directions in residual space are empirically meaningful for behavior steering (sources 1, 2, 10, 11).
2. Refusal and other behaviors can be strongly mediated by low-dimensional directions, but identifiability is imperfect and may be multi-dimensional (sources 3, 4, 5, 8).
3. Residual representations are dynamic and context-sensitive, so identity should be treated as trajectory, not static vector content (source 13).

## Competing Explanations To Separate
1. `Passive dilution`  
Injected signal attenuates because larger models are higher-dimensional averages.
2. `Active stabilization`  
Network dynamics contain restorative mechanisms that drive state back toward pretrained trajectory.

The rebound test is a useful discriminator: passive dilution predicts monotonic decay, while active stabilization admits over-corrective dynamics (source 24 motivates this possibility via self-repair behavior under ablation).

## Key Conceptual Risks
1. Identity may not be a single manifold; behaviors may be distributed across multiple directions (sources 5, 8, 9).
2. Measured recovery could be an artifact of metric geometry (choice of norm, whitening, layer scaling).
3. Injection effects can be confounded by token-position dependence and layer-specific sensitivity.

## Operational Decision
Treat identity stability as a model of internal control dynamics, not as a claim of immutable semantics. This allows the thesis to remain falsifiable under multi-directional, context-dependent steering evidence.

