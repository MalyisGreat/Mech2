# Protocol Lock V1

## 1) Drift Definition (Locked)
1. Residual state tracked at `token_position` from config (default `-1`, final prompt token).
2. Drift vector at layer index `l`:  
`delta_l = h_l(injected) - h_l(baseline)`
3. Drift magnitude:
- `drift_l = ||delta_l||_2`
- `relative_drift_l = drift_l / (||h_l(baseline)||_2 + 1e-12)`
4. Drift layer range:
- Start: `drift_start_index = inject_layer + 1`
- End: final hidden-state index.

## 2) Recovery Definition (Locked)
For active range `l >= drift_start_index`:
1. `peak_drift = max(drift_l)`
2. `end_drift = drift_last`
3. `recovery_fraction = (peak_drift - end_drift) / peak_drift` (if peak > 0)
4. `drift_auc = trapz(drift_l over active range)`
5. `recovery_slope = (drift_last - drift_first) / (n_active - 1)`
6. Overshoot handling:
- `projection_l` is signed projection onto initial perturbation direction.
- `crossed_baseline = any(projection_l < 0 after start)`
- `overshoot_index = sum(abs(min(projection_l, 0)))`

## 3) Comparability Rules (Locked)
1. Cross-model layer mapping uses both:
- `layer_index` (absolute)
- `layer_depth_ratio = layer_index / (n_layers - 1)` (relative)
2. Report cross-model trends primarily by `layer_depth_ratio` buckets; keep absolute layer analyses as appendix.

## 4) Alpha / Effective Push (Locked)
1. Raw intervention strength: `alpha` from config.
2. Vector norm logged as `vector_norm`.
3. Effective push magnitude logged:
- `effective_push_abs = |alpha| * vector_norm`
- `effective_push_rel_baseline = effective_push_abs / baseline_norm_at_start`

## 5) Required Controls (Locked)
1. `random_orthogonal` control vector is mandatory.
2. Prompt-bootstrap noise bands are mandatory (saved in `suite_bootstrap_bands.csv`).
3. No-injection baseline is always run for every prompt condition by design.

## 6) Showcase Figure Set (Locked)
Frozen paper showcase suite:
1. Concepts: `politeness`, `empathy`, `skepticism`
2. Prompts per concept (evaluation): `3`
3. Model sizes: `pythia-160m`, `pythia-1b`, `pythia-2.8b`
4. Export both Plotly HTML and PNG for each figure.

Config reference:
`configs/showcase_frozen_suite.yaml`

## 7) Run Logging / Reproducibility (Locked)
Each run `metrics_full.csv` must include:
1. `model_id`, `concept_name`, `vector_method`, `alpha`, `layer_index`, `layer_depth_ratio`
2. `peak_drift`, `peak_drift_relative`
3. `drift_auc`, `drift_auc_relative`
4. `recovery_fraction`, `recovery_slope`
5. `overshoot_index`, `crossed_baseline`
6. `effective_push_abs`, `effective_push_rel_baseline`

All plots must be saved as:
1. `.html` interactive
2. `.png` static

