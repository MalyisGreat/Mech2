# Experimental Blueprint

## Objective
Empirically test whether larger models exhibit stronger residual-trajectory identity preservation under controlled concept-direction injections, before and after adaptation.

## Model Families
## Phase A (Core scale test, pretrained)
1. Pythia: `160M, 410M, 1B, 2.8B, 6.9B, 12B` (source 22).

## Phase B (Adaptation test)
1. Select 3 sizes from Phase A (small/mid/large).
2. For each size: create two adapted variants:
- Full fine-tuning.
- LoRA (and optionally DoRA as sensitivity check; sources 29, 31).

## Prompt and Concept Construction
1. Build a balanced prompt set with contrastive concept labels:
- Positive concept prompts.
- Negative/neutral matched prompts.
2. Split into:
- Vector-estimation split.
- Injection-evaluation split.
- Generalization split (domain-shift prompts).
3. Estimate concept directions via:
- Mean-difference vectors.
- Linear probe normals.
- Optional SAE-composed vectors.

## Intervention Grid
For each model and prompt:
1. Injection layers: sweep early/mid/late blocks.
2. Injection location: final prompt token and selected intermediate tokens.
3. Injection strength `alpha`: `0.25, 0.5, 1, 2, 4, 8` (post-normalization).
4. Controls:
- No injection baseline.
- Random orthogonal direction injection.
- Anti-concept direction injection.

## Data Collected Per Run
1. Residual states for baseline and injected trajectories across all layers.
2. Logits and generated tokens.
3. Drift/recovery metrics and derived diagnostics.
4. Metadata: seed, prompt ID, vector method, layer, token, alpha, model size, adaptation type.

## Subagent-Style Execution Lanes
1. `Lane A - Data/Prompts`: build and validate balanced concept datasets.
2. `Lane B - Direction Estimation`: compute and validate concept vectors.
3. `Lane C - Intervention Engine`: run injection sweeps and capture residual traces.
4. `Lane D - Analysis`: compute metrics/statistics and generate plots.
5. `Lane E - Adaptation`: train FT/LoRA variants and rerun lane C.
6. `Lane F - QA/Repro`: verify determinism, config logging, and result integrity.

## Layerwise Trajectory Visualization
1. Plot drift vs layer for each alpha and model size.
2. Plot recovery gain profiles by depth.
3. Plot overshoot index heatmaps (layer x alpha).
4. Compare pretrained vs adapted variants side-by-side.

## Implementation Notes
1. Use forward hooks for residual stream capture and injection.
2. Keep standardized config files for each run.
3. Persist all run artifacts to structured directories:
- `runs/<date>/<model>/<adaptation>/<seed>/<condition>.jsonl`

## Minimal Milestone Order
1. Implement single-model pilot with one concept vector method.
2. Validate metric sanity and controls.
3. Expand to full model-size sweep.
4. Add adaptation comparisons.
5. Final integrated analysis and write-up.

