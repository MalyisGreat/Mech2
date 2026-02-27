# Research Suite V2 Findings

## Scope
Upgraded experiment suite on current model set:
1. Models: `70m, 160m, 410m, 1b, 1.4b, 2.8b`
2. Concepts: `politeness, empathy, confidence`
3. Seeds: `42, 123`
4. Vector methods: `mean_diff, linear_probe, random_orthogonal (control)`
5. Injection strengths: `-2, -1, +1, +2`
6. Layer positions: `0.2, 0.5`

Total rows analyzed: `6912`.

## Main Improvements Over Prior Pass
1. Multi-concept rather than single-concept evidence.
2. Multi-seed rather than single-seed evidence.
3. Explicit random orthogonal control vectors.
4. Signed interventions (anti-direction and direction).
5. Normalized drift metrics for better cross-model comparability.
6. Stratified summaries and confidence intervals.

## Primary Findings
1. **Scale-linked relative drift reduction appears robust**  
Across concepts and methods, `peak_drift_relative` tends to decrease with model scale (negative slopes vs log-params), including concept vectors.

2. **Recovery signal is not concept-specific in this setup**  
Recovery increases with scale for concept vectors, but also for random orthogonal controls. This weakens any claim that recovery alone reflects concept-targeted stabilization.

3. **Concept-vs-control deltas are mostly small for recovery/relative drift**  
Average deltas (`mean_diff` or `linear_probe` minus random orthogonal) are near zero for recovery and peak-relative drift in most concept/method combinations.

4. **Behavioral shift signal is modest and mixed**  
`next_token_kl` deltas vs control are generally small; only `confidence + linear_probe` showed a consistent positive delta in this suite.

5. **Seed stability is strong**  
Across stratified conditions, seed-to-seed correlations are high (typically ~0.97 to ~1.00), indicating the observed patterns are reproducible under the tested seeds.

## Interpretation
Current evidence supports:
1. Larger models are less relatively displaced by injected perturbations (generic robustness trend).

Current evidence does **not yet** strongly support:
1. Concept-specific active stabilization (because control vectors show similar recovery scaling).

## Practical Conclusion for Thesis Positioning
At this stage, the strongest defensible claim is:
1. Scaling is associated with reduced normalized perturbation sensitivity.
2. Recovery metrics alone are insufficient to claim concept-specific identity enforcement.

The thesis should frame active identity stabilization as an open hypothesis requiring stronger discriminative tests beyond current recovery measures.

## Key Artifacts
1. Suite manifest: `runs/research_suite_v2_20260226_210325/suite_manifest.csv`
2. Combined metrics: `runs/research_suite_v2_20260226_210325/suite_metrics_full.csv`
3. Stratified summary: `runs/research_suite_v2_20260226_210325/suite_stratified_summary.csv`
4. Model summary: `runs/research_suite_v2_20260226_210325/suite_model_summary.csv`
5. Effects vs control: `runs/research_suite_v2_20260226_210325/suite_effect_vs_control.csv`
6. Scale trends: `runs/research_suite_v2_20260226_210325/suite_scale_trends.csv`
7. Seed consistency: `runs/research_suite_v2_20260226_210325/suite_seed_consistency.csv`

