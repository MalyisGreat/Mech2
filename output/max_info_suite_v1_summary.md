# Max Info Suite V1 Summary

## Run ID
`max_info_suite_v1_20260226_220300`

## Objective
Collect high-volume, concept-diverse, word-vector-aware intervention data in one coordinated suite run.

## Configuration
1. Models: `pythia-70m, 160m, 410m, 1b, 1.4b, 2.8b`
2. Concepts (12): `politeness, empathy, confidence, cooperation, honesty, caution, creativity, precision, optimism, skepticism, safety, leadership`
3. Seeds: `42`
4. Vector methods: `mean_diff, linear_probe, word_centroid, random_orthogonal`
5. Injection strengths: `-2, -1, +1, +2`
6. Injection depths: `0.2, 0.5, 0.8`
7. Evaluation prompts per concept: `8`

## Data Volume
1. Suite runs: `12` (one per concept)
2. Total rows: `27,648`
3. Models covered: `6`
4. Methods covered: `4` (including explicit word-centroid vectors)

## Core Artifacts
1. Manifest: `runs/max_info_suite_v1_20260226_220300/suite_manifest.csv`
2. Unified metrics: `runs/max_info_suite_v1_20260226_220300/suite_metrics_full.csv`
3. Stratified summary: `runs/max_info_suite_v1_20260226_220300/suite_stratified_summary.csv`
4. Model summary: `runs/max_info_suite_v1_20260226_220300/suite_model_summary.csv`
5. Effects vs control: `runs/max_info_suite_v1_20260226_220300/suite_effect_vs_control.csv`
6. Scale trends: `runs/max_info_suite_v1_20260226_220300/suite_scale_trends.csv`
7. Report: `runs/max_info_suite_v1_20260226_220300/suite_report.md`

## Word Vector Atlas
Per-model word-centroid vector statistics were exported to:
`output/word_vector_atlas/`

For each model:
1. `word_vector_norms.csv`
2. `concept_cosine_matrix.csv`
3. `concept_neighbors.csv`
4. `vectors_metadata.json`

## Quick Method-Level Snapshot
Average over all concepts and models:
1. `word_centroid`: recovery `0.0868`, peak_rel `0.0359`
2. `mean_diff`: recovery `0.0821`, peak_rel `0.0368`
3. `linear_probe`: recovery `0.0787`, peak_rel `0.0376`
4. `random_orthogonal`: recovery `0.0768`, peak_rel `0.0363`

Interpretation: word-centroid vectors are active and competitive with activation-derived vectors in this suite.

