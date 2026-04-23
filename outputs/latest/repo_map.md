# Repo Map

## Grounding

- Manuscript source of truth used for this upgrade pass: `D:/research paper part 1 (1).docx`
- Confirmed manuscript title: `Patterned Continuity: Identity and Resistance in Language Models`
- Important note: the current manuscript is not versioned inside this repo. The repo contains older planning and synthesis notes in `research/` and `output/`, but those should not override the manuscript.

## Current Paper Backbone

### Core pipeline

1. Vector extraction
   - `src/identity_stability/prompt_bank.py`
   - `src/identity_stability/concept_registry.py`
   - `src/identity_stability/vectors.py`
2. Intervention
   - `src/identity_stability/modeling.py`
   - `src/identity_stability/intervention.py`
3. Metric logging
   - `src/identity_stability/metrics.py`
   - `src/identity_stability/experiment.py`
4. Multi-run orchestration
   - `scripts/run_experiment.py`
   - `scripts/run_research_suite.py`
   - `scripts/run_prior_findings_addon.py`
   - `src/identity_stability/multi_gpu.py`
5. Aggregation and suite summaries
   - `scripts/analyze_research_suite.py`
   - `scripts/summarize_run.py`
6. Figure and atlas exports
   - `scripts/plot_3d_concept_trajectories.py`
   - `scripts/export_word_vector_atlas.py`

### Data flow

1. `build_prompt_set(...)` creates estimation pairs and evaluation prompts.
2. `extract_layer_activations(...)` gathers residual activations at the estimation token.
3. `estimate_concept_vectors(...)` builds `mean_diff`, `linear_probe`, and control vectors.
4. `run_trace_batch(...)` runs baseline and injected forward passes with a layer pre-hook.
5. `compute_trajectory_metrics(...)` computes drift, recovery, overshoot, cosine alignment, and next-token KL.
6. `run_experiment(...)` writes per-run artifacts such as:
   - `resolved_config.json`
   - `prompt_set.json`
   - `metrics_full.csv`
   - `metrics_summary.csv`
   - `layer_summary.csv`
   - `quick_report.md`
   - `vector_registry.csv`
   - `model_registry.csv`
   - `run_provenance.json`
   - `run_summary.json`
7. `analyze_research_suite.py` merges run manifests into suite-level outputs such as:
   - `suite_metrics_full.csv`
   - `suite_stratified_summary.csv`
   - `suite_model_summary.csv`
   - `suite_effect_vs_control.csv`
   - `suite_prompt_style_summary.csv`
   - `suite_bootstrap_bands.csv`
   - `suite_seed_consistency.csv`
   - `suite_scale_trends*.csv`
   - `suite_scaling_laws*.csv`

## Important Configs

### Baseline and manuscript-relevant

- `configs/research_suite_base.yaml`
  - Pythia sweep backbone
- `configs/prior_findings_addon.yaml`
  - add-on suite with token-position and layer-zero-relevant support
- `configs/prior_findings_addon_pilot.yaml`
  - smaller add-on pilot
- `configs/pilot.yaml`
  - minimal Pythia pilot
- `configs/smoke_arch_compat.yaml`
  - smallest current smoke config

### Cross-family directional check

- `configs/final_models_h100_fast.yaml`
  - includes `gpt2`, `Qwen2.5`, and `Qwen3`
  - matches the manuscript's framing of a smaller cross-family directional screen rather than the statistical backbone

## Cached Result Sets Already Present

### Backbone-aligned Pythia suite

- `runs/research_suite_v2_20260226_210325`
  - 6 Pythia models
  - 3 concepts: `politeness`, `empathy`, `confidence`
  - 2 seeds: `42`, `123`
  - methods: `mean_diff`, `linear_probe`, `random_orthogonal`
  - strengths: `-2`, `-1`, `+1`, `+2`
  - layer positions: `0.2`, `0.5`
  - token position: `-1`
  - suite rows: `6912`
  - role in current manuscript: best in-repo cached support for the "containment / resistance with modest identity interpretation" framing

### Broader within-family extension

- `runs/max_info_suite_v1_20260226_220300`
  - 12 concepts
  - 6 Pythia models
  - 1 seed
  - methods include `word_centroid`
  - layer positions `0.2`, `0.5`, `0.8`
  - suite rows: `27648`
  - role: broader concept-diverse follow-up, not the manuscript backbone

### Token-position and add-on checks

- `runs/prior_findings_addon_pilot_suite_20260226_225516`
  - includes layer-zero and add-on concepts
- `runs/prior_findings_token_position_v2_20260226_230323`
  - explicit token-position comparison with `token_position=-1` and `0`

### Cross-family directional screen

- `runs/20260226_231747`
  - models: `gpt2`, `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen3-0.6B`
  - concept: `morality`
  - seed: `7`
  - methods: `mean_diff`
  - 2 estimation prompts and 2 evaluation prompts
  - role: directional architecture sanity check only, not a statistical backbone

### Fresh smoke validation from this pass

- `runs/20260409_215635`
  - model: `gpt2`
  - concept: `morality`
  - method: `mean_diff`
  - purpose: verify current end-to-end pipeline still runs

## Manuscript-Adjacent Assets In Repo

- `experiments/PROTOCOL_LOCK_V1.md`
  - locked metric definitions and control rules
- `experiments/risks_and_reproducibility.md`
  - validity threats and claim thresholds
- `research/notes/`
  - working notes
- `research/sources/source_registry.md`
  - literature registry
- `research/syntheses/`
  - older conceptual and literature syntheses

## Current Gaps Relative To The Upgrade Request

Missing or not yet present before this upgrade pass:

- `AGENTS.md` in this repo
- `README_identity_battery.md`
- `configs/identity_battery/`
- the requested identity-battery YAML assets in `data/`
- the requested identity-battery experiment entrypoints
- `outputs/latest/` reporting artifacts

## Guidance For The Upgrade

1. Preserve `src/identity_stability/experiment.py` as the baseline engine.
2. Keep mean-difference concept directions separate from random orthogonal controls in every new analysis path.
3. Treat `research_suite_v2_20260226_210325` as the main cached baseline reference for the current manuscript.
4. Treat `runs/20260226_231747` as a directional cross-family check only.
5. Do not use older `output/*.md` thesis framing as source of truth when it conflicts with the manuscript.
