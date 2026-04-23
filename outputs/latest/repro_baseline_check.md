# Repro Baseline Check

## Scope

This check was done before adding any identity-battery code. The goal was to confirm:

1. which saved artifacts implement the current paper backbone,
2. whether the current pipeline still runs end to end,
3. which manuscript claims are directly verified from cache or fresh execution,
4. what remains inferred from the manuscript rather than reconstructed from repo-local outputs.

## Source Of Truth

- Grounding manuscript: `D:/research paper part 1 (1).docx`
- Confirmed from manuscript text:
  - the paper is strongest as a containment / resistance paper with a modest identity interpretation,
  - the backbone is a within-family Pythia sweep,
  - the smaller GPT-2 / Qwen2.5 / Qwen3 screen is directional only,
  - larger Pythia models usually show less downstream change and smaller next-word shift after mean-difference interventions,
  - recovery is uneven and non-monotonic rather than a universal snap-back law,
  - one layer-zero mean-difference condition is a degenerate no-op and should remain a caveat,
  - mean-difference vectors must stay analytically separate from controls.

## Fresh Validation Performed

### Command run

```powershell
python scripts/run_experiment.py --config configs/smoke_arch_compat.yaml
```

### Fresh output produced

- Run directory: `runs/20260409_215635`

### What this fresh run verified

1. Config loading still works.
2. Prompt generation still works.
3. Model loading still works on current environment.
4. Vector estimation still works.
5. Residual-stream intervention hook still works.
6. Metric logging still works.
7. Standard run artifacts are still emitted:
   - `metrics_full.csv`
   - `metrics_summary.csv`
   - `layer_summary.csv`
   - `quick_report.md`
   - `vector_registry.csv`
   - `model_registry.csv`
   - `run_provenance.json`
   - `run_summary.json`
   - compute and GPU telemetry summaries

### Fresh run result snapshot

From `runs/20260409_215635/quick_report.md`:

- model: `gpt2`
- concept: `morality`
- rows: `2`
- mean recovery: `0.4584`
- mean peak drift relative: `0.013867`
- mean CAD: `1.0491`
- mean persistence: `0.5416`

This was a smoke validation only. It confirms executability, not the paper's statistical backbone.

## Cached Backbone Verification

### Primary cached suite used

- `runs/research_suite_v2_20260226_210325`

### What this suite contains

Verified from `suite_manifest.csv`, `suite_model_summary.csv`, and `suite_effect_vs_control.csv`:

1. models: `EleutherAI/pythia-70m`, `160m`, `410m`, `1b`, `1.4b`, `2.8b`
2. concepts: `politeness`, `empathy`, `confidence`
3. seeds: `42`, `123`
4. vector methods: `mean_diff`, `linear_probe`, `random_orthogonal`
5. intervention strengths: `-2`, `-1`, `+1`, `+2`
6. layer positions: `0.2`, `0.5`
7. total suite rows: `6912`

### Manuscript patterns confirmed from cache

1. Larger Pythia models usually have much smaller next-token shift after `mean_diff` interventions.
   - `confidence`: KL falls monotonically from `0.0904` at `70m` to `0.00050` at `2.8b`
   - `empathy`: KL falls monotonically from `0.0893` at `70m` to `0.00059` at `2.8b`
   - `politeness`: KL falls from `0.0654` at `70m` to `0.00042` at `2.8b`, with a small `1b` to `1.4b` bump
2. Recovery is non-monotonic rather than a universal snap-back law.
   - `160m` and often `410m` sit at or near zero recovery
   - `1b` and `1.4b` show the strongest recovery
   - `2.8b` does not continue that rise and often drops back down
3. Peak displacement and next-token shift do not move together cleanly.
   - `410m` remains a strong anomaly: very small KL but high relative peak drift
4. Mean-difference and controls are already kept separate in saved outputs.
   - `suite_effect_vs_control.csv` stores:
     - `recovery_mean_control`
     - `peak_rel_mean_control`
     - `next_token_kl_mean_control`
     - explicit deltas vs control

### Interpretation supported by the cached suite

Directly supported:

1. within Pythia, scaling strongly reduces next-token perturbation magnitude,
2. within Pythia, containment / damping is the most robust pattern,
3. recovery is too uneven to support a universal snap-back claim,
4. controls matter and must remain separate from concept vectors.

Not directly supported as a strong claim by this suite alone:

1. universal concept-specific self-restoration,
2. identity language stronger than the manuscript's current modest framing.

## Cached Add-On And Diagnostic Verification

### Layer-zero diagnostic

Verified from:

- `configs/prior_findings_addon.yaml`
- `runs/prior_findings_addon_pilot_suite_20260226_225516/suite_metrics_full.csv`

Confirmed:

1. layer-zero conditions are represented in the saved add-on pipeline,
2. `mean_diff` rows at `layer_index=0` exist,
3. those rows are degenerate no-ops in the cached pilot data,
4. example cached rows show `peak_drift=0.0`, `peak_drift_relative=0.0`, `recovery_fraction=0.0`, and `next_token_kl=0.0`.

This matches the manuscript's instruction to keep layer zero as a diagnostic caveat, not a headline result.

### Token-position effect

Verified from:

- `runs/prior_findings_token_position_v2_20260226_230323/suite_metrics_full.csv`
- `output/prior_findings_addon_summary.md`

Confirmed:

1. `token_position=0` is more persistent and less recoverable than `token_position=-1`,
2. the repo already has a cached directional result showing token position materially changes continuity metrics,
3. `estimation_token_position` was added to avoid a degenerate token-position setup.

## Cached Cross-Family Directional Check

Verified from:

- `runs/20260226_231747/resolved_config.json`
- `runs/20260226_231747/quick_report.md`
- `runs/20260226_231747/metrics_summary.csv`

Confirmed:

1. a saved cross-family run exists with `gpt2`, `Qwen/Qwen2.5-0.5B-Instruct`, and `Qwen/Qwen3-0.6B`,
2. it uses `morality`, `mean_diff`, one layer, one alpha, and only 2 evaluation prompts,
3. it behaves like a directional screen, not a statistical backbone.

This matches the manuscript's design constraint: cross-family evidence should be treated as directional only.

## What Was Verified Freshly vs From Cache

### Freshly run in this pass

1. `scripts/run_experiment.py` with `configs/smoke_arch_compat.yaml`
2. end-to-end baseline run output at `runs/20260409_215635`

### Verified from cached outputs

1. Pythia backbone suite structure and key damping / non-monotonic recovery patterns
2. control separation in suite-level outputs
3. token-position directional effect
4. layer-zero degenerate no-op
5. existence of a small cross-family directional check

### Inferred from manuscript rather than reconstructed from current repo-local outputs

1. the full manuscript count of `632,448` measured conditions
2. the exact final manuscript phrasing and section-level argumentation
3. the full six-model, four-label, three-seed, two-token-position backbone as a single repo-local cached suite artifact

The manuscript claims were read directly from the manuscript text, but the exact `632,448` total was not reconstructed from a single saved suite in this repo during this pass.

## Reproducibility Blockers And Caveats

1. The manuscript itself is currently external to the repo.
2. Some older cached suite CSVs predate newer analyzer columns.
   - for example, some suite summaries carry blank legacy fields such as `drift_auc`, `cad`, `persistence`, or `degradation` even though the current pipeline emits them
3. Older `output/*.md` summaries sometimes reflect a stronger recovery framing than the manuscript now allows.
4. The exact repo-local artifact corresponding to the manuscript's complete `632,448` condition count was not identified as a single cached suite during this pass.

## Baseline Preservation Decision

The baseline experiment engine is intact and should be preserved. The upgrade should be additive:

1. keep `src/identity_stability/experiment.py` as the baseline run path,
2. keep suite-level control separation,
3. keep the Pythia family as the statistical backbone,
4. treat cross-family results as directional,
5. keep layer zero as a diagnostic caveat,
6. avoid strengthening identity claims unless the new battery genuinely supports them.
