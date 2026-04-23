# Overnight Token-Phase Confirmation Report

- Exact config path used: `C:\Users\joshj\joseph-stroud-identity-stability-research\configs\token_phase_extensive.yaml`
- Suite directory: `C:\Users\joshj\joseph-stroud-identity-stability-research\runs\token_phase_extensive_20260410_005855`
- Manifest: `C:\Users\joshj\joseph-stroud-identity-stability-research\runs\token_phase_extensive_20260410_005855\suite_manifest.csv`
- Smoke wall time (seconds): `108.9`
- Estimated full runtime (hours): `7.64`

## Prompt Set

- Prompt rows captured in analysis: `40` unique prompt/style pairs from the standard suite counts (`estimation_prompt_count=16`, `evaluation_prompt_count=8`).
- Exact prompt IDs: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\token_phase_extensive\prompt_ids_used.csv`

## Exact Commands Launched

- `C:\Users\joshj\miniconda3\python.exe C:\Users\joshj\joseph-stroud-identity-stability-research\scripts\run_prior_findings_addon.py --config C:\Users\joshj\joseph-stroud-identity-stability-research\configs\token_phase_extensive.yaml --concepts politeness empathy confidence --seeds 42 123 314 2718 --token-positions -1 0 --suite-name token_phase_extensive`
- `C:\Users\joshj\miniconda3\python.exe C:\Users\joshj\joseph-stroud-identity-stability-research\scripts\analyze_research_suite.py --manifest C:\Users\joshj\joseph-stroud-identity-stability-research\runs\token_phase_extensive_20260410_005855\suite_manifest.csv --bootstrap-iters 500`

## Core Readout

- Token position materially changes persistence: token_position=0 is more persistent on average.
- Token position materially changes next-token KL: token_position=0 reduces output-side KL on average.
- Mean drift AUC difference (`token_position=0 - -1`): `5.4270`
- Mean recovery difference (`token_position=0 - -1`): `-0.0627`
- Mean persistence difference (`token_position=0 - -1`): `0.0627`
- Mean next-token KL difference (`token_position=0 - -1`): `-0.005097`
- The persistence effect is sign-consistent in 4/6 model sizes, and the KL effect is sign-consistent in 6/6 model sizes.
- `confidence` showed the largest concept-level divergence, with lower recovery at token_position=0 on average.
- 410m does not stand out as the clearest anomaly in the aggregated overnight summary.

## Best Manuscript Sentences

- Results: Across the Pythia sweep, intervention phase strongly moderated downstream behavior: injections at token_position=0 generally produced more persistent internal disturbance than token_position=-1, even when output-side KL did not increase in parallel.
- Discussion: These results strengthen a trajectory-phase interpretation of containment: robustness depends not only on model scale but also on when a perturbation is introduced along the prompt-to-generation path.
- Strongest caveat: The confirmatory sweep is still within-family Pythia-only and uses the three backbone concepts confirmed in the cached suite, so the token-phase result should be treated as a strong within-family moderator before cross-family generalization.
