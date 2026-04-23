# Overnight Token-Phase Confirmation Report

- Exact config path used: `configs\overnight_token_phase_confirm.yaml`
- Suite directory: `C:\Users\joshj\joseph-stroud-identity-stability-research\runs\overnight_token_phase_confirm_20260410_001437`
- Manifest: `C:\Users\joshj\joseph-stroud-identity-stability-research\runs\overnight_token_phase_confirm_20260410_001437\suite_manifest.csv`
- Smoke wall time (seconds): `69.0`
- Estimated full runtime (hours): `1.01`

## Prompt Set

- Prompt rows captured in analysis: `24` unique prompt/style pairs from the standard suite counts (`estimation_prompt_count=16`, `evaluation_prompt_count=8`).
- Exact prompt IDs: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\overnight_token_phase_confirm\prompt_ids_used.csv`

## Exact Commands Launched

- `C:\Users\joshj\miniconda3\python.exe C:\Users\joshj\joseph-stroud-identity-stability-research\scripts\run_prior_findings_addon.py --config configs\overnight_token_phase_confirm.yaml --concepts politeness empathy confidence --seeds 42 123 314 --token-positions -1 0 --suite-name overnight_token_phase_confirm`
- `C:\Users\joshj\miniconda3\python.exe C:\Users\joshj\joseph-stroud-identity-stability-research\scripts\analyze_research_suite.py --manifest C:\Users\joshj\joseph-stroud-identity-stability-research\runs\overnight_token_phase_confirm_20260410_001437\suite_manifest.csv --bootstrap-iters 500`

## Core Readout

- Token position materially changes persistence: token_position=0 is more persistent on average.
- Token position materially changes next-token KL: token_position=0 reduces output-side KL on average.
- Mean drift AUC difference (`token_position=0 - -1`): `8.7042`
- Mean recovery difference (`token_position=0 - -1`): `-0.0462`
- Mean persistence difference (`token_position=0 - -1`): `0.0462`
- Mean next-token KL difference (`token_position=0 - -1`): `-0.006090`
- The persistence effect is sign-consistent in 3/4 model sizes, and the KL effect is sign-consistent in 4/4 model sizes.
- `politeness` showed the largest concept-level divergence, with lower recovery at token_position=0 on average.
- 410m does not stand out as the clearest anomaly in the aggregated overnight summary.

## Best Manuscript Sentences

- Results: Across the Pythia sweep, intervention phase strongly moderated downstream behavior: injections at token_position=0 generally produced more persistent internal disturbance than token_position=-1, even when output-side KL did not increase in parallel.
- Discussion: These results strengthen a trajectory-phase interpretation of containment: robustness depends not only on model scale but also on when a perturbation is introduced along the prompt-to-generation path.
- Strongest caveat: The confirmatory sweep is still within-family Pythia-only and uses the three backbone concepts confirmed in the cached suite, so the token-phase result should be treated as a strong within-family moderator before cross-family generalization.
