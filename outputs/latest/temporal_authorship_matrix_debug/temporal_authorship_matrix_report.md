# Temporal Authorship Matrix Report

- Config: `C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\temporal_authorship_matrix_debug.yaml`
- Output root: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\temporal_authorship_matrix_debug`
- Model: `EleutherAI/pythia-410m-deduped`
- Revisions: `step4000, step16000, step64000, step128000, step143000`

- `temporal_authorship_self_preference_rate`: mean `0.3333` [0.1667, 0.5000]` over `n=30` units
- `temporal_authorship_diagonal_avg_logprob`: mean `-1.0172` [-1.0579, -0.9764]` over `n=5` units
- `temporal_authorship_off_diagonal_avg_logprob`: mean `-1.1357` [-1.1939, -1.0846]` over `n=20` units
- `temporal_authorship_diagonal_margin_logprob`: mean `-0.1227` [-0.2170, -0.0005]` over `n=5` units
- `checkpoint_age_recognition_final_self_preference_rate`: mean `0.3333` [0.0000, 0.6667]` over `n=6` units

## By Evaluator

- `step128000`: self-preference `0.1667`, mean diagonal margin `-0.2252`, prompts `n=6`
- `step143000`: self-preference `0.3333`, mean diagonal margin `-0.1747`, prompts `n=6`
- `step16000`: self-preference `0.5000`, mean diagonal margin `-0.0742`, prompts `n=6`
- `step4000`: self-preference `0.5000`, mean diagonal margin `0.1055`, prompts `n=6`
- `step64000`: self-preference `0.1667`, mean diagonal margin `-0.2452`, prompts `n=6`

## Figures

- Heatmap: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\temporal_authorship_matrix_debug\figures\temporal_authorship_matrix_heatmap.png`
- Self-preference curve: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\temporal_authorship_matrix_debug\figures\temporal_authorship_self_preference_curve.png`