# Temporal Authorship Matrix Report

- Config: `C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\temporal_authorship_matrix_72_v2.yaml`
- Output root: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\temporal_authorship_matrix_72_v2`
- Model: `EleutherAI/pythia-410m-deduped`
- Revisions: `step16000, step64000, step128000, step143000`

- `temporal_authorship_self_preference_rate`: mean `0.2800` [0.2150, 0.3400]` over `n=200` units
- `temporal_authorship_diagonal_avg_logprob`: mean `-1.5176` [-1.6040, -1.4182]` over `n=4` units
- `temporal_authorship_off_diagonal_avg_logprob`: mean `-1.6129` [-1.6756, -1.5472]` over `n=12` units
- `temporal_authorship_diagonal_margin_logprob`: mean `0.0413` [-0.0086, 0.1025]` over `n=4` units
- `checkpoint_age_recognition_final_self_preference_rate`: mean `0.2200` [0.1200, 0.3400]` over `n=50` units

## By Evaluator

- `step128000`: self-preference `0.2200`, mean diagonal margin `0.0248`, prompts `n=50`
- `step143000`: self-preference `0.2200`, mean diagonal margin `-0.0230`, prompts `n=50`
- `step16000`: self-preference `0.3800`, mean diagonal margin `0.1284`, prompts `n=50`
- `step64000`: self-preference `0.3000`, mean diagonal margin `0.0348`, prompts `n=50`

## Generation Quality

- `step128000`: valid-rate `0.2500`, unique-ratio `0.3471`, top-token-rate `0.1671`, top-bigram-rate `0.1053`, n=`72`
- `step143000`: valid-rate `0.2222`, unique-ratio `0.3434`, top-token-rate `0.1593`, top-bigram-rate `0.1051`, n=`72`
- `step16000`: valid-rate `0.3194`, unique-ratio `0.3793`, top-token-rate `0.1616`, top-bigram-rate `0.1025`, n=`72`
- `step64000`: valid-rate `0.2917`, unique-ratio `0.3598`, top-token-rate `0.1574`, top-bigram-rate `0.1048`, n=`72`

## Figures

- Heatmap: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\temporal_authorship_matrix_72_v2\figures\temporal_authorship_matrix_heatmap.png`
- Self-preference curve: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\temporal_authorship_matrix_72_v2\figures\temporal_authorship_self_preference_curve.png`