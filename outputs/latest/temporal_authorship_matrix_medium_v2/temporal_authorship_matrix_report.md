# Temporal Authorship Matrix Report

- Config: `C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\temporal_authorship_matrix_medium_v2.yaml`
- Output root: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\temporal_authorship_matrix_medium_v2`
- Model: `EleutherAI/pythia-410m-deduped`
- Revisions: `step16000, step64000, step128000, step143000`

- `temporal_authorship_self_preference_rate`: mean `0.2845` [0.2069, 0.3707]` over `n=116` units
- `temporal_authorship_diagonal_avg_logprob`: mean `-1.4760` [-1.5902, -1.3717]` over `n=4` units
- `temporal_authorship_off_diagonal_avg_logprob`: mean `-1.5826` [-1.6596, -1.5069]` over `n=12` units
- `temporal_authorship_diagonal_margin_logprob`: mean `0.0583` [-0.0390, 0.1555]` over `n=4` units
- `checkpoint_age_recognition_final_self_preference_rate`: mean `0.2069` [0.0690, 0.3793]` over `n=29` units

## By Evaluator

- `step128000`: self-preference `0.1724`, mean diagonal margin `-0.0424`, prompts `n=29`
- `step143000`: self-preference `0.2069`, mean diagonal margin `-0.0356`, prompts `n=29`
- `step16000`: self-preference `0.4483`, mean diagonal margin `0.1872`, prompts `n=29`
- `step64000`: self-preference `0.3103`, mean diagonal margin `0.1238`, prompts `n=29`

## Generation Quality

- `step128000`: valid-rate `0.2778`, unique-ratio `0.3398`, top-token-rate `0.1703`, top-bigram-rate `0.0982`, n=`36`
- `step143000`: valid-rate `0.2778`, unique-ratio `0.3472`, top-token-rate `0.1496`, top-bigram-rate `0.0995`, n=`36`
- `step16000`: valid-rate `0.4167`, unique-ratio `0.4224`, top-token-rate `0.1458`, top-bigram-rate `0.0925`, n=`36`
- `step64000`: valid-rate `0.3056`, unique-ratio `0.3702`, top-token-rate `0.1369`, top-bigram-rate `0.0831`, n=`36`

## Figures

- Heatmap: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\temporal_authorship_matrix_medium_v2\figures\temporal_authorship_matrix_heatmap.png`
- Self-preference curve: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\temporal_authorship_matrix_medium_v2\figures\temporal_authorship_self_preference_curve.png`