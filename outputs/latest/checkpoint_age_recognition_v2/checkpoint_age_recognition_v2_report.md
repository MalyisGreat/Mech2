# Checkpoint Age Recognition V2 Report

- Config: `C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\checkpoint_age_recognition_v2_full.yaml`
- Output root: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\checkpoint_age_recognition_v2`
- Model: `EleutherAI/pythia-410m-deduped`
- Anchor revision: `step143000`
- Comparison revisions: `step128000, step64000, step16000`
- Evaluator revisions: `step143000`

- `checkpoint_age_centered_choice_rate`: mean `0.5417` [0.3333, 0.7083]` over `n=24` units
- `checkpoint_age_raw_choice_rate`: mean `0.5417` [0.3333, 0.7083]` over `n=24` units
- `checkpoint_age_centered_margin_logprob`: mean `0.0402` [-0.0486, 0.1221]` over `n=24` units
- `checkpoint_age_raw_margin_logprob`: mean `0.0402` [-0.0486, 0.1221]` over `n=24` units
- `prompt_screen_mean_anchor_js`: mean `0.0384` [0.0346, 0.0424]` over `n=120` units

## By Comparison Revision

- `step128000`: choose-current `0.5556`, centered margin `-0.0017`, n=`9`
- `step16000`: choose-current `0.7500`, centered margin `0.1401`, n=`8`
- `step64000`: choose-current `0.2857`, centered margin `-0.0201`, n=`7`

## By Evaluator-Comparison Pair

- evaluator `step143000` vs `step128000`: choose-current `0.5556`, centered margin `-0.0017`, n=`9`
- evaluator `step143000` vs `step16000`: choose-current `0.7500`, centered margin `0.1401`, n=`8`
- evaluator `step143000` vs `step64000`: choose-current `0.2857`, centered margin `-0.0201`, n=`7`

## Generation Quality

- `step128000`: valid-rate `0.3056`, unique-ratio `0.3597`, top-token-rate `0.1655`, top-bigram-rate `0.0984`, n=`72`
- `step143000`: valid-rate `0.2361`, unique-ratio `0.3426`, top-token-rate `0.1577`, top-bigram-rate `0.1014`, n=`72`
- `step16000`: valid-rate `0.3889`, unique-ratio `0.3958`, top-token-rate `0.1523`, top-bigram-rate `0.0925`, n=`72`
- `step64000`: valid-rate `0.2917`, unique-ratio `0.3634`, top-token-rate `0.1500`, top-bigram-rate `0.0985`, n=`72`

## Figures

- Choice-rate curve: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\checkpoint_age_recognition_v2\figures\checkpoint_age_recognition_curve.png`