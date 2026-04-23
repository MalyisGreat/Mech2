# Checkpoint Age Recognition V2 Report

- Config: `C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\checkpoint_age_recognition_v2_smoke.yaml`
- Output root: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\checkpoint_age_recognition_v2_smoke`
- Model: `EleutherAI/pythia-410m-deduped`
- Anchor revision: `step143000`
- Comparison revisions: `step128000, step64000, step16000`
- Evaluator revisions: `step143000`

- `checkpoint_age_centered_choice_rate`: mean `0.6000` [0.2000, 1.0000]` over `n=5` units
- `checkpoint_age_raw_choice_rate`: mean `0.6000` [0.2000, 1.0000]` over `n=5` units
- `checkpoint_age_centered_margin_logprob`: mean `0.1401` [-0.0413, 0.3189]` over `n=5` units
- `checkpoint_age_raw_margin_logprob`: mean `0.1401` [-0.0413, 0.3189]` over `n=5` units
- `prompt_screen_mean_anchor_js`: mean `0.0513` [0.0425, 0.0604]` over `n=40` units

## By Comparison Revision

- `step128000`: choose-current `0.5000`, centered margin `0.0395`, n=`2`
- `step16000`: choose-current `1.0000`, centered margin `0.3713`, n=`2`
- `step64000`: choose-current `0.0000`, centered margin `-0.1210`, n=`1`

## By Evaluator-Comparison Pair

- evaluator `step143000` vs `step128000`: choose-current `0.5000`, centered margin `0.0395`, n=`2`
- evaluator `step143000` vs `step16000`: choose-current `1.0000`, centered margin `0.3713`, n=`2`
- evaluator `step143000` vs `step64000`: choose-current `0.0000`, centered margin `-0.1210`, n=`1`

## Generation Quality

- `step128000`: valid-rate `0.4167`, unique-ratio `0.4622`, top-token-rate `0.1600`, top-bigram-rate `0.0840`, n=`12`
- `step143000`: valid-rate `0.3333`, unique-ratio `0.4145`, top-token-rate `0.1456`, top-bigram-rate `0.0907`, n=`12`
- `step16000`: valid-rate `0.3333`, unique-ratio `0.3843`, top-token-rate `0.1443`, top-bigram-rate `0.0976`, n=`12`
- `step64000`: valid-rate `0.3333`, unique-ratio `0.4174`, top-token-rate `0.1451`, top-bigram-rate `0.0971`, n=`12`

## Figures

- Choice-rate curve: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\checkpoint_age_recognition_v2_smoke\figures\checkpoint_age_recognition_curve.png`