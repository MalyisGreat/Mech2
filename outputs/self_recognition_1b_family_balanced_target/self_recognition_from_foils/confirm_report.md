# Self-Recognition Confirmatory Report

- Config: `configs\identity_battery\self_recognition_1b_family_balanced_target.yaml`
- Results: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\self_recognition_1b_family_balanced_target\self_recognition_from_foils\results.csv`
- Seeds: `[42, 123, 314, 1618, 2718]`
- Overall row accuracy: `0.4236` with row-bootstrap 95% CI `[0.3861, 0.4597]`
- Overall clustered accuracy: `0.4236` with cluster-bootstrap 95% CI `[0.3403, 0.5000]` over `24` prompt/axis/condition clusters.
- Chance level: `0.3333`
- Overall row-level one-sided exact binomial p-value vs chance: `0.000000`
- Overall cluster sign-test p-value vs chance: `0.038406` using `12/16` non-tie clusters above chance and `8` exact chance ties.

## Target Cell

- `1b / family_self`: accuracy `0.4236` with row-bootstrap 95% CI `[0.3861, 0.4597]`, `305/720` hits, row p-value `0.000000`, row Holm-adjusted `0.000000`.
- Clustered target estimate: `0.4236` with cluster-bootstrap 95% CI `[0.3403, 0.5000]`, `12/16` non-tie clusters above chance, `8` exact chance ties, cluster sign-test p-value `0.038406`, Holm-adjusted `0.038406`.

## Strongest Cell

- `1b / family_self` is currently highest at `0.4236` over `720` trials.

## Label-Bias Audit

- Additional audit tables were written to `confirm_label_bias_audit_by_model_frame.csv` and `confirm_label_bias_audit_by_cell.csv`.
- Strongest selected-label skew: `1b / family_self` selected one digit label at rate `0.6389` with normalized label entropy `0.8232`.
- In that same cell, accuracy spread across self-answer label positions was `0.4792`. Large spreads indicate answer-position bias, even when candidate placement is balanced by construction.

## Interpretation

- Primary inference is now clustered by prompt/axis/strength cells. Row-level binomial tests are retained only as diagnostics because repeated seeds and choice-label permutations are not independent observations.
- The earlier `1b / family_self` self-recognition bump survives the clustered confirmatory test and remains significant after Holm correction across model/frame cells, so it now looks like a real local effect rather than pure small-sample noise.
- Even if one cell remains above chance, the paper should still treat answer ownership as unstable unless the effect generalizes across nearby models, frames, and axes.
