# Self-Recognition Confirmatory Report

- Config: `configs\identity_battery\self_recognition_1b_family_balanced_clean.yaml`
- Results: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\self_recognition_1b_family_balanced_clean\self_recognition_from_foils\results.csv`
- Seeds: `[42, 123, 314, 1618, 2718]`
- Overall row accuracy: `0.3403` with row-bootstrap 95% CI `[0.3055, 0.3764]`
- Overall clustered accuracy: `0.3403` with cluster-bootstrap 95% CI `[0.2986, 0.3889]` over `24` prompt/axis/condition clusters.
- Chance level: `0.3333`
- Overall row-level one-sided exact binomial p-value vs chance: `0.359590`
- Overall cluster sign-test p-value vs chance: `0.636719` using `4/8` non-tie clusters above chance and `16` exact chance ties.

## Target Cell

- `1b / family_self`: accuracy `0.3403` with row-bootstrap 95% CI `[0.3055, 0.3764]`, `245/720` hits, row p-value `0.359590`, row Holm-adjusted `0.359590`.
- Clustered target estimate: `0.3403` with cluster-bootstrap 95% CI `[0.2986, 0.3889]`, `4/8` non-tie clusters above chance, `16` exact chance ties, cluster sign-test p-value `0.636719`, Holm-adjusted `0.636719`.

## Strongest Cell

- `1b / family_self` is currently highest at `0.3403` over `720` trials.

## Label-Bias Audit

- Additional audit tables were written to `confirm_label_bias_audit_by_model_frame.csv` and `confirm_label_bias_audit_by_cell.csv`.
- Strongest selected-label skew: `1b / family_self` selected one digit label at rate `0.7708` with normalized label entropy `0.5977`.
- In that same cell, accuracy spread across self-answer label positions was `0.7083`. Large spreads indicate answer-position bias, even when candidate placement is balanced by construction.

## Interpretation

- Primary inference is now clustered by prompt/axis/strength cells. Row-level binomial tests are retained only as diagnostics because repeated seeds and choice-label permutations are not independent observations.
- The `1b / family_self` self-recognition bump is the strongest model/frame cell, but it does not clear the multiplicity-corrected clustered confirmatory threshold across the full balanced grid. The safer interpretation is a local answer-ownership pocket, not robust self-model coherence.
- Even if one cell remains above chance, the paper should still treat answer ownership as unstable unless the effect generalizes across nearby models, frames, and axes.
