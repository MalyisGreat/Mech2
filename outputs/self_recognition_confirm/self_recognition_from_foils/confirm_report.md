# Self-Recognition Confirmatory Report

- Config: `configs\identity_battery\self_recognition_confirm.yaml`
- Results: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\self_recognition_confirm\self_recognition_from_foils\results.csv`
- Seeds: `[42, 123, 314, 1618, 2718]`
- Overall row accuracy: `0.3193` with row-bootstrap 95% CI `[0.2990, 0.3401]`
- Overall clustered accuracy: `0.3193` with cluster-bootstrap 95% CI `[0.2932, 0.3453]` over `384` prompt/axis/condition clusters.
- Chance level: `0.3333`
- Overall row-level one-sided exact binomial p-value vs chance: `0.908815`
- Overall cluster sign-test p-value vs chance: `0.600678` using `190/384` non-tie clusters above chance and `0` exact chance ties.

## Target Cell

- `1b / family_self`: accuracy `0.5333` with row-bootstrap 95% CI `[0.4500, 0.6167]`, `64/120` hits, row p-value `0.000005`, row Holm-adjusted `0.000082`.
- Clustered target estimate: `0.5333` with cluster-bootstrap 95% CI `[0.3917, 0.6667]`, `17/24` non-tie clusters above chance, `0` exact chance ties, cluster sign-test p-value `0.031957`, Holm-adjusted `0.479360`.

## Strongest Cell

- `1b / family_self` is currently highest at `0.5333` over `120` trials.

## Interpretation

- Primary inference is now clustered by prompt/axis/strength cells. Row-level binomial tests are retained only as diagnostics because repeated seeds and choice-label permutations are not independent observations.
- The earlier `1b / family_self` self-recognition bump does not yet clear the clustered confirmatory threshold. The safer interpretation is that self-recognition remains fragile and cell-specific until the label-balanced rerun confirms it.
- Even if one cell remains above chance, the paper should still treat answer ownership as unstable unless the effect generalizes across nearby models, frames, and axes.
