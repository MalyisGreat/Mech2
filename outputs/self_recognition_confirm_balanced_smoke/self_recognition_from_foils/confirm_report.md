# Self-Recognition Confirmatory Report

- Config: `configs\identity_battery\self_recognition_confirm_balanced_smoke.yaml`
- Results: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\self_recognition_confirm_balanced_smoke\self_recognition_from_foils\results.csv`
- Seeds: `[42, 123]`
- Overall row accuracy: `0.2812` with row-bootstrap 95% CI `[0.2240, 0.3490]`
- Overall clustered accuracy: `0.2812` with cluster-bootstrap 95% CI `[0.2292, 0.3229]` over `16` prompt/axis/condition clusters.
- Chance level: `0.3333`
- Overall row-level one-sided exact binomial p-value vs chance: `0.947687`
- Overall cluster sign-test p-value vs chance: `1.000000` using `0/4` non-tie clusters above chance and `12` exact chance ties.

## Target Cell

- `1b / family_self`: accuracy `0.2500` with row-bootstrap 95% CI `[0.1250, 0.3750]`, `12/48` hits, row p-value `0.918822`, row Holm-adjusted `1.000000`.
- Clustered target estimate: `0.2500` with cluster-bootstrap 95% CI `[0.0833, 0.3333]`, `0/1` non-tie clusters above chance, `3` exact chance ties, cluster sign-test p-value `1.000000`, Holm-adjusted `1.000000`.

## Strongest Cell

- `70m / baseline_helpful` is currently highest at `0.3333` over `48` trials.

## Interpretation

- Primary inference is now clustered by prompt/axis/strength cells. Row-level binomial tests are retained only as diagnostics because repeated seeds and choice-label permutations are not independent observations.
- The earlier `1b / family_self` self-recognition bump does not yet clear the clustered confirmatory threshold. The safer interpretation is that self-recognition remains fragile and cell-specific until the label-balanced rerun confirms it.
- Even if one cell remains above chance, the paper should still treat answer ownership as unstable unless the effect generalizes across nearby models, frames, and axes.
