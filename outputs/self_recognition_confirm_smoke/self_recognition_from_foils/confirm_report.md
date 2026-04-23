# Self-Recognition Confirmatory Report

- Config: `outputs\self_recognition_confirm\smoke_config.yaml`
- Results: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\self_recognition_confirm_smoke\self_recognition_from_foils\results.csv`
- Seeds: `[42, 123]`
- Overall accuracy: `0.2812` with 95% bootstrap CI `[0.1250, 0.4375]`
- Chance level: `0.3333`
- Overall one-sided exact binomial p-value vs chance: `0.789441`

## Target Cell

- `1b / family_self`: accuracy `0.5000` with 95% CI `[0.1250, 0.8750]`, `4/8` hits, p-value `0.258650`.

## Strongest Cell

- `1b / family_self` is currently highest at `0.5000` over `8` trials.

## Interpretation

- The earlier `1b / family_self` self-recognition bump does not yet look robust enough to treat as a stable positive result. The confirmatory run favors a weaker interpretation: self-recognition remains fragile and cell-specific.
- Even if one cell remains above chance, the paper should still treat answer ownership as unstable unless the effect generalizes across nearby models, frames, and axes.
