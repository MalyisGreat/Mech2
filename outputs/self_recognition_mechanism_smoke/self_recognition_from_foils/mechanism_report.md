# Self-Recognition Mechanism Report

- Config: `configs\identity_battery\self_recognition_mechanism_smoke.yaml`
- Results: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\self_recognition_mechanism_smoke\self_recognition_from_foils\results.csv`
- Models: `['EleutherAI/pythia-1b']`
- Frames: `['family_self', 'tool_only']`
- Strength magnitudes: `[0.5, 1.0]`
- Prompt sources: `self_prediction_bank` plus contrastive prompts = `True`

## Target Cell: 1b / family_self

- Strength `0.50`: accuracy `0.3125` with 95% CI `[0.1250, 0.5000]`, `5/16` hits, Holm-adjusted p `1.000000`.
- Strength `1.00`: accuracy `0.3750` with 95% CI `[0.1875, 0.6250]`, `6/16` hits, Holm-adjusted p `1.000000`.

## Family-Self Neighbor Comparison At Strength 1.0

- `1b`: accuracy `0.3750` with Holm-adjusted p `1.000000`.

## Prompt-Source Generalization For 1b / family_self

- `self_prediction_bank` at strength `0.50`: accuracy `0.3125` over `16` trials.
- `self_prediction_bank` at strength `1.00`: accuracy `0.3750` over `16` trials.

## Axis Structure For 1b / family_self

- Strongest axis/strength cell: `expansive_vs_terse` at strength `1.00` with accuracy `0.5000`.
- Weakest axis/strength cell: `expansive_vs_terse` at strength `0.50` with accuracy `0.2500`.

## Interpretation

- The target effect strengthens as the contrary foil is pushed farther from baseline: accuracy rises from `0.3125` at the weakest steer to `0.3750` at the strongest steer.
- The key question is whether `1b / family_self` remains distinct from neighboring family-self cells and whether it survives the expanded prompt pool rather than only the original prompt bank.
- If the effect remains local to `1b / family_self`, the right interpretation is still a narrow, frame-sensitive answer-ownership pocket rather than a general self-model.
