# Self/Other Boundary Confirm Report

- Config: `configs\identity_battery\self_other_boundary_confirm_clean.yaml`
- Smoke mode: `True`
- Results: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\self_other_boundary_confirm_clean_smoke\self_other_boundary\results.csv`
- Partial mode: `False`
- Models: `['EleutherAI/pythia-1b']`
- Frames: `['family_self']`
- Seeds: `[11]`
- Other-frame map: `{'family_self': 'tool_only'}`

## Main Result

- Overall no-steer boundary match: `0.0000` against `0.3333` chance.
- Overall steered boundary match: `0.0000`.
- Overall transfer delta (steered minus no-steer): `0.0000`.
- Overall self-moved-toward-other rate under steering: `0.2500`.

## Best Cell

- Strongest no-steer cell: `1b / family_self` at `0.0000`.
- `1b / family_self` no-steer boundary match: `0.0000` with 95% CI `[0.0000, 0.0000]`.
- `1b / family_self` clustered no-steer boundary match: `0.0000` with clustered 95% CI `[0.0000, 0.0000]`.
- `1b / family_self` steered boundary match: `0.0000`.
- `1b / family_self` clustered sign-test p-value vs chance: `1.000000`.
- `1b / family_self` Holm-adjusted clustered sign-test p-value vs chance: `1.000000`.

## Family-Self Pattern

- `1b`: no-steer `0.0000`, steered `0.0000`, delta `0.0000`.

## Transfer / Collapse Pattern

- Largest boundary drop under steering: `1b / family_self` with delta `0.0000`.
- Negative deltas mean the model's predicted self/other ordering became less aligned with the actual ordering after contrary steering.

## Interpretation

- If this report shows above-chance no-steer boundary match concentrated in a narrow model/frame pocket and weak elsewhere, that supports local self/other boundary knowledge rather than a broad self-model.
- If steering pushes self outputs toward the other-frame distribution while boundary match falls, that supports fragile self/other separation under pressure rather than robust identity coherence.
