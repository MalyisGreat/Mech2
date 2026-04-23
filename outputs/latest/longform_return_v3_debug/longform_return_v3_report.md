# Long-Form Return V3 Report

- Config: `configs\identity_battery\longform_return_v3_debug.yaml`
- Output root: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\longform_return_v3_debug`
- Models: `EleutherAI/pythia-410m`
- Frames: `baseline_helpful, family_self`
- Items: `2`
- Seeds: `11`

- `longform_return_chunk1_style_preference`: mean `-0.1461` [-0.2348, -0.0674]` over `n=4` units
- `longform_return_final_style_preference`: mean `-0.0964` [-0.1329, -0.0492]` over `n=4` units
- `longform_return_chunk1_axis_index`: mean `nan` [nan, nan]` over `n=0` units
- `longform_return_final_axis_index`: mean `nan` [nan, nan]` over `n=0` units
- `longform_return_half_life_chunk`: mean `2.0000` [2.0000, 2.0000]` over `n=1` units
- `longform_forced_shift_magnitude`: mean `0.0086` [0.0029, 0.0126]` over `n=4` units

## By Model / Frame / Axis

- `410m / baseline_helpful / cautious_vs_assertive`: chunk1-style `-0.0592`, final-style `-0.1437`, chunk1-axis `nan`, final-axis `nan`, half-life `2.0000`, forced-shift `0.0096`, n=`1`
- `410m / baseline_helpful / expansive_vs_terse`: chunk1-style `-0.1615`, final-style `-0.1136`, chunk1-axis `nan`, final-axis `nan`, half-life `nan`, forced-shift `0.0003`, n=`1`
- `410m / family_self / cautious_vs_assertive`: chunk1-style `-0.0756`, final-style `-0.1004`, chunk1-axis `nan`, final-axis `nan`, half-life `nan`, forced-shift `0.0136`, n=`1`
- `410m / family_self / expansive_vs_terse`: chunk1-style `-0.2879`, final-style `-0.0277`, chunk1-axis `nan`, final-axis `nan`, half-life `nan`, forced-shift `0.0110`, n=`1`

## Figures

- Overall chunk curve: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\longform_return_v3_debug\figures\longform_return_v3_chunk_curve.png`