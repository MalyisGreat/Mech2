# Identity Master Suite Report

- Config: `C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\master_identity_validation_core_smoke.yaml`
- Output root: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\master_identity_validation_core_smoke`

## Self/Other Boundary V5

- `skipped: no completed boundary output in this suite`

## Self Prediction Transfer V3

- `self_prediction_self_accuracy`: mean `0.4766` [0.4141, 0.5391]` over `n=32` natural units
- `self_prediction_other_accuracy`: mean `0.5000` [0.3984, 0.5939]` over `n=32` natural units
- `self_prediction_gap_accuracy`: mean `0.4315` [0.2827, 0.5863]` over `n=28` natural units
- `self_prediction_actual_gap_rate`: mean `0.4453` [0.3594, 0.5312]` over `n=32` natural units
- `self_prediction_family_self_minus_baseline_helpful_self_accuracy_mean`: mean `0.0781` [-0.0625, 0.2188]` over `n=16` natural units
- `self_prediction_family_self_minus_baseline_helpful_other_accuracy_mean`: mean `0.0312` [-0.1719, 0.2031]` over `n=16` natural units
- `self_prediction_family_self_minus_baseline_helpful_gap_direction_accuracy_mean`: mean `-0.0321` [-0.3013, 0.2821]` over `n=13` natural units
- `self_prediction_family_self_minus_baseline_helpful_actual_gap_rate`: mean `-0.0156` [-0.1875, 0.1562]` over `n=16` natural units

## Commitment Persistence V2

- `skipped: no completed commitment output in this suite`

## Self Recognition Near-Foil V2

- `nearfoil_ownership_accuracy`: mean `0.4444` [0.1111, 0.7778]` over `n=9` natural units
- `nearfoil_pair_valid_rate`: mean `0.8438` [0.7812, 0.8984]` over `n=128` natural units
- `nearfoil_same_frame_resample`: mean `0.0000` [0.0000, 0.0000]` over `n=3` natural units
- `nearfoil_family_self_minus_baseline_helpful_choose_self_baseline`: mean `nan` [nan, nan]` over `n=0` natural units
- `nearfoil_family_self_minus_baseline_helpful_pair_valid`: mean `0.1875` [0.0469, 0.3125]` over `n=64` natural units
- `nearfoil_family_self_minus_baseline_helpful_style_distance`: mean `0.0905` [-0.1076, 0.2880]` over `n=64` natural units
- `nearfoil_family_self_minus_baseline_helpful_semantic_overlap`: mean `-0.0159` [-0.0972, 0.0613]` over `n=64` natural units
