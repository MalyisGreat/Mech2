# CPU Identity Overnight Push Report

- Config: `C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\cpu_identity_overnight_push_report_smoke.yaml`
- Output root: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\cpu_identity_overnight_push_smoke`

## Prompt Template Screening

- `template_instruction_gap_accuracy`: mean `0.7500` [0.2500, 1.0000]` over `n=4` units
- `template_instruction_valid_choice_rate`: mean `1.0000` [1.0000, 1.0000]` over `n=6` units
- `template_instruction_family_self_minus_baseline_gap_accuracy`: mean `-1.0000` [-1.0000, -1.0000]` over `n=1` units
- `template_chat_gap_accuracy`: mean `0.6667` [0.0000, 1.0000]` over `n=3` units
- `template_chat_valid_choice_rate`: mean `1.0000` [1.0000, 1.0000]` over `n=6` units
- `template_chat_family_self_minus_baseline_gap_accuracy`: mean `0.0000` [0.0000, 0.0000]` over `n=1` units
- `template_plain_gap_accuracy`: mean `0.6667` [0.2500, 1.0000]` over `n=4` units
- `template_plain_valid_choice_rate`: mean `1.0000` [1.0000, 1.0000]` over `n=6` units
- `template_plain_family_self_minus_baseline_gap_accuracy`: mean `-1.0000` [-1.0000, -1.0000]` over `n=1` units

## Kinship Ladder Dissociation

- `kinship_choose_host`: mean `0.2812` [0.1562, 0.4375]` over `n=16` pairs
- `kinship_swap_consistency`: mean `0.5625` [0.3125, 0.8125]` over `n=16` pairs
- `kinship_family_self_minus_baseline_choose_host`: mean `0.1875` [-0.1250, 0.5000]` over `n=8` pairs
- `baseline_helpful / biography_overlap_stranger`: choose-host `0.5000`, swap-consistency `1.0000`, n=`2`
- `baseline_helpful / demographic_twin`: choose-host `0.0000`, swap-consistency `1.0000`, n=`2`
- `baseline_helpful / kin_family_member`: choose-host `0.0000`, swap-consistency `1.0000`, n=`2`
- `baseline_helpful / random_stranger`: choose-host `0.2500`, swap-consistency `0.5000`, n=`2`
- `family_self / biography_overlap_stranger`: choose-host `0.5000`, swap-consistency `0.0000`, n=`2`
- `family_self / demographic_twin`: choose-host `0.5000`, swap-consistency `0.0000`, n=`2`
- `family_self / kin_family_member`: choose-host `0.2500`, swap-consistency `0.5000`, n=`2`
- `family_self / random_stranger`: choose-host `0.2500`, swap-consistency `0.5000`, n=`2`

## Behavioral Fingerprint Stability

- `fingerprint_self_minus_decoy_accuracy`: mean `-0.1667` [-0.3333, -0.0208]` over `n=12` units
- `fingerprint_self_margin_vs_scrambled`: mean `1.0552` [0.8327, 1.2664]` over `n=12` units
- `fingerprint_family_self_minus_baseline_self_minus_decoy`: mean `-0.3750` [-0.6250, -0.1250]` over `n=4` units

## Commitment Persistence

- `commitment_adherence`: mean `0.5000` [0.5000, 0.5000]` over `n=4` dialogues
- `commitment_reveal_pair_accuracy`: mean `nan` [nan, nan]` over `n=0` dialogues
- `commitment_post_counter_adherence`: mean `nan` [nan, nan]` over `n=0` dialogues

## Source Monitoring Attribution

- `source_monitoring_choose_self`: mean `0.4062` [0.3125, 0.5000]` over `n=16` pairs
- `source_monitoring_swap_consistency`: mean `0.1875` [0.0000, 0.3750]` over `n=16` pairs
- `source_monitoring_family_self_minus_baseline_choose_self`: mean `0.0625` [0.0000, 0.1875]` over `n=8` pairs
- `baseline_helpful / other_frame / paraphrased`: choose-self `0.5000`, swap-consistency `0.0000`, n=`2`
- `baseline_helpful / other_frame / raw`: choose-self `0.2500`, swap-consistency `0.5000`, n=`2`
- `baseline_helpful / same_frame_other_seed / paraphrased`: choose-self `0.5000`, swap-consistency `0.0000`, n=`2`
- `baseline_helpful / same_frame_other_seed / raw`: choose-self `0.2500`, swap-consistency `0.5000`, n=`2`
- `family_self / other_frame / paraphrased`: choose-self `0.5000`, swap-consistency `0.0000`, n=`2`
- `family_self / other_frame / raw`: choose-self `0.2500`, swap-consistency `0.5000`, n=`2`
- `family_self / same_frame_other_seed / paraphrased`: choose-self `0.5000`, swap-consistency `0.0000`, n=`2`
- `family_self / same_frame_other_seed / raw`: choose-self `0.5000`, swap-consistency `0.0000`, n=`2`
