# Diachronic Ship-of-Theseus Identity Graft Report

- Config: `configs\identity_battery\diachronic_ship_of_theseus_graft_debug.yaml`
- Output root: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_debug`
- Model: `EleutherAI/pythia-410m-deduped`
- Selected prompts: `2` from `8` screened prompts
- Pair directions: `later_host_young_donor`

## Prompt Selection

- `clean_prompt_js_selected`: mean `0.0310` [0.0252, 0.0367]` over `n=2` prompts
- `clean_prompt_js_all_candidates`: mean `0.0213` [0.0169, 0.0266]` over `n=8` prompts

## Headline Results

- `later_host_young_donor_primary_single_layer_last_prompt_lambda1_dif`: mean `0.4996` [0.4266, 0.5573]` over `n=6` prompt-level cells
- `young_host_later_donor_primary_single_layer_last_prompt_lambda1_dif`: mean `nan` [nan, nan]` over `n=0` prompt-level cells
- `later_host_young_donor_primary_prefix_last_prompt_lambda1_dif`: mean `0.4996` [0.4266, 0.5573]` over `n=6` prompt-level cells
- `later_host_young_donor_primary_suffix_last_prompt_lambda1_dif`: mean `0.4361` [0.4352, 0.4369]` over `n=6` prompt-level cells
- `name_only_last_prompt_dif`: mean `0.4910` [0.4584, 0.5236]` over `n=2` prompt-level cells

## Control Checks

- `control_primary_donor_identity_fraction`: mean `0.4996` [0.4266, 0.5573]` over `n=6` prompt-level cells
- `control_adjacent_donor_identity_fraction`: mean `0.3034` [0.1230, 0.4825]` over `n=6` prompt-level cells
- `control_very_early_donor_identity_fraction`: mean `0.3929` [0.2375, 0.5201]` over `n=6` prompt-level cells
- `control_random_same_norm_donor_identity_fraction`: mean `0.4866` [0.3890, 0.5466]` over `n=6` prompt-level cells
- `control_shuffled_prompt_donor_identity_fraction`: mean `0.4728` [0.3848, 0.5535]` over `n=6` prompt-level cells
- `control_name_only_donor_identity_fraction`: mean `nan` [nan, nan]` over `n=0` prompt-level cells

## Secondary Metrics

- `primary_cad`: mean `173.9516` [144.6875, 203.3853]` over `n=6` prompt-level cells
- `primary_recovery_fraction`: mean `0.0834` [0.0000, 0.1795]` over `n=6` prompt-level cells
- `primary_persistence`: mean `0.9166` [0.8205, 1.0000]` over `n=6` prompt-level cells
- `primary_next_token_kl`: mean `0.1765` [0.0963, 0.2588]` over `n=6` prompt-level cells
- `primary_activation_norm_deviation`: mean `0.2277` [0.0220, 0.4535]` over `n=6` prompt-level cells
- `primary_text_donor_identity_fraction`: mean `0.0833` [0.0000, 0.2500]` over `n=6` prompt-level cells
- `primary_semantic_donor_identity_fraction`: mean `0.0642` [-0.0000, 0.1926]` over `n=6` prompt-level cells
- `self_report_causal_correlation`: `nan` over `n=182` prompt-level cells
- `self_report_minus_causal_abs_error`: mean `0.7682` [0.7324, 0.8049]` over `n=182` prompt-level cells

## Figures

- Heatmap, last prompt token: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_debug\figures\donor_identity_heatmap_last_prompt.png`
- Heatmap, first prompt token: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_debug\figures\donor_identity_heatmap_first_prompt.png`
- Prefix vs suffix curves: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_debug\figures\prefix_suffix_curves_last_prompt.png`
- Control comparison: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_debug\figures\control_comparison_last_prompt.png`
- Self-report vs causal identity: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_debug\figures\self_report_vs_causal_identity.png`