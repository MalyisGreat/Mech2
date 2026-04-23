# Diachronic Ship-of-Theseus Identity Graft Report

- Config: `C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\diachronic_ship_of_theseus_graft_overnight.yaml`
- Output root: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_overnight`
- Model: `EleutherAI/pythia-410m-deduped`
- Selected prompts: `8` from `120` screened prompts
- Pair directions: `later_host_young_donor, young_host_later_donor`

## Prompt Selection

- `clean_prompt_js_selected`: mean `0.0501` [0.0393, 0.0634]` over `n=8` prompts
- `clean_prompt_js_all_candidates`: mean `0.0205` [0.0187, 0.0226]` over `n=120` prompts

## Headline Results

- `later_host_young_donor_primary_single_layer_last_prompt_lambda1_dif`: mean `0.5064` [0.4527, 0.5603]` over `n=40` prompt-level cells
- `young_host_later_donor_primary_single_layer_last_prompt_lambda1_dif`: mean `0.4689` [0.4181, 0.5213]` over `n=40` prompt-level cells
- `later_host_young_donor_primary_prefix_last_prompt_lambda1_dif`: mean `0.5064` [0.4527, 0.5603]` over `n=40` prompt-level cells
- `later_host_young_donor_primary_suffix_last_prompt_lambda1_dif`: mean `0.4670` [0.4426, 0.4933]` over `n=40` prompt-level cells
- `name_only_last_prompt_dif`: mean `0.5001` [0.4544, 0.5535]` over `n=16` prompt-level cells

## Control Checks

- `control_primary_donor_identity_fraction`: mean `0.5064` [0.4527, 0.5603]` over `n=40` prompt-level cells
- `control_adjacent_donor_identity_fraction`: mean `0.3873` [0.3100, 0.4664]` over `n=40` prompt-level cells
- `control_very_early_donor_identity_fraction`: mean `0.4715` [0.4135, 0.5286]` over `n=40` prompt-level cells
- `control_random_same_norm_donor_identity_fraction`: mean `0.4445` [0.4106, 0.4767]` over `n=40` prompt-level cells
- `control_shuffled_prompt_donor_identity_fraction`: mean `0.4831` [0.4493, 0.5198]` over `n=40` prompt-level cells
- `control_name_only_donor_identity_fraction`: mean `nan` [nan, nan]` over `n=0` prompt-level cells

## Secondary Metrics

- `primary_cad`: mean `240.1243` [221.8771, 258.4311]` over `n=80` prompt-level cells
- `primary_recovery_fraction`: mean `0.1892` [0.1520, 0.2244]` over `n=80` prompt-level cells
- `primary_persistence`: mean `0.8108` [0.7756, 0.8480]` over `n=80` prompt-level cells
- `primary_next_token_kl`: mean `0.2967` [0.2456, 0.3539]` over `n=80` prompt-level cells
- `primary_activation_norm_deviation`: mean `0.1097` [0.0681, 0.1612]` over `n=80` prompt-level cells
- `primary_text_donor_identity_fraction`: mean `0.2601` [0.2063, 0.3131]` over `n=80` prompt-level cells
- `primary_semantic_donor_identity_fraction`: mean `0.2451` [0.1934, 0.2968]` over `n=80` prompt-level cells
- `self_report_causal_correlation`: `0.0622` over `n=12032` prompt-level cells
- `self_report_minus_causal_abs_error`: mean `0.5201` [0.5134, 0.5269]` over `n=12032` prompt-level cells

## Figures

- Heatmap, last prompt token: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_overnight\figures\donor_identity_heatmap_last_prompt.png`
- Heatmap, first prompt token: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_overnight\figures\donor_identity_heatmap_first_prompt.png`
- Prefix vs suffix curves: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_overnight\figures\prefix_suffix_curves_last_prompt.png`
- Control comparison: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_overnight\figures\control_comparison_last_prompt.png`
- Self-report vs causal identity: `C:\Users\joshj\joseph-stroud-identity-stability-research\outputs\latest\diachronic_ship_of_theseus_graft_overnight\figures\self_report_vs_causal_identity.png`