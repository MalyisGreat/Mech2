# Diachronic Ship-of-Theseus Identity Graft

- Config: `C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\diachronic_ship_of_theseus_graft_overnight.yaml`
- Model: `EleutherAI/pythia-410m-deduped`
- Selected prompt count: `8` from `120` candidate prompts
- Prompt bank: `C:\Users\joshj\joseph-stroud-identity-stability-research\data\diachronic_ship_of_theseus_prompts.yaml`
- Pairs: `later_host_young_donor, young_host_later_donor`
- Token positions: `[-1, 0]`
- Layer buckets: `early, early_middle, middle, late_middle, late`
- Lambdas: `[0.0, 0.25, 0.5, 0.75, 1.0]`
- Graft modes: `['single_layer', 'prefix', 'suffix']`
- Token position labels follow the current repo convention: `-1 = last prompt token`, `0 = first prompt token`.
