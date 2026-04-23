# Research Upgrade Report

This report is provisional and reflects a completed smoke tier plus a partial pilot tier.

## Headline Findings

1. Strongest positive result remains the baseline backbone rather than a new identity effect: within Pythia, larger models are less behaviorally perturbed and show stronger disturbance containment.
2. Strongest moderator remains token position / trajectory phase from the cached add-on results already documented in the baseline audit.
3. Strongest null finding after partial pilot expansion: self-report versus behavior coupling remains extremely small overall in the full-Pythia partial pilot, with mean coupling magnitude around `0.000060` across frames and sizes.
4. Strongest negative bridge finding after partial pilot expansion: hidden-charter consistency is only modest even with no steering and degrades further under contrary steering, dropping from mean consistency `0.4083` in `no_steer` to `0.1750` in `authoritative_push`.

## Answers To Core Questions

- Whether identity framing changed internal resistance: only smoke-tier evidence exists in this pass; use `outputs/latest/identity_boundary_sweep/summary.csv`.
- Whether identity framing changed long-form return: only smoke-tier evidence exists in this pass, and the current mixed-model output is not strong enough for headline use; treat it as provisional.
- Whether self-report predicted behavior: partial pilot evidence argues mostly no. In `outputs/pilot_partial_20260409/self_report_behavior/summary.csv`, coupling magnitudes stay near zero across frames, sizes, and dimensions.
- Whether hidden-charter consistency held up: partial pilot evidence argues weakly at best, but the current probe is partly confounded by measurement breakdown under steering.
- Whether adaptive steering changed the interpretation of resistance: only smoke-tier evidence exists in this pass and should not be used as a headline result yet.

## Recommended Paper Shape

1. Backbone result: larger Pythia models are less behaviorally perturbed.
2. Major moderator: token position / phase-of-trajectory strongly shapes persistence.
3. Identity bridge result: the added probes currently provide weak or negative evidence for strong self-model coherence.

Suggested thesis sentence:

`Our strongest evidence supports disturbance containment rather than universal self-restoration: larger models often damp internally induced perturbations more effectively, while added identity probes provide limited evidence that this robustness is mediated by stable explicit self-modeling.`

## Run State

- Completed in this pass: baseline audit, full smoke-tier identity battery, and partial pilot-tier runs for `hidden_style_charter` and `self_report_behavior` across the six-model Pythia family.
- Pilot-tier status: partially completed in this pass.
- Full-tier status: not completed in this pass.
- Blocker: the full pilot and full-tier boundary-style grid still expands across 6 models x 4 frames x 4 axes x 5 layer choices x 2 token sites x 6 strengths x 3 seeds x 3 vector kinds, before counting prompts, making same-turn execution impractical after implementation.

## Commands Used

```powershell
python scripts/identity_boundary_sweep.py --config configs\identity_battery\smoke.yaml
python scripts/longform_return.py --config configs\identity_battery\smoke.yaml
python scripts/self_report_behavior.py --config configs\identity_battery\smoke.yaml
python scripts/hidden_style_charter.py --config configs\identity_battery\smoke.yaml
python scripts/ood_robustness.py --config configs\identity_battery\smoke.yaml
python scripts/adaptive_baseline.py --config configs\identity_battery\smoke.yaml
python scripts/analyze_identity_battery.py --config configs\identity_battery\smoke.yaml
python scripts/hidden_style_charter.py --config configs\identity_battery\pilot_partial.yaml
python scripts/self_report_behavior.py --config configs\identity_battery\pilot_partial.yaml
```

## Suggested Results Wording

- Keep all boundary, long-form, OOD, and adaptive claims provisional until broader pilot replication runs complete.
- Lean into the negative bridge result: explicit self-description does not reliably predict behavior, and hidden commitments are weak and easily disrupted.
- Emphasize continuity and containment over identity unless long-form return, hidden-charter consistency, and OOD robustness converge in later pilot/full runs.

## Suggested Limitations Wording

- Smoke-tier results are sufficient for correctness and instrumentation checks, not final inferential claims.
- The current pass adds a partial pilot on hidden-charter consistency and self-report coupling, but not yet the full multi-module pilot or cross-family full-tier replication required for stronger causal language.
- The current pilot summaries are descriptive only at the cell level because `summary.csv` rows have `n = 1`.
- The current charter probe is partly a measurement problem: `no_steer` rows reveal charter `A` throughout, and `valid_answer_count` drops sharply under contrary steering, so part of the observed consistency loss may reflect instruction-following failure rather than only hidden-identity disruption.

## Candidate Titles

1. Patterned Continuity Without Strong Self-Model Coherence
2. Scaling, Trajectory Phase, and Weak Self-Model Coherence in Language Models
3. Identity Framing as a Weak Modulator of Steering Resistance

## 2026-04-22 Ship-of-Theseus Graft Blocker

- Implemented: `scripts/diachronic_ship_of_theseus_graft.py`, prompt bank `data/diachronic_ship_of_theseus_prompts.yaml`, and report/config stack under `configs/identity_battery/`.
- Exact full-grid design implemented: two directional checkpoint pairs, token positions `-1` and `0`, five layer buckets, lambdas `0.00/0.25/0.50/0.75/1.00`, graft modes `single_layer/prefix/suffix`, and controls `primary/adjacent/very_early/random_same_norm/shuffled_prompt/name_only`.
- Compute blocker: the exact 120-prompt grid expands to roughly `180,480` graft rows before counting clean-cache warmup across checkpoint revisions. An initial larger smoke reached only `100` graft rows after the expensive revision-precompute stage, making the literal 120-prompt first launch impractical on the current machine as an overnight run.
- Action taken: kept the exact full config for reproducibility, added a tiny debug validator, and launched an overnight-sized run that preserves the full causal design but uses `8` selected prompts from the 120-prompt neutral bank.
