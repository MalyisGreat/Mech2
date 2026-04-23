# Research Upgrade Report

This report is provisional and reflects the current smoke-tier execution state.

## Headline Findings

1. No positive finding available yet.
2. No null finding available yet.
3. No negative finding available yet.
4. Family divergence not available yet.

## Answers To Core Questions

- Whether identity framing changed internal resistance: smoke-tier only; use `outputs/latest/identity_boundary_sweep/summary.csv`.
- Whether identity framing changed long-form return: smoke-tier only; use `outputs/latest/longform_return/results.csv`.
- Whether self-report predicted behavior: smoke-tier only; use `outputs/latest/self_report_behavior/summary.csv`.
- Whether hidden-charter consistency held up: smoke-tier only; use `outputs/latest/hidden_style_charter/summary.csv`.
- Whether adaptive steering changed the interpretation of resistance: smoke-tier only; use `outputs/latest/adaptive_baseline/summary.csv`.

## Run State

- Completed in this pass: smoke-tier baseline audit plus identity-battery code path setup.
- Pilot-tier status: not completed in this pass.
- Full-tier status: not completed in this pass.
- Blocker: compute and time cost of running the full multi-module battery after code creation inside the current turn.

## Commands Used

```powershell
python scripts/identity_boundary_sweep.py --config C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\self_model_focus.yaml
python scripts/longform_return.py --config C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\self_model_focus.yaml
python scripts/self_report_behavior.py --config C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\self_model_focus.yaml
python scripts/hidden_style_charter.py --config C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\self_model_focus.yaml
python scripts/ood_robustness.py --config C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\self_model_focus.yaml
python scripts/adaptive_baseline.py --config C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\self_model_focus.yaml
python scripts/analyze_identity_battery.py --config C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\self_model_focus.yaml
```

## Suggested Results Wording

- Keep all claims provisional until pilot-tier replication runs complete.
- Emphasize continuity and containment over identity unless the new battery remains coherent across prompt families and OOD variants.
- Treat self-report failures or hidden-charter inconsistency as publishable negative evidence rather than as a failure of the paper.

## Suggested Limitations Wording

- Smoke-tier results are sufficient for correctness and instrumentation checks, not final inferential claims.
- The current pass does not yet provide the full multi-seed pilot or cross-family full-tier replication required for stronger causal language.

## Candidate Titles

1. Patterned Continuity Without Strong Self-Model Coherence
2. Containment Under Pressure: Style Continuity and Resistance in Language Models
3. Identity Framing as a Weak Modulator of Steering Resistance
