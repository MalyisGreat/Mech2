# Identity Probe Decoding Hardening Report

## Purpose

This pass hardens the identity-focused probes only. It does **not** alter the baseline backbone suite or the token-phase suite.

The immediate problem was measurement contamination: several identity-probe result files contained generated texts with literal `User:` / `Assistant:` scaffolding inside the candidate answers themselves. That made the old self-recognition and self/other-boundary conclusions too dependent on a bad decoding path.

## Code And Config Changes

- Added identity-only prompt-template and stop-string support in:
  - `src/identity_stability/steered_generation.py`
  - `scripts/identity_battery_common.py`
- Switched the main identity probes to use configurable clean prompting:
  - `scripts/self_prediction_calibration.py`
  - `scripts/self_report_behavior.py`
  - `scripts/self_recognition_from_foils.py`
  - `scripts/self_other_boundary.py`
- Hardened boundary reporting to add clustered summaries and clustered sign tests:
  - `scripts/report_self_other_boundary.py`
- Added clean rerun configs:
  - `configs/identity_battery/self_prediction_clean_smoke.yaml`
  - `configs/identity_battery/self_recognition_1b_family_balanced_clean.yaml`
  - `configs/identity_battery/self_other_boundary_confirm_clean.yaml`

## Commands Run

```powershell
python -m py_compile src\identity_stability\steered_generation.py scripts\identity_battery_common.py scripts\self_prediction_calibration.py scripts\self_report_behavior.py scripts\self_recognition_from_foils.py scripts\self_other_boundary.py scripts\report_self_other_boundary.py scripts\report_self_recognition_confirm.py
python scripts\self_prediction_calibration.py --config configs\identity_battery\self_prediction_clean_smoke.yaml
python scripts\self_other_boundary.py --config configs\identity_battery\self_other_boundary_confirm_clean.yaml --smoke
python scripts\report_self_other_boundary.py --config configs\identity_battery\self_other_boundary_confirm_clean.yaml --smoke
python scripts\self_other_boundary.py --config configs\identity_battery\self_other_boundary_confirm_clean.yaml
python scripts\report_self_other_boundary.py --config configs\identity_battery\self_other_boundary_confirm_clean.yaml
python scripts\self_recognition_from_foils.py --config configs\identity_battery\self_recognition_1b_family_balanced_clean.yaml
python scripts\report_self_recognition_confirm.py --config configs\identity_battery\self_recognition_1b_family_balanced_clean.yaml
```

## Leakage Audit

- Old targeted self-recognition run:
  - `rows = 720`
  - `accuracy = 0.4236`
  - `contains User: = 0.9167`
  - `contains Assistant: = 0.9167`
- Clean targeted self-recognition rerun:
  - `rows = 720`
  - `accuracy = 0.3403`
  - `contains User: = 0.0000`
  - `contains Assistant: = 0.0000`
- Old self/other-boundary partial run:
  - `rows = 48`
  - `boundary_match_no_steer = 0.0833`
  - `boundary_match_steer = 0.0833`
  - `contains User: = 0.9167`
  - `contains Assistant: = 0.9167`
- Clean self/other-boundary rerun:
  - `rows = 72`
  - `boundary_match_no_steer = 0.4028`
  - `boundary_match_steer = 0.3889`
  - `contains User: = 0.0000`
  - `contains Assistant: = 0.0000`

The hardening successfully removed scaffold leakage from the clean reruns.

## Clean Self-Recognition Rerun

Source:

- `outputs/self_recognition_1b_family_balanced_clean/self_recognition_from_foils/confirm_report.md`

Key result:

- Clean `1b / family_self` self-recognition accuracy: `0.3403`
- Row-bootstrap 95% CI: `[0.3055, 0.3764]`
- Cluster-bootstrap 95% CI: `[0.2986, 0.3889]`
- Chance: `0.3333`
- Cluster sign-test p-value: `0.636719`
- Cluster Holm-adjusted p-value: `0.636719`

Comparison to old targeted run:

- Old targeted `1b / family_self`: `0.4236`
- Clean targeted `1b / family_self`: `0.3403`

Interpretation:

The old local answer-ownership pocket does **not** survive the clean decoding rerun in anything like its previous strength. After leakage removal, the targeted `1b / family_self` result is effectively near chance.

Axis pattern in the clean rerun:

- `collaborative_vs_authoritative`: `0.4167`
- `expansive_vs_terse`: `0.3333`
- `cautious_vs_assertive`: `0.3056`
- `selfref_vs_impersonal`: `0.3056`

This is weak and heterogeneous, not a stable local self-recognition effect.

## Clean Self/Other Boundary Rerun

Source:

- `outputs/self_other_boundary_confirm_clean/self_other_boundary/confirm_report.md`

Key result:

- Clean `1b / family_self` no-steer boundary match: `0.4028`
- Clean `1b / family_self` steered boundary match: `0.3889`
- Transfer delta: `-0.0139`
- Cluster-bootstrap 95% CI for no-steer match: `[0.2083, 0.5833]`
- Cluster sign-test p-value: `0.846272`
- Cluster Holm-adjusted p-value: `0.846272`

Comparison to old partial contaminated run:

- Old partial no-steer boundary match: `0.0833`
- Clean full no-steer boundary match: `0.4028`
- Old partial steered boundary match: `0.0833`
- Clean full steered boundary match: `0.3889`

Interpretation:

The earlier dramatic boundary-collapse result was **not** stable. With clean answer text and a full 72-row run, the boundary probe becomes a mixed local signal: descriptively above chance, statistically weak under clustered inference, and only minimally reduced by contrary steering.

Axis pattern in the clean rerun:

- `selfref_vs_impersonal`: `1.0000`
- `cautious_vs_assertive`: `0.5000`
- `expansive_vs_terse`: `0.1111`
- `collaborative_vs_authoritative`: `0.0000`

So the clean boundary result is strongly axis-dependent rather than a general self/other-boundary phenomenon.

## Paper-Level Takeaway

This hardening pass weakens the identity-forward reading.

- The strongest old local self-recognition pocket (`1b / family_self`) drops from `0.4236` to `0.3403` once the decoding contamination is removed.
- The strongest old boundary-collapse result also fails to hold in the same dramatic form after cleaning; it becomes a modest, mixed, axis-sensitive signal instead of a strong negative result.

The most defensible interpretation after this pass is:

`The identity probes are highly measurement-sensitive, and once decoding contamination is removed they support at most weak, local, axis-specific self-related structure rather than robust self-model coherence.`

That pushes the paper even more clearly toward:

`patterned continuity / containment without strong self-model coherence`

rather than toward a stronger identity claim.

## Recommendation

If additional identity work is run after this pass, it should use the clean decoding path by default. But the paper payoff from identity probing is now mostly constraint-setting, not confirmation of strong selfhood.

The stronger research move after this point is likely to keep the containment / token-phase backbone central and treat identity probes as fragile, mixed add-ons unless a clean rerun produces a much stronger replicated effect than this one did.
