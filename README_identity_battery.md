# Identity Battery

This extension layer adds an identity bridge battery on top of the existing steering pipeline without changing the baseline Pythia backbone.

## Baseline-first artifacts

Before adding new code, the current paper implementation was mapped and checked in:

- `outputs/latest/repo_map.md`
- `outputs/latest/repro_baseline_check.md`

## New entrypoints

- `scripts/identity_boundary_sweep.py`
- `scripts/longform_return.py`
- `scripts/self_report_behavior.py`
- `scripts/hidden_style_charter.py`
- `scripts/ood_robustness.py`
- `scripts/adaptive_baseline.py`
- `scripts/analyze_identity_battery.py`

## Shared utilities

- `src/identity_stability/identity_data.py`
- `src/identity_stability/text_features.py`
- `src/identity_stability/steered_generation.py`
- `src/identity_stability/identity_analysis.py`

## Tiered configs

- `configs/identity_battery/smoke.yaml`
- `configs/identity_battery/pilot.yaml`
- `configs/identity_battery/full.yaml`

## Recommended execution order

```powershell
python scripts/identity_boundary_sweep.py --config configs/identity_battery/smoke.yaml
python scripts/longform_return.py --config configs/identity_battery/smoke.yaml
python scripts/self_report_behavior.py --config configs/identity_battery/smoke.yaml
python scripts/hidden_style_charter.py --config configs/identity_battery/smoke.yaml
python scripts/ood_robustness.py --config configs/identity_battery/smoke.yaml
python scripts/adaptive_baseline.py --config configs/identity_battery/smoke.yaml
python scripts/analyze_identity_battery.py --config configs/identity_battery/smoke.yaml
```

## Output layout

All new artifacts are written under `outputs/latest/`:

- `outputs/latest/identity_boundary_sweep/`
- `outputs/latest/longform_return/`
- `outputs/latest/self_report_behavior/`
- `outputs/latest/hidden_style_charter/`
- `outputs/latest/ood_robustness/`
- `outputs/latest/adaptive_baseline/`
- `outputs/latest/figures/`
- `outputs/latest/research_upgrade_report.md`
- `outputs/latest/manuscript_patch_notes.md`

## Scientific guardrail

This battery is designed to support either of two publishable outcomes:

1. identity-relevant continuity is real, or
2. continuity exists without robust self-model coherence.

The code should not bias toward the first outcome.
