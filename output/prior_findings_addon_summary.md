# Prior Findings Add-On Summary (Current Models)

## What Was Added
- New concepts for transfer from prior experiments:
  - `morality` (`good` vs `evil`)
  - `constructiveness` (`create/build` vs `destroy/sabotage`)
- Prompt-style tagging and controls:
  - `factual`, `technical`, `emotional`, `ambiguous`
- New per-row metrics:
  - `drift_at_start`, `drift_at_start_relative`
  - `cad`, `cad_relative`
  - `degradation`, `persistence`
- New config axis:
  - `estimation_token_position` (decouples vector-estimation token from evaluation token)
- New suite outputs:
  - `suite_prompt_style_summary.csv`
  - `suite_scaling_laws.csv`

## Runs Executed
- Smoke validation:
  - `runs/20260226_225232`
- Current-model pilot run:
  - `runs/20260226_225516`
- Token-position suite v2 (`token_position=-1` and `0`, with `estimation_token_position=-1`):
  - `runs/prior_findings_token_position_v2_20260226_230323`

## Token Position Findings (v2)
From `runs/prior_findings_token_position_v2_20260226_230323/suite_metrics_full.csv`:

All vector methods (mean over rows):
- `token_position=-1`: `cad=2.2715`, `degradation=0.1592`, `persistence=0.7277`, `recovery=0.1056`
- `token_position=0`: `cad=3.6470`, `degradation=0.1702`, `persistence=0.8043`, `recovery=0.0290`

Concept vectors only (`mean_diff` + `linear_probe`):
- `token_position=-1`: `cad=1.2077`, `degradation=0.0195`, `persistence=0.6359`, `recovery=0.0308`
- `token_position=0`: `cad=1.4907`, `degradation=0.0273`, `persistence=0.6667`, `recovery=0.0000`

Interpretation: early-token injection (`0`) is more persistent and recovers less than final-token injection (`-1`) in this pilot.

## Scaling-Law Fit Snapshot (v2)
From `runs/prior_findings_token_position_v2_20260226_230323/suite_scaling_laws.csv`:

- `mean_diff`:
  - `cad ~ 77.19 * params^-0.2038` (`R^2=0.6115`)
  - `degradation ~ 0.00235 * params^0.1109` (`R^2=0.1260`)
  - `persistence ~ 0.8343 * params^-0.0124` (`R^2=0.2140`)
- `linear_probe`:
  - `cad ~ 50.98 * params^-0.1842` (`R^2=0.5971`)
  - `degradation ~ 0.00087 * params^0.1582` (`R^2=0.2143`)
  - `persistence ~ 0.8293 * params^-0.0121` (`R^2=0.2054`)

## Prompt-Style Snapshot (v2)
From `runs/prior_findings_token_position_v2_20260226_230323/suite_prompt_style_summary.csv`:

Degradation ranking (highest to lowest): `emotional`, `technical`, `ambiguous`, `factual`.
Range is narrow in this pilot (`0.1654` to `0.1644` mean degradation).

## Notes
- A NumPy compatibility issue was fixed (`np.trapezoid`/`np.trapz` fallback) after initial smoke failure.
- Initial token-position run had degenerate concept vectors at `token_position=0`; fixed by adding `estimation_token_position`.
