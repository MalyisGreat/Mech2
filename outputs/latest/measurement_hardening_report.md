# Measurement Hardening Report

## Purpose

This pass addressed the highest-risk audit findings in the identity-focused probes:

- forced digit-label bias in self-recognition from foils
- row-level pseudoreplication in confirmatory self-recognition reporting
- forced digit-label bias in prompt-conditional self-prediction

The goal was not to add a new identity claim. The goal was to test whether the existing identity-adjacent findings survive stricter measurement.

## Code Changes

- `scripts/self_recognition_from_foils.py` now supports `self_recognition_choice_mode: balanced_permutations`, which evaluates all six assignments of the three candidate answers to digit labels `1`, `2`, and `3`.
- `scripts/report_self_recognition_confirm.py` now reports clustered estimates and cluster sign tests by prompt/axis/strength cells. Row-level binomial tests are retained as diagnostics only.
- `scripts/self_prediction_calibration.py` now supports digit-label bias correction by subtracting a neutral digit-calibration prompt from the prediction prompt logits.
- `src/identity_stability/identity_analysis.py` now includes reusable clustered mean and clustered bootstrap helpers.

## Commands Run

```powershell
python -m py_compile scripts\self_recognition_from_foils.py scripts\report_self_recognition_confirm.py scripts\self_prediction_calibration.py src\identity_stability\identity_analysis.py
python scripts\report_self_recognition_confirm.py --config configs\identity_battery\self_recognition_confirm.yaml
python scripts\self_recognition_from_foils.py --config configs\identity_battery\self_recognition_confirm_balanced_smoke.yaml
python scripts\report_self_recognition_confirm.py --config configs\identity_battery\self_recognition_confirm_balanced_smoke.yaml
python scripts\self_prediction_calibration.py --config configs\identity_battery\self_prediction_bias_smoke.yaml
python scripts\self_recognition_from_foils.py --config configs\identity_battery\self_recognition_1b_family_balanced_target.yaml
python scripts\report_self_recognition_confirm.py --config configs\identity_battery\self_recognition_1b_family_balanced_target.yaml
python scripts\self_recognition_from_foils.py --config configs\identity_battery\self_recognition_confirm_balanced.yaml
python scripts\report_self_recognition_confirm.py --config configs\identity_battery\self_recognition_confirm_balanced.yaml
```

## Existing Legacy Confirm Reanalysis

Output:

`outputs/self_recognition_confirm/self_recognition_from_foils/confirm_report.md`

The old legacy confirm result still has a strong row-level signal for `1b / family_self`, but the clustered correction weakens it:

- overall accuracy: `0.3193`, below chance `0.3333`
- `1b / family_self` row accuracy: `0.5333`
- clustered target estimate: `0.5333`
- cluster-bootstrap 95% CI: `[0.3917, 0.6667]`
- cluster sign-test: `17/24` clusters above chance in the old report, but after explicit tie handling the valid non-tie denominator is reported separately
- Holm-adjusted clustered result on legacy full grid: not strong enough to treat as a broad confirmatory identity result

Important diagnostic:

- selected label counts in the old target cell were heavily label-biased: label `1` was selected `77/120` times
- target accuracy depended strongly on where the self answer landed: self at label `1` had accuracy about `0.805`, versus `0.360` at label `2` and `0.448` at label `3`

Interpretation:

The legacy self-recognition bump was partly contaminated by digit-label bias. The old p-values should not be used as manuscript evidence.

## Balanced Smoke

Output:

`outputs/self_recognition_confirm_balanced_smoke/self_recognition_from_foils/confirm_report.md`

Design:

- models: `70m`, `1b`
- frames: `baseline_helpful`, `family_self`
- axes: `expansive_vs_terse`, `selfref_vs_impersonal`
- seeds: `42`, `123`
- rows: `192`
- all six answer-label permutations used

Results:

- overall balanced accuracy: `0.2812`
- `1b / family_self`: `0.2500`
- no smoke cell was above chance
- self-baseline appeared exactly evenly under labels `1`, `2`, and `3`

Interpretation:

The smoke showed that the new balanced-permutation path works and that the old positive cell could collapse under label balancing in small samples. This justified a targeted confirmatory rerun.

## Targeted Balanced Rerun: `1b / family_self`

Output:

`outputs/self_recognition_1b_family_balanced_target/self_recognition_from_foils/confirm_report.md`

Design:

- model: `EleutherAI/pythia-1b`
- frame: `family_self`
- axes: all four self-recognition axes
- seeds: `42`, `123`, `314`, `1618`, `2718`
- prompt limit: `6`
- rows: `720`
- all six answer-label permutations used

Main result:

- overall balanced accuracy: `0.4236`
- row hits: `305/720`
- clustered estimate: `0.4236`
- cluster-bootstrap 95% CI: `[0.3403, 0.5000]`
- cluster sign test: `12/16` non-tie clusters above chance, with `8` exact chance ties
- cluster sign-test p-value: `0.038406`

Axis breakdown:

- `expansive_vs_terse`: `0.5000`
- `collaborative_vs_authoritative`: `0.5000`
- `selfref_vs_impersonal`: `0.3889`
- `cautious_vs_assertive`: `0.3056`

Label-bias diagnostic:

- self-baseline label assignment was exactly balanced: `240` rows each under labels `1`, `2`, and `3`
- selected labels remained biased toward `1`: `460` label-1 selections, `130` label-2 selections, `130` label-3 selections
- because self labels were balanced, label-1 bias alone cannot explain above-chance accuracy, but it remains a measurement caveat

Interpretation:

The `1b / family_self` self-recognition pocket survives the strongest immediate label-bias fix, but it is weaker than the legacy result and not broad. It should be described as a local, axis-heterogeneous pocket of answer ownership, not as general self-recognition or a strong identity result.

## Full Balanced Confirm Grid

Output:

`outputs/self_recognition_confirm_balanced/self_recognition_from_foils/confirm_report.md`

Design:

- models: `70m`, `410m`, `1b`, `2.8b`
- frames: `baseline_helpful`, `instance_self`, `family_self`, `tool_only`
- axes: all four self-recognition axes
- seeds: `42`, `123`, `314`, `1618`, `2718`
- prompt limit: `6`
- rows: `11,520`
- all six answer-label permutations used

Main result:

- overall accuracy: `0.3138`, below chance `0.3333`
- overall clustered 95% CI: `[0.2999, 0.3290]`
- cluster sign test: `78/195` non-tie clusters above chance with `189` exact chance ties
- overall cluster sign-test p-value: `0.997970`

Strongest model/frame cell:

- `1b / family_self`: `0.4236`
- row hits: `305/720`
- clustered 95% CI: `[0.3403, 0.5000]`
- cluster sign test: `12/16` non-tie clusters above chance with `8` exact chance ties
- unadjusted cluster sign-test p-value: `0.038406`
- full-grid Holm-adjusted clustered p-value across model/frame cells: `0.614502`

Nearby and larger-model checks:

- `70m / family_self`: `0.2986`
- `410m / family_self`: `0.2847`
- `2.8b / family_self`: `0.2292`
- `1b / baseline_helpful`: `0.3542`
- `1b / instance_self`: `0.3681`
- `1b / tool_only`: `0.2639`

Interpretation:

The full balanced grid preserves the `1b / family_self` pocket as the strongest descriptive cell, but it does not support a broad or monotonic identity claim. The pocket fails to generalize downward to `70m` or `410m`, fails upward at `2.8b`, and does not survive multiplicity-corrected clustered inference across the full model/frame grid. The best reading is a local answer-ownership effect under one model/frame combination, embedded in an overall negative self-recognition result.

Measurement caveat:

Even after answer-label balancing, selected-label priors remain large. The strongest skew appears in `70m / baseline_helpful`, which selected one digit label at rate `0.8611`; the `1b / family_self` cell still selected label `1` at rate `0.6389`. Balanced placement prevents this from trivially explaining the whole `1b / family_self` bump, but the residual label preference means forced-choice identity probes remain fragile.

## Bias-Corrected Self-Prediction Smoke

Output:

`outputs/self_prediction_bias_smoke/self_prediction_calibration/`

Design:

- models: `70m`, `1b`
- frames: `baseline_helpful`, `family_self`
- axes: `expansive_vs_terse`, `selfref_vs_impersonal`
- prompt limit: `2`
- rows: `16`
- digit-label bias correction enabled

Results:

- overall sign accuracy: `0.5625`
- overall calibration error: `0.6887`
- `1b / family_self`: `0.0000` sign accuracy in this small smoke
- `70m / family_self`: `0.7500` sign accuracy in this small smoke

Interpretation:

The older self-prediction result is not ready as a manuscript-grade identity result. Bias correction does not destroy all direction-prediction behavior, but the frame-specific pattern is unstable in this smoke.

## Hardened Self/Other Boundary Follow-Up

Output:

`outputs/self_other_boundary_confirm/self_other_boundary/confirm_report.md`

Design:

- hardened 5-way forced-choice boundary readout replaced the old paired YES/NO boundary probe
- digit-label bias correction enabled
- focus: `family_self` vs `tool_only`
- models targeted: `1b`, `2.8b`
- axes: all four identity axes
- seeds requested: `11`, `17`, `23`
- current artifact is a partial checkpoint after the first `48` `1b / family_self` rows, written intentionally because the early result was already decisive

Main partial result:

- `1b / family_self` no-steer boundary match: `0.0833` against `0.3333` chance
- steered boundary match: `0.0833`
- transfer delta: `0.0000`
- bootstrap 95% CI for no-steer boundary match: `[0.0208, 0.1667]`
- exact p-value vs chance: `0.999991`
- self-prediction advantage remained positive: `0.2170`
- self-moved-toward-other rate under steering: `0.3333`

Interpretation:

This is the strongest new constraint added in this pass. The earlier `1b / family_self` answer-ownership pocket does **not** carry over to a harder self/other-boundary transfer probe. On the sharpened metric, the same model/frame combination is not merely weak; it is substantially below chance so far, even while self-prediction advantage stays positive. The cleanest reading is: the model can retain some coarse anticipatory sense of its own answer direction without maintaining a stable self/other boundary.

Why this matters:

- it lowers the ceiling on any strong identity interpretation
- it sharpens the best current synthesis into `coarse self-style anticipation without robust self/other boundary coherence`
- it makes the local `1b / family_self` self-recognition pocket look more task-specific and probe-fragile than previously hoped

## Paper-Level Takeaway

The measurement fixes strengthen the paper by sharpening the claim:

The broad identity claim should stay weak. The specific `1b / family_self` answer-ownership pocket is not merely a one-shuffle artifact and remains the strongest full-grid cell, but it is local, axis-specific, non-monotonic across model size, not multiplicity-robust under clustered full-grid inference, and still affected by response-label preference.

Best manuscript phrasing:

> A full label-balanced follow-up preserved a local `1b / family_self` answer-ownership pocket, but the overall self-recognition grid was below chance and the local cell did not generalize across model size or survive full-grid multiplicity correction.

Best limitation phrasing:

> Forced-choice identity probes are unusually sensitive to answer-label priors; even after counterbalancing, residual label preference and axis heterogeneity require treating self-recognition as a local phenomenon rather than a general model property.
