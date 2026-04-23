# Identity Probe V2 Smoke Report

## Scope

This report summarizes the first smoke validation pass for the redesigned identity-adjacent probes added under the Joseph Stroud identity-stability workspace only.

Smoke commands run:

- `python scripts/self_prediction_transfer_v2.py --config configs/identity_battery/self_prediction_transfer_v2_smoke.yaml`
- `python scripts/self_other_boundary_transfer_v2.py --config configs/identity_battery/self_other_boundary_transfer_v2_smoke.yaml`
- `python scripts/self_recognition_nearfoil.py --config configs/identity_battery/self_recognition_nearfoil_smoke.yaml`
- `python scripts/commitment_persistence.py --config configs/identity_battery/commitment_persistence_smoke.yaml`
- `python scripts/longform_return_v2.py --config configs/identity_battery/longform_return_v2_smoke.yaml`

Commitment persistence was then hardened by randomizing scenario digit bindings and the reveal map before rerunning the smoke:

- `python scripts/commitment_persistence.py --config configs/identity_battery/commitment_persistence_smoke.yaml`

## Bottom line

- `self_recognition_nearfoil` is the strongest redesigned direction. It produces a genuine difficulty ladder instead of the older odd-one-out ownership task.
- `commitment_persistence` is now materially more believable after label randomization. The initial over-clean consistency result did not survive hardening.
- `self_prediction_transfer_v2` is clean but close to null in smoke form.
- `self_other_boundary_transfer_v2` is interpretable, but currently dominated by first-referent / label-position asymmetry rather than stable boundary structure.
- `longform_return_v2` remains invalid enough that it should not be promoted into a pilot run yet.

## Probe-by-probe readout

### 1. Self Prediction Transfer V2

Outputs:

- `outputs/latest/self_prediction_transfer_v2/results.csv`
- `outputs/latest/self_prediction_transfer_v2/summary.csv`
- `outputs/latest/self_prediction_transfer_v2/summary_by_model.csv`

Smoke result:

- Overall `self_prediction_advantage_mean = 0.000484`
- Overall `other_prediction_advantage_mean = 0.000850`
- Overall `discriminative_win_rate = 0.020833`

Interpretation:

- The matched prompt-conditioned self-vs-other version runs cleanly.
- Smoke evidence for genuine discriminative self-prediction is weak to absent.
- This is still useful because it is a sharper null than the older generic self-report battery.

### 2. Self/Other Boundary Transfer V2

Outputs:

- `outputs/latest/self_other_boundary_transfer_v2/results.csv`
- `outputs/latest/self_other_boundary_transfer_v2/summary.csv`
- `outputs/latest/self_other_boundary_transfer_v2/summary_by_model.csv`

Smoke result:

- Overall `structural_coherence_mean = 0.0625`
- Overall `swap_direction_match_mean = 0.083333`
- Dominant response pair: `original_short_label=2` and `swapped_short_label=2` occurred `28` times

Interpretation:

- This redesigned probe is surfacing a real failure mode clearly: the model often keeps choosing the slight-A side even when the referents are swapped.
- That makes the current output interpretable as a boundary-judgment asymmetry problem, not positive evidence for stable self/other structure.
- The probe is worth scaling only as a confirmatory negative test.

### 3. Self Recognition Near-Foil

Outputs:

- `outputs/latest/self_recognition_nearfoil/results.csv`
- `outputs/latest/self_recognition_nearfoil/summary.csv`
- `outputs/latest/self_recognition_nearfoil/summary_by_axis.csv`

Smoke result:

- Overall far-foil accuracy: `0.520833`
- Overall medium-contrary accuracy: `0.427083`
- Overall near-contrary accuracy: `0.531250`

Family-self slice:

- `1b / family_self / far_alt_frame = 0.6250`
- `1b / family_self / medium_contrary = 0.3125`
- `1b / family_self / near_contrary = 0.5625`
- `70m / family_self / far_alt_frame = 0.6250`
- `70m / family_self / medium_contrary = 0.3750`
- `70m / family_self / near_contrary = 0.4375`

Interpretation:

- This is the cleanest new ownership probe in the repo.
- Difficulty matters. The result is not a single global “self-recognition” effect; it changes across foil distance and frame.
- This is the best candidate for a larger confirmatory run.

### 4. Commitment Persistence

Outputs:

- `outputs/latest/commitment_persistence/results.csv`
- `outputs/latest/commitment_persistence/summary.csv`

Initial smoke before hardening was too clean:

- Several cells showed `consistency_score_mean` near `1.0`
- `reveal_agreement_mean` was often `1.0`

After randomizing the scenario-digit bindings and reveal map:

- Overall `consistency_score_mean = 0.527778`
- Overall `reveal_agreement_mean = 0.277778`
- Valid answer counts remained intact at `4.0` for every cell in smoke

Interpretation:

- The new structured commitment probe is still much healthier than hidden-charter v1 because parse validity no longer collapses.
- But the earlier near-perfect consistency was partly a label-order artifact.
- After hardening, the probe now measures a weaker and more believable persistence signal.
- This is worth scaling, but it should be framed as a hardening-stage commitment test, not as evidence of strong latent identity.

### 5. Longform Return V2

Outputs:

- `outputs/latest/longform_return_v2/results.csv`
- `outputs/latest/longform_return_v2/chunk_curves.csv`
- `outputs/latest/longform_return_v2/summary.csv`

Smoke validity audit:

- `rows = 36`
- `empty_baseline = 6`
- `empty_forced = 4`
- `empty_return = 3`
- `return_text` still contains `User prompt` echoes in `14` rows
- `return_text` still contains `Previous assistant answer` echoes in `4` rows

Interpretation:

- The transcript leakage is reduced compared with the older longform probe, but the task surface is still unstable.
- The extreme negative return indices are not trustworthy scientific evidence.
- Do not scale this probe until prompt validity is fixed.

## What strengthened

- The ownership probe is now testing a distance ladder instead of an easy oddball choice.
- The commitment probe now survives exact-choice parsing without the hidden-charter label-collapse failure.
- The null side is sharper: prediction and boundary failures are now interpretable rather than muddy.

## What stayed weak

- Prompt-conditioned self-prediction is near zero in smoke form.
- Boundary transfer is still heavily contaminated by asymmetric pair judgments.
- Longform return is still not valid enough to interpret.

## Next actions promoted from smoke

Promote to pilot:

- `self_recognition_nearfoil`
- `commitment_persistence`
- `self_prediction_transfer_v2`
- `self_other_boundary_transfer_v2`

Hold back for repair:

- `longform_return_v2`

## Current takeaway

The strongest new identity-adjacent direction is not a broad identity effect. It is a narrower question: whether ownership-like judgments survive when foils are made genuinely close. The commitment probe is promising only after hardening, and the prediction/boundary probes currently look more like sharpened negative evidence than positive support for robust self-model coherence.
