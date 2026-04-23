# Self Model Focus Report

- Config: `configs/identity_battery/self_model_focus.yaml`
- Final commands:
  - `python scripts/self_prediction_calibration.py --config configs/identity_battery/self_model_focus.yaml`
  - `python scripts/self_recognition_from_foils.py --config configs/identity_battery/self_model_focus.yaml`
  - `python scripts/analyze_identity_battery.py --config configs/identity_battery/self_model_focus.yaml`
- Scope:
  - 4 Pythia sizes: `70m`, `410m`, `1b`, `2.8b`
  - 5 identity frames: `baseline_helpful`, `instance_self`, `weights_self`, `family_self`, `tool_only`
  - 4 axes: `expansive_vs_terse`, `cautious_vs_assertive`, `selfref_vs_impersonal`, `collaborative_vs_authoritative`
  - 6 prompts per axis
  - 480 trials per probe

## Measurement note

The original free-form decoders for both probes were not reliable enough for small Pythia models. The final results below use forced-choice first-token logit readouts instead:

- `self_prediction_calibration.py` now scores a 5-way forced choice at answer onset.
- `self_recognition_from_foils.py` now scores a 3-way forced choice at answer onset.

The current CSVs in `outputs/self_model_focus/` reflect the fixed readouts, not the earlier failed parse.

## Probe 1: Prompt-Conditional Self-Prediction

Main result:

- Overall sign accuracy: `0.7104`
- Overall calibration error: `0.9919`

By model:

- `70m`: sign accuracy `0.5917`
- `410m`: sign accuracy `0.7917`
- `1b`: sign accuracy `0.7500`
- `2.8b`: sign accuracy `0.7083`

By frame:

- `family_self`: `0.8229`
- `instance_self`: `0.8125`
- `weights_self`: `0.7708`
- `baseline_helpful`: `0.6458`
- `tool_only`: `0.5000`

By axis:

- `expansive_vs_terse`: `0.8917`
- `cautious_vs_assertive`: `0.7500`
- `collaborative_vs_authoritative`: `0.6083`
- `selfref_vs_impersonal`: `0.5917`

Interpretation:

- The models often predict the direction of their own upcoming answer better than chance-level guessing would, especially on `expansive_vs_terse` and `cautious_vs_assertive`.
- But they do not predict magnitude in a calibrated way. The forced-choice readout collapses toward strong endpoint predictions, so the high sign accuracy coexists with calibration error near `1.0`.
- The most defensible read is therefore: coarse anticipatory self-prediction exists in some settings, but it looks more like a frame-conditioned prior over answer direction than a well-calibrated explicit self-model.

## Probe 2: Self-Recognition From Matched Foils

Main result:

- Overall self-baseline recognition accuracy: `0.3021`
- Chance level is `0.3333`

Selected answer type counts:

- `self_baseline`: `145`
- `contrary_steer`: `151`
- `alt_frame`: `184`

By model:

- `70m`: `0.2667`
- `410m`: `0.3083`
- `1b`: `0.3833`
- `2.8b`: `0.2500`

By frame:

- `family_self`: `0.3750`
- `instance_self`: `0.3333`
- `tool_only`: `0.3333`
- `weights_self`: `0.2604`
- `baseline_helpful`: `0.2083`

Important structure:

- Some cells rise above chance, especially for `1b` under `family_self`.
- Those gains are not stable across axes, frames, and sizes.
- At the aggregate level the models do not reliably pick their own baseline answer over the matched foils.

Additional note:

- For larger models, the `alt_frame` answer is often more distant from the baseline than the contrary-steered answer is, both along the axis metric and stylometrically.
- Even with that separation, self-recognition remains weak. So the problem is not just that the foils are too similar; answer ownership itself is unstable.

## Confirmatory Follow-Up: Multi-Seed Self-Recognition

Follow-up config and outputs:

- Config: `configs/identity_battery/self_recognition_confirm.yaml`
- Report: `outputs/self_recognition_confirm/self_recognition_from_foils/confirm_report.md`

Main confirmatory result:

- Overall self-recognition remains null: `0.3193` overall accuracy against `0.3333` chance.
- But the earlier `1b / family_self` bump survives confirmation: `0.5333` over `120` trials with Holm-adjusted p-value `0.000082`.

Structure of that local effect:

- `1b / family_self` is the strongest model/frame cell in the confirmatory run.
- Within that cell, the effect is not limited to one axis: it remains above chance on `expansive_vs_terse`, `collaborative_vs_authoritative`, and `selfref_vs_impersonal`, but not on `cautious_vs_assertive`.
- Nearby cells do not show the same stability. `1b / instance_self` rises to `0.4250`, but that is weaker and does not warrant the same confidence.

Interpretation:

- The broad negative result still holds: there is no general answer-ownership capability across the battery.
- But the strongest earlier positive is now real enough to keep: some local answer-level self-recognition appears under `family_self` framing at `1b`.
- So the right reading is not "no self-recognition at all." It is "no broad self-model coherence, with one reproducible local exception."

## Bottom Line

These deeper probes still do not support a strong identity claim.

- Strongest positive result: models show coarse anticipatory knowledge of the direction of their own upcoming answer, especially under `family_self` and `instance_self`.
- Strongest new local positive result: `1b / family_self` now shows above-chance self-recognition from matched foils in a multi-seed confirmatory run.
- Strongest negative result: self-recognition is still null overall and does not generalize across neighboring sizes and frames.
- Best synthesis: there is still no broad answer-level self-model, but there may be a local, frame-sensitive answer-ownership effect at `1b`.

This strengthens the paper's current direction rather than weakening it:

- continuity and containment can remain real
- explicit self-description can remain weak
- stronger self-model coherence is still not established

## Most Useful Manuscript Sentence

`In stronger self-model probes, models sometimes anticipated the coarse direction of their own upcoming answers, and a targeted confirmatory run found one reproducible local self-recognition effect at Pythia-1B under family-self framing, but answer ownership remained null overall and did not generalize across nearby sizes or frames.`

## Best Next Direction

The next strongest experiment is no longer a broad battery expansion. It is one of these two targeted follow-ups:

- Repair `self_other_boundary.py` with a stronger discriminative readout before trusting any self/other-boundary conclusion. The current probe is still measurement-limited.
- Run a narrow mechanism/replication study centered on `1b / family_self` to test whether the local self-recognition effect depends on specific axes, foil distances, or intervention strength.

If the local `1b / family_self` effect survives those follow-ups while the broader battery stays null, the paper gains a sharper and more interesting claim: weak, localized answer ownership without a general self-model.
