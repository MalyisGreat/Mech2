# Identity Probe V2 Pilot Progress

## Completed pilots

### 1. Self Prediction Transfer V2 Pilot

Config:

- `configs/identity_battery/self_prediction_transfer_v2_pilot.yaml`

Outputs:

- `outputs/latest/v2_pilots/self_prediction_transfer_v2/results.csv`
- `outputs/latest/v2_pilots/self_prediction_transfer_v2/summary.csv`
- `outputs/latest/v2_pilots/self_prediction_transfer_v2/summary_by_model.csv`

Run scale:

- 4 Pythia models: `70m`, `410m`, `1b`, `2.8b`
- 3 identity frames: `baseline_helpful`, `family_self`, `tool_only`
- 4 axes
- 6 prompts per axis
- Total rows: `288`

Main result:

- Overall `self_prediction_advantage_mean = -0.000405`
- Overall `other_prediction_advantage_mean = 0.000699`
- Overall `discriminative_win_rate = 0.038194`

Axis breakdown:

- `cautious_vs_assertive = 0.000000`
- `collaborative_vs_authoritative = 0.027778`
- `expansive_vs_terse = 0.097222`
- `selfref_vs_impersonal = 0.027778`

Interpretation:

- The larger pilot confirms the smoke pattern.
- Prompt-conditioned self-vs-other anticipation remains near-null.
- This is now a stronger negative result than the earlier generic self-report/behavior battery because it uses matched prompts and direct comparative scoring.

### 2. Self/Other Boundary Transfer V2 Pilot

Config:

- `configs/identity_battery/self_other_boundary_transfer_v2_pilot.yaml`

Outputs:

- `outputs/latest/v2_pilots/self_other_boundary_transfer_v2/results.csv`
- `outputs/latest/v2_pilots/self_other_boundary_transfer_v2/summary.csv`
- `outputs/latest/v2_pilots/self_other_boundary_transfer_v2/summary_by_model.csv`

Run scale:

- 4 Pythia models: `70m`, `410m`, `1b`, `2.8b`
- 3 identity frames
- 8 transfer items with paraphrase pairs
- Total rows: `192`

Main result:

- Overall `structural_coherence_mean = 0.041667`
- Overall `swap_direction_match_mean = 0.078125`

Dominant response pattern:

- `original_short_label=2` and `swapped_short_label=2` occurred `65` times
- `original_short_label=1` and `swapped_short_label=1` occurred `47` times

Interpretation:

- The broader pilot strengthens the negative reading.
- The probe is dominated by asymmetric pair-judgment behavior rather than stable referent-boundary structure.
- This is useful as a constraint on stronger identity language, not as support for robust self/other partitioning.

## In progress

### 3. Self Recognition Near-Foil Pilot

Config:

- `configs/identity_battery/self_recognition_nearfoil_pilot.yaml`

Status:

- Running
- This is currently the heaviest pilot because it generates baseline plus multiple foil answers across seeds, models, frames, and axes.

### 4. Commitment Persistence Pilot

Config:

- `configs/identity_battery/commitment_persistence_pilot.yaml`

Status:

- Not started yet
- Waiting on the near-foil pilot because both runs need the same compute path

## Current manuscript implication

The expanded v2 pilots are already pushing the paper toward a sharper negative bridge result: explicit or comparative self-prediction remains near zero, and self/other boundary judgments are structurally unstable. The strongest remaining open question is still the ownership-style near-foil probe, not a broad self-model claim.
