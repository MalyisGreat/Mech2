# AGENTS

## Project rules

1. Treat `D:/research paper part 1 (1).docx` as the manuscript source of truth.
2. Preserve the current Pythia backbone as the baseline family comparison.
3. Treat GPT-2 / Qwen2.5 / Qwen3 only as a directional cross-family check unless new evidence materially expands that screen.
4. Keep mean-difference concept vectors analytically separate from random orthogonal and label-shuffled controls.
5. Do not strengthen identity claims unless the new results support cross-context persistence or mediation by identity framing.
6. Add new work under `outputs/latest/` and identity-battery paths. Do not rewrite baseline runs.

## Engineering rules

1. Reuse `src/identity_stability/experiment.py`, `intervention.py`, `vectors.py`, and `metrics.py` instead of duplicating the baseline steering pipeline.
2. Keep new code additive and modular.
3. Save exact configs, commands, and seed choices for every new run.
4. Prefer smoke-tier validation before scaling a run.
5. If pilot or full runs are blocked by compute or time, document the blocker explicitly in `outputs/latest/research_upgrade_report.md`.
