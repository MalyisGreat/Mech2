# Subagent Coordination Plan

## Coordination Protocol
1. Break each workstream into bounded tasks with explicit outputs.
2. Run literature collection first, then synthesis, then integrated design.
3. Track dependencies and unblock downstream lanes quickly.
4. Escalate only if a dependency is missing or contradictory evidence blocks synthesis.

## Subagent Lanes and Task Breakdowns

## `SA-1` Concept/Definitions
1. Define identity, drift, recovery, rebound in operational terms.
2. Formalize equations and measurement points.
3. List alternative explanations and falsification tests.
4. Emit conceptual framing memo.
Status: completed

## `SA-2` Steering Literature
1. Collect activation engineering and ITI papers.
2. Collect refusal-direction literature and counterpoints.
3. Collect context-aware steering and identifiability papers.
4. Extract method implications and confounds.
Status: completed

## `SA-3` Scale/Stability Literature
1. Collect scale law baselines.
2. Collect controlled model-suite resources.
3. Collect self-repair and stability evidence.
4. Map evidence to active-vs-passive hypotheses.
Status: completed

## `SA-4` Fine-Tuning Comparison
1. Collect LoRA/QLoRA/DoRA primary papers.
2. Collect direct LoRA-vs-FT forgetting/adaptation evidence.
3. Define adaptation-sensitive identity metrics.
Status: completed

## `SA-5` Experimental Design
1. Propose model families and comparison regimes.
2. Define intervention grid and controls.
3. Define logging schema and artifact layout.
Status: completed

## `SA-6` Metrics and Statistics
1. Specify primary metrics and normalizations.
2. Specify mixed-effects testing strategy.
3. Add causal plausibility checks and failure criteria.
Status: completed

## `SA-7` Risks/Reproducibility
1. Enumerate validity threats.
2. Define replication and determinism controls.
3. Set decision thresholds for strong claims.
Status: completed

## `SA-8` Integration
1. Consolidate all lanes into a final thesis-oriented brief.
2. Ensure each claim maps to source-backed evidence.
3. Produce actionable next execution steps.
Status: in_progress

## Dependency Notes
1. `SA-5` depends on `SA-2`, `SA-3`, `SA-4`.
2. `SA-6` depends on `SA-1`, `SA-5`.
3. `SA-8` depends on all prior lanes.

