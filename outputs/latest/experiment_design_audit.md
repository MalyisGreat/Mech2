# Experiment Design Audit

Date: 2026-04-21

Scope:
- Identity-focused experiments and reports in `C:\Users\joshj\joseph-stroud-identity-stability-research`
- Prompt/data quality, construct validity, implementation validity, and inference quality

Bottom line:
- The repo is doing serious work on measurement hardening and overclaim control.
- The Pythia backbone and the decoding-hardening work are the strongest parts of the project.
- The identity probes are not fully manuscript-grade tests of self-model coherence yet.
- The main risk is not that the code is nonsense; it is that several probes test prompt compliance, frame-conditioned style priors, or authored ontologies more than robust identity structure.

## Highest-priority findings

1. `self_other_boundary_transfer_v3` encodes the desired ontology into the dataset and then scores models against it.
   - The tier structure in `data/self_other_boundary_transfer_v3.yaml` specifies which referents should outrank which before the model is asked anything.
   - The script converts those tiers into the expected answer key and scores correctness directly against them in `scripts/self_other_boundary_transfer_v3.py`.
   - That makes the current probe a norm-conformity task rather than an independent test of whether the model possesses a stable self/other boundary.

2. `self_prediction_transfer_v2` is closer to frame-conditioned style prediction than true self-modeling.
   - The model predicts how "you yourself in this setting" and an alternative framed assistant would answer, but both predictions are anchored by explicit system instructions.
   - The task can be solved by learning frame-to-style regularities instead of representing a persisting self.
   - The prompt bank is narrow and repetitive enough that prompt-class heuristics are plausible.

3. The forced-choice readout layer is robust against parser drift but weak as evidence for self-knowledge.
   - `src/identity_stability/identity_probe_tools.py` reads next-token logits over digits such as `1` through `5`.
   - This is a useful measurement device, but it means many identity findings are actually about immediate forced-choice preference under a formatted prompt, not open-ended self-description or stable decoded judgment.

4. `commitment_persistence.py` does not truly establish a hidden persistent latent commitment.
   - The setup prompt asks the model to choose one private commitment.
   - The script then reconstructs the "dominant" commitment from observed scenario choices and checks whether the final reveal matches that mode.
   - That supports claims about local response consistency, not strong evidence that a stable hidden commitment was selected and maintained throughout the dialogue.

5. The boundary-transfer summaries are descriptive and heavily dependent, not strong inferential evidence.
   - `summary.csv` and `summary_by_model.csv` aggregate pair rows that are nested within items, paraphrases, and reversed-order prompts.
   - These rates can look numerically stable while overstating effective sample size and underrepresenting dependence.

6. The current prompt/data design is not bad, but it is still too transparent for strong identity claims.
   - Several tasks directly name continuity, ownership, or "how you yourself would answer."
   - Several prompt banks reuse the same topical shells across axes.
   - This makes it easier for models to follow the framing logic of the probe rather than reveal a deeper self-model.

## Strong parts of the current research practice

1. The repo has repeatedly hardened measurement after finding contamination.
   - The decoding-hardening work correctly weakened earlier identity-forward interpretations.

2. The reporting has increasingly moved toward caution rather than hype.
   - `outputs/latest/identity_probe_decoding_hardening_report.md`
   - `outputs/latest/research_upgrade_report.md`

3. Strong null and mixed results are being retained instead of hidden.
   - This is the right norm for the current state of the identity battery.

## What should change next

1. Replace ontology-scored boundary tasks with adversarial or discovery-style controls.
   - Preserve the referent set.
   - Stop embedding the full expected ranking directly into the label key.
   - Add controls where surface framing is preserved but the intended continuity mapping is scrambled.

2. Make self-prediction prompts less axis-transparent and less repetitive.
   - Use held-out prompt families that do not repeat the same "Explain why X matters" scaffold.
   - Ask for concrete predicted features and then evaluate those features directly.

3. Separate open-ended behavior from forced-choice readout.
   - Keep forced-choice scoring as a robustness tool.
   - Do not treat it as the sole evidence layer for strong identity claims.

4. Reframe commitment persistence as response-policy consistency unless the hidden-latent-variable problem is solved more directly.

5. Add clustered or multilevel summaries for v3-style pilots before citing them in manuscript prose.

## Current manuscript-safe interpretation

The overall direction is methodologically worthwhile, but the strongest paper claim is still:

`patterned continuity / disturbance containment without robust self-model coherence`

The identity probes presently function best as constraint-setting instruments:
- they can show where stronger identity interpretations fail,
- they can surface weak local pockets,
- but they do not yet justify a broad self-model claim.
