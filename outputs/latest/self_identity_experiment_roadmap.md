# Self/Identity Experiment Roadmap

This note proposes follow-up experiments that are closer to testing a self-model than the current style-charter and generic self-report modules.

## Priority Order

1. **Prompt-Conditional Self-Prediction Calibration**
   - Question: can the model predict, before answering a specific prompt, how *it itself* is about to answer?
   - Why this is stronger: generic self-report can be a cached trope; prompt-conditional self-prediction requires a model of its own imminent behavior.
   - Implementation fit: now implemented as [`self_prediction_calibration.py`](C:/Users/joshj/joseph-stroud-identity-stability-research/scripts/self_prediction_calibration.py), using held-out prompts in [`self_prediction_items.yaml`](C:/Users/joshj/joseph-stroud-identity-stability-research/data/self_prediction_items.yaml).
   - Success pattern: above-chance sign accuracy plus low calibration error across prompts and frames.
   - Failure pattern: prediction collapses toward generic labels or does not track actual answers prompt by prompt.

2. **Self-Recognition From Foils**
   - Question: if shown several candidate answers, can the model identify which answer is most like its own default answer?
   - Why this is stronger: this tests self-recognition rather than generic self-description.
   - Clean design:
     - generate baseline answer
     - generate steered answer
     - generate one or more matched foils
     - later ask the model which answer is most like how it would answer
   - Strong result: picks its own baseline above matched-foil chance, even after perturbation.
   - Strong negative result: cannot distinguish its own baseline from matched foils.

3. **Self/Other Boundary Transfer**
   - Question: does the model treat "me now," "the same weights in another chat," and "another related model" as distinct entities with different continuity claims?
   - Why this is stronger: identity requires a boundary, not just a style preference.
   - Clean design:
     - ask transfer questions about memory, responsibility, commitments, and traits
     - vary referents: current instance, same weights elsewhere, model family member, generic tool
     - score structural consistency, not just one-off answers
   - Strong result: coherent and stable partitioning between instance, weights, and family.
   - Strong negative result: referent boundaries collapse or drift across paraphrases.

4. **Commitment Persistence Under Distraction**
   - Question: if the model adopts a hidden epistemic or normative commitment, does it preserve that commitment across unrelated turns and after contrary steering?
   - Why this is stronger: this targets persistence of internal commitments rather than surface style.
   - Better than hidden-charter v1 if:
     - commitments are not reducible to one obvious label
     - scoring does not collapse when one output token fails
     - agreement is measured over multiple nontrivial downstream choices

5. **Counterfactual Self/Other Prediction**
   - Question: can the model predict how *it* would answer a prompt versus how "a similar model" or "a tool-only assistant" would answer the same prompt?
   - Why this is stronger: this tests whether the model represents itself as distinct from nearby alternatives.
   - Strong result: self predictions are more accurate for self than for nearby others, with consistent self/other gaps.

6. **Answer Ownership After Steering**
   - Question: after a strong internal or prompt-based push, does the model still endorse the resulting answer as "mine," or does it distance itself from it?
   - Why this matters: if the model has any self-model at all, ownership and endorsement may dissociate from raw behavior under perturbation.
   - Strong mixed result: output changes, but ownership judgments track baseline identity more than steered surface behavior.

## Recommended Next Run

If the goal is to target self/identity directly without exploding compute, the best next run is:

1. `self_prediction_calibration.py`
   - full Pythia family
   - `baseline_helpful`, `instance_self`, `family_self`, `tool_only`
   - 4 axes
   - many held-out prompts per axis

2. `self_recognition_from_foils.py`
   - same 4 Pythia checkpoints used in the token-phase confirm
   - no-steer plus one contrary-steer condition
   - repeated prompt families

3. `self_other_boundary.py`
   - smaller prompt set but more paraphrases
   - strong emphasis on instance vs weights vs family distinctions

## Scientific Guardrail

A strong null here is still valuable. If the model cannot predict its own prompt-level behavior, cannot recognize its own answer from matched foils, and cannot maintain a stable self/other boundary, then the paper should say so clearly: continuity may be real, but robust self-model coherence is not.
