# Manuscript Patch Notes

## Introduction

- Keep the manuscript anchored on containment / resistance rather than personhood or a true-self claim.
- Add the identity bridge battery as a test of whether framing, hidden commitments, and long-form return mediate continuity.
- Reframe the paper as a three-part argument:
  - larger Pythia models are less behaviorally perturbed
  - token position / trajectory phase is a major moderator
  - added identity probes currently provide weak or negative evidence for strong self-model coherence
- Use a thesis sentence closer to:
  - `Our strongest evidence supports disturbance containment rather than universal self-restoration: larger models often damp internally induced perturbations more effectively, while added identity probes provide limited evidence that this robustness is mediated by stable explicit self-modeling.`

## Identity Section

- Strengthen identity language only if long-form return, hidden-charter consistency, and OOD robustness converge.
- The current partial pilot pushes the manuscript the other way: hidden-charter consistency is limited even without steering and usually degrades under contrary steering, while self-report-to-behavior coupling stays near zero.
- Revise identity language downward and describe continuity as patterned but weakly tied to explicit self-modeling unless later pilot/full runs overturn this pattern.
- Present the bridge battery as an empirical constraint on identity-forward interpretations, not as a failed add-on.

## Methods

- Add the identity battery entrypoints, exact YAML assets, smoke-tier config, and reporting outputs under `outputs/latest/`.
- State explicitly that mean-difference, random orthogonal, and label-shuffled controls are analyzed separately.
- Describe the adaptive baseline as a lightweight prompt-conditional layer selector.
- Note that the current partial-pilot summary rows are descriptive rather than inferential because each summary cell currently has `n = 1`.
- Note that the current hidden-charter probe shows label collapse and answer-validity collapse under steering, so the first pilot version mixes continuity loss with measurement failure.

## Limitations And Conclusion

- Preserve the manuscript's rejection of a universal snap-back law.
- Note that smoke-tier outputs verify instrumentation and reporting but are not yet the final inferential basis for manuscript-strength claims.
- Note that the current pass includes only a partial pilot for hidden-charter consistency and self-report coupling, not the full boundary, long-form, OOD, and adaptive pilot battery.
- If later pilot-tier results remain mixed, conclude that continuity can survive pressure without robust self-report coherence.
- Do not lean on the current smoke-tier long-form or adaptive outputs as headline evidence.
- Treat the first hidden-charter pilot as suggestive negative evidence with clear probe-design limitations.

## Claim Triage

- Strengthen: containment / damping claims already supported by the baseline Pythia sweep.
- Strengthen cautiously: a negative bridge result in which explicit self-description is weak and hidden commitments are fragile under steering.
- Weaken: any claim that recovery alone demonstrates concept-specific identity enforcement.
- Leave unchanged: the directional-only status of the GPT-2 / Qwen screen unless larger replication is run.
- Add as next-step priority: a small confirmatory battery with fewer sizes/frames/conditions but many more repeated prompts and a hidden-charter v2 probe that avoids label collapse and enforces YES/NO parsing.
