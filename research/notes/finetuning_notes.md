# Fine-Tuning Notes

## Focus
How adaptation methods alter internal identity and controllability.

## Key Notes by Source ID
1. `S29 (LoRA)`:
- Low-rank adapters constrain update subspace and preserve base weights.
- Useful as a "light-touch adaptation" regime.

2. `S30 (QLoRA)`:
- Makes large-model adaptation practical under memory constraints.
- Important for feasibility if running larger checkpoints.

3. `S31 (DoRA)`:
- Extends LoRA with magnitude-direction decomposition.
- Relevant sensitivity variant if LoRA/FT differences are ambiguous.

4. `S32 (LoRA learns less/forgets less)`:
- Suggests LoRA is less invasive in representational change.
- Strongly relevant to "identity retention under adaptation" claim.

## Operational Outcomes
1. Match task performance before comparing identity metrics.
2. Measure baseline shift pre/post adaptation.
3. Evaluate concept-direction retention after adaptation.
4. Avoid interpreting FT-vs-LoRA results if adaptation quality is unmatched.

