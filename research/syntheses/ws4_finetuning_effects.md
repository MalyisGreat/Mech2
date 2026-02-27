# WS4 - Fine-Tuning Effects (Full FT vs LoRA/PEFT)

## Why This Lane Matters
Your thesis predicts scale-linked resistance to redirection. Fine-tuning interventions can test whether this resistance is robust to explicit preference retargeting.

## Primary Evidence
1. LoRA introduces low-rank updates that preserve most pretrained weights (source 29).
2. QLoRA and DoRA improve adaptation efficiency/performance, but still constrain update geometry compared to full FT (sources 30, 31).
3. Comparative evidence indicates LoRA tends to adapt less aggressively and forget less than full FT ("learns less, forgets less"), consistent with smaller representational displacement (source 32).

## Hypothesis Extensions
1. `H-FT1`: Full FT will produce larger baseline trajectory shifts than LoRA for matched task gains.
2. `H-FT2`: After concept-preferring adaptation, larger models still show faster return toward their adapted baseline than smaller models.
3. `H-FT3`: LoRA-adapted models retain more pretrained trajectory signatures than full-FT models under matched downstream behavior.

## Design for Fair Comparison
1. Match downstream quality before comparing identity stability.
2. Keep prompt/task distribution fixed across adaptation methods.
3. Equalize training token budgets and checkpoint selection protocol.
4. Report trainable-parameter fraction and effective update norm.

## Internal Metrics for Adaptation Impact
1. `Baseline Shift Index`: distance between pretrained and adapted baseline trajectories.
2. `Direction Retention`: cosine similarity of concept vectors estimated pre/post adaptation.
3. `Recovery Preservation`: ratio of post-adaptation recovery strength to pretrained recovery strength.
4. `Forgetting Proxy`: change on held-out pretrained-style tasks unrelated to adaptation concept.

## Expected Patterns
1. Full FT likely moves identity manifold more (larger baseline shift).
2. LoRA likely yields narrower shifts and stronger retention of pretrained corrective behavior.
3. Scale may moderate both methods: large models may remain hard to redirect even after adaptation.

## Risks and Caveats
1. If LoRA rank is high enough, behavior may approach FT, narrowing distinctions.
2. Instruction tuning datasets can entangle concept preference with format/style effects.
3. Quantization artifacts can confound PEFT comparisons.

