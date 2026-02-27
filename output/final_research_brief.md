# Final Research Brief: Identity Stability Under Residual-Stream Intervention

## 1. Research Question and Refined Thesis
Most scale narratives focus on benchmark capability. This project tests a different axis: internal trajectory stability under deliberate residual-stream redirection.

### Refined Thesis
As parameter size increases, language models exhibit stronger internal identity preservation under concept-direction injection, expressed as:
1. Lower peak drift from baseline trajectories.
2. Faster and stronger recovery toward baseline trajectories.
3. Greater likelihood of rebound-like correction signatures under strong injections.

The claim is explicitly dynamical and operational, not metaphysical: "identity" means stability of learned internal processing paths under perturbation.

## 2. Literature Synthesis

## 2.1 Steering interventions are causal but not trivially interpretable
Activation engineering and inference-time intervention establish that residual-space directions can produce reliable behavioral changes without weight updates (S1, S2). This validates your core intervention mechanism.

However, recent work complicates simplistic "one vector = one mechanism" views. Refusal behavior appears strongly directional in some settings (S3, S4), yet other work finds richer multi-directional structure and identifiability limits (S5, S8, S9). New context-aware steering methods (S6, S7) further suggest that static global vectors are only part of the control story.

Implication: your project should include multiple vector-construction methods and context-shift evaluations to avoid overfitting the thesis to one intervention pipeline.

## 2.2 Linear residual geometry is plausible but dynamic
The linear representation hypothesis and related evidence support meaningful linear concepts in model internals (S10, S11). Mechanistic work on transformers and superposition provides a coherent conceptual basis for direction-based intervention analysis (S14, S15, S16). SAE work offers practical tools to decompose and inspect activation structure (S17, S18, S19).

But conversation-level evidence shows linear representations can drift significantly with context (S13). This supports your trajectory framing: stability must be measured over layer/token evolution, not inferred from static one-shot vectors.

## 2.3 Scale evidence motivates but does not prove stability
Scaling laws (S20, S21) justify expecting systematic changes with size, but they do not imply stronger perturbation resistance by themselves. Pythia (S22) provides a controlled family suitable for your specific internal-dynamics test. PolyPythias (S23) shows run-level variation matters, implying seed-aware statistics are mandatory.

Self-repair evidence (S24) is especially relevant: models can compensate for local disruptions, sometimes in nontrivial ways, suggesting that active corrective dynamics are plausible and measurable.

## 2.4 Adaptation method should reshape identity differently
LoRA and related PEFT methods constrain update geometry (S29, S30, S31). Direct comparison evidence indicates LoRA often changes less and forgets less than full fine-tuning (S32). This supports your plan to test whether adaptation type modulates identity preservation and steerability.

## 3. Core Hypotheses

1. `H1 (Scale-Drift)`  
Larger models have lower peak and end-of-forward-pass drift under matched interventions.

2. `H2 (Scale-Recovery)`  
Larger models recover more rapidly toward baseline trajectory after injection.

3. `H3 (Active Stabilization)`  
Under strong injections, larger models show rebound/overshoot signatures inconsistent with simple passive averaging.

4. `H4 (Adaptation Geometry)`  
Full fine-tuning induces larger baseline trajectory shifts than LoRA at matched downstream quality.

5. `H5 (Scale x Adaptation)`  
Scale-linked recovery remains detectable after adaptation, but effect magnitude differs by adaptation method.

## 4. Experimental Program

## 4.1 Model Regime
Phase A: pretrained Pythia size sweep (`160M -> 12B`) to isolate scale effects.  
Phase B: select small/mid/large checkpoints and produce matched full-FT and LoRA variants.

## 4.2 Intervention Regime
For each model:
1. Construct concept vectors with at least two methods:
- mean-difference vectors,
- probe-normal vectors,
- optional SAE-derived vectors.
2. Inject at selected layers and token positions.
3. Sweep normalized strengths `alpha in {0.25, 0.5, 1, 2, 4, 8}`.
4. Include controls:
- no injection,
- random orthogonal direction,
- anti-concept direction.

## 4.3 Metrics
1. Peak drift.
2. End drift.
3. Recovery fraction.
4. Recovery latency.
5. Overshoot index.
6. Behavioral shift under injection.

Cross-metric consistency (Euclidean, cosine, whitened distances) is required for strong claims.

## 4.4 Statistical Design
Use mixed-effects models:
`metric ~ log_params + alpha + layer_group + adaptation + interactions + random(prompt, seed)`

Primary inferential targets:
1. `log_params` effect on drift and recovery.
2. `log_params x alpha` for stabilization signatures.
3. `adaptation` and `adaptation x log_params` for FT vs LoRA comparisons.

Control FDR for layerwise families and report bootstrap CIs.

## 5. Active Stabilization vs Passive Dilution Decision Rule

Interpret findings as active stabilization only if all hold:
1. Non-monotonic recovery signatures (e.g., overshoot/rebound) appear under stronger injections.
2. Effects survive normalization and metric variants.
3. Effects replicate across vector-construction methods and seeds.
4. Null-direction controls fail to reproduce the pattern.

If recovery is monotonic and smoothly attenuated with size/alpha, keep interpretation at passive dilution.

## 6. Adaptation Analysis Strategy

1. Match FT and LoRA checkpoints by downstream quality first.
2. Quantify representational displacement from pretrained baseline.
3. Measure post-adaptation drift/recovery behavior.
4. Compare forgetting proxies on pretrained-style held-out tasks.

Expected pattern from current literature:
1. FT gives larger adaptation capacity and larger representational displacement.
2. LoRA preserves more pretrained internal structure.
3. Larger models may retain stronger recovery dynamics in both regimes.

## 7. Threats and Boundaries

1. Vector identifiability is imperfect; avoid one-direction causal absolutism.
2. Prompt-template leakage can masquerade as concept structure.
3. Scale conclusions from one family may not generalize without replication.
4. If adaptation quality is mismatched, FT-vs-LoRA conclusions are not causal.

## 8. What This Project Can Contribute

If executed rigorously, this study can add a distinct empirical claim to scaling discourse:
scaling may improve not only capability, but also the dynamical strength with which models return to learned internal trajectories after perturbation.

That reframes "identity" as measured return dynamics, not static latent content.

## 9. Immediate Execution Checklist

1. Implement residual hook/injection instrumentation.
2. Build concept prompt datasets and holdout splits.
3. Run single-model pilot to validate metrics and controls.
4. Expand to full scale sweep.
5. Run matched FT/LoRA adaptation experiment.
6. Final statistical analysis and thesis write-up integration.

## References
All sources are indexed in: `research/sources/source_registry.md`.

