# Source Registry (Primary Sources)

## Steering, Intervention, and Residual Dynamics

1. Turner et al. (2023), *Steering Language Models With Activation Engineering*  
URL: https://arxiv.org/abs/2308.10248  
Use: Core method for activation addition and inference-time steering.

2. Li et al. (2023), *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*  
URL: https://arxiv.org/abs/2306.03341  
Use: Evidence that linear activation interventions can causally shift model behavior.

3. Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*  
URL: https://arxiv.org/abs/2406.11717  
Use: Strong evidence for a behavior-mediating direction in residual activations.

4. Wang et al. (2025), *Refusal Direction is Universal Across Safety-Aligned Languages*  
URL: https://arxiv.org/abs/2505.17306  
Use: Cross-lingual transferability of steering directions.

5. Joad et al. (2026), *There Is More to Refusal in Large Language Models than a Single Direction*  
URL: https://arxiv.org/abs/2602.02132  
Use: Counterpoint showing multi-directional structure and nuanced refusal geometry.

6. Li et al. (2026), *Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models*  
URL: https://arxiv.org/abs/2602.01654  
Use: Context-dependent steering beyond a single static vector.

7. Han et al. (2026), *Steer2Adapt: Dynamically Composing Steering Vectors Elicits Efficient Adaptation of LLMs*  
URL: https://arxiv.org/abs/2602.07276  
Use: Dynamic composition of steering vectors, relevant for multi-concept interventions.

8. Venkatesh and Kurapath (2026), *On the Identifiability of Steering Vectors in Large Language Models*  
URL: https://arxiv.org/abs/2602.06801  
Use: Limits of interpreting steering vectors as unique causal factors.

9. Mayne et al. (2024), *Can sparse autoencoders be used to decompose and interpret steering vectors?*  
URL: https://arxiv.org/abs/2411.08790  
Use: Important caveats for post-hoc interpretation of steering vectors.

## Linear Representations, Residual Stream, and Mechanistic Priors

10. Park et al. (2023), *The Linear Representation Hypothesis and the Geometry of Large Language Models*  
URL: https://arxiv.org/abs/2311.03658  
Use: Formal grounding of linear concept directions and steering geometry.

11. Nanda et al. (2023), *Emergent Linear Representations in World Models of Self-Supervised Sequence Models*  
URL: https://arxiv.org/abs/2309.00941  
Use: Empirical evidence for linear concept structure and controllability.

12. Lawson et al. (2024), *Residual Stream Analysis with Multi-Layer SAEs*  
URL: https://arxiv.org/abs/2409.04185  
Use: Layerwise residual analysis methodology and scale-linked cross-layer similarity signal.

13. Lampinen et al. (2026), *Linear representations in language models can change dramatically over a conversation*  
URL: https://arxiv.org/abs/2601.20834  
Use: Evidence that representational directions can drift with context.

14. Geva et al. (2020), *Transformer Feed-Forward Layers Are Key-Value Memories*  
URL: https://arxiv.org/abs/2012.14913  
Use: Mechanistic baseline for how semantic information is encoded and transformed.

15. Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*  
URL: https://transformer-circuits.pub/2021/framework/  
Use: Foundational framework for residual pathways and circuit-level analysis.

16. Elhage et al. (2022), *Toy Models of Superposition*  
URL: https://arxiv.org/abs/2209.10652  
Use: Theoretical basis for polysemanticity and vector superposition.

17. Cunningham et al. (2023), *Sparse Autoencoders Find Highly Interpretable Features in Language Models*  
URL: https://arxiv.org/abs/2309.08600  
Use: SAE methodology for decomposing internal activations into interpretable features.

18. Gao et al. (2024), *Scaling and evaluating sparse autoencoders*  
URL: https://arxiv.org/abs/2406.04093  
Use: Scaling behavior and evaluation criteria for SAE-based representation analysis.

19. Rajamanoharan et al. (2024), *Improving Dictionary Learning with Gated Sparse Autoencoders*  
URL: https://arxiv.org/abs/2404.16014  
Use: Improved reconstruction/sparsity tradeoffs for feature extraction pipelines.

## Scale, Stability, and Self-Repair

20. Kaplan et al. (2020), *Scaling Laws for Neural Language Models*  
URL: https://arxiv.org/abs/2001.08361  
Use: Baseline scaling law framing.

21. Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models*  
URL: https://arxiv.org/abs/2203.15556  
Use: Compute-optimal scaling and size-token tradeoff.

22. Biderman et al. (2023), *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*  
URL: https://arxiv.org/abs/2304.01373  
Use: Controlled model family for scale-comparative experiments.

23. van der Wal et al. (2025), *PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs*  
URL: https://arxiv.org/abs/2503.09543  
Use: Seed-level pretraining stability and variance characterization.

24. Rushing and Nanda (2024), *Explorations of Self-Repair in Language Models*  
URL: https://arxiv.org/abs/2402.15390  
Use: Evidence for partial and sometimes overcorrective recovery after ablations.

## Localization and Causal Intervention Tooling

25. Zhang and Nanda (2023), *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*  
URL: https://arxiv.org/abs/2309.16042  
Use: Methodological guardrails for activation patching studies.

26. Syed et al. (2023), *Attribution Patching Outperforms Automated Circuit Discovery*  
URL: https://arxiv.org/abs/2310.10348  
Use: Efficient causal attribution approximation for large-scale interventions.

27. Kramar et al. (2024), *AtP*: An efficient and scalable method for localizing LLM behaviour to components*  
URL: https://arxiv.org/abs/2403.00745  
Use: Scalable patching variant with false-negative analysis.

28. Bhaskar et al. (2024), *Finding Transformer Circuits with Edge Pruning*  
URL: https://arxiv.org/abs/2406.16778  
Use: Circuit extraction at higher scales with sparse faithful subgraphs.

## Fine-Tuning Intervention Effects

29. Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*  
URL: https://arxiv.org/abs/2106.09685  
Use: Core PEFT baseline.

30. Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*  
URL: https://arxiv.org/abs/2305.14314  
Use: Practical large-scale PEFT with strong memory efficiency.

31. Liu et al. (2024), *DoRA: Weight-Decomposed Low-Rank Adaptation*  
URL: https://arxiv.org/abs/2402.09353  
Use: PEFT variant that narrows FT-vs-LoRA performance gap.

32. Biderman et al. (2024), *LoRA Learns Less and Forgets Less*  
URL: https://arxiv.org/abs/2405.09673  
Use: Direct evidence on adaptation capacity vs forgetting tradeoff compared to full FT.

