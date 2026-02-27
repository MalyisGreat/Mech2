# WS3 - Scaling and Internal Stability Signals

## Scale Framing
Classical scaling laws describe loss/performance behavior with model and data scale (sources 20, 21), but they do not directly answer whether internal trajectories become more resistant to intervention. This project extends scaling analysis into internal dynamical stability.

## Relevant Empirical Signals
1. Controlled model suites (Pythia) make size-comparative internal analysis feasible with minimized data/architecture confounds (source 22).
2. Multi-run evidence suggests meaningful stability structure and outlier behavior even under fixed settings, implying seed variance must be modeled explicitly (source 23).
3. Self-repair studies show networks can route around local damage and partially recover function, supporting plausibility of active corrective dynamics (source 24).
4. Residual-stream SAE analyses indicate cross-layer regularities that can strengthen with scale, relevant to trajectory coherence hypotheses (source 12).

## Distinguishing Passive Dilution vs Active Stabilization
### Passive Dilution Predictions
1. Drift decays monotonically with depth.
2. Recovery slope changes smoothly with injection strength.
3. No sign inversion in projection residuals.
4. Larger models show lower variance but no qualitative phase shift.

### Active Stabilization Predictions
1. Non-monotonic return dynamics (possible rebound/overshoot).
2. Thresholded response where stronger injections trigger stronger corrective return.
3. Layer-localized restoration zones (specific depth ranges with high corrective gain).
4. Correction signatures persisting after normalization against layer scale.

## Proposed Stability Diagnostics
1. `Return Gain`: local derivative of recovery wrt depth, `d(recover_l)/dl`.
2. `Overshoot Index`: signed area where projection crosses baseline and reverses.
3. `Stabilization Latency`: layers/tokens until drift falls below fixed fraction of post-injection peak.
4. `Robustness Exponent`: fit drift attenuation vs model size under matched intervention.

## Scale-Comparative Experimental Advice
1. Use a single family with consistent tokenizer/architecture style first (Pythia).
2. Run at least three random seeds for vector estimation and prompt subsampling.
3. Evaluate both relative and absolute depth coordinates (to avoid depth-count confounds).
4. Include matched-compute or matched-token-size controls where feasible.

## Interpretation Boundaries
1. Stronger recovery in larger models does not by itself prove intentional "selfhood"; it only supports stronger dynamical return toward pretrained trajectories.
2. If rebound appears only under select prompts/layers, the claim should be narrowed to regime-conditional stabilization.
3. If no rebound appears but return is still faster with scale, passive dilution remains a viable explanation.

