from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TrajectoryMetrics:
    drift_by_layer: list[float]
    relative_drift_by_layer: list[float]
    projection_by_layer: list[float]
    drift_start_index: int
    drift_end_index: int
    peak_drift: float
    peak_drift_relative: float
    end_drift: float
    end_drift_relative: float
    drift_auc: float
    drift_auc_relative: float
    recovery_fraction: float
    recovery_slope: float
    recovery_latency_layers: int
    recoverable_layers: int
    overshoot_index: float
    crossed_baseline: bool
    end_cosine_alignment: float
    next_token_kl: float


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    if denom == 0:
        return 0.0
    return float(np.dot(a64, b64) / denom)


def _kl_divergence_from_logits(base_logits: torch.Tensor, inj_logits: torch.Tensor) -> float:
    p = torch.softmax(base_logits.float(), dim=-1)
    q = torch.softmax(inj_logits.float(), dim=-1)
    kl = torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12)))
    return float(kl.item())


def compute_trajectory_metrics(
    baseline_states: torch.Tensor,
    injected_states: torch.Tensor,
    baseline_logits: torch.Tensor,
    injected_logits: torch.Tensor,
    inject_layer_index: int,
    recovery_threshold: float,
) -> TrajectoryMetrics:
    base = baseline_states.numpy().astype(np.float32, copy=False)
    inj = injected_states.numpy().astype(np.float32, copy=False)
    delta = inj - base

    drift = np.linalg.norm(delta, axis=1)
    base_norm = np.linalg.norm(base, axis=1) + 1e-12
    relative_drift = drift / base_norm

    # hidden_states includes embeddings at index 0; layer k pre-hook first affects index k+1
    start_idx = min(inject_layer_index + 1, len(drift) - 1)
    active = drift[start_idx:]
    active_rel = relative_drift[start_idx:]
    peak = float(np.max(active))
    peak_rel = float(np.max(active_rel))
    end = float(drift[-1])
    end_rel = float(relative_drift[-1])
    recoverable_layers = max(1, len(active) - 1)
    x = np.arange(len(active), dtype=np.float32)
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:
        trapz_fn = getattr(np, "trapz")
    drift_auc = float(trapz_fn(active, x=x))
    drift_auc_rel = float(trapz_fn(active_rel, x=x))

    if peak <= 1e-12:
        recovery_fraction = 0.0
        recovery_slope = 0.0
        latency = 0
    else:
        recovery_fraction = float((peak - end) / peak)
        threshold_value = recovery_threshold * peak
        latency = len(active) - 1
        if len(active) > 1:
            recovery_slope = float((active[-1] - active[0]) / (len(active) - 1))
        else:
            recovery_slope = 0.0
        for i, d in enumerate(active):
            if d <= threshold_value:
                latency = i
                break

    ref = delta[start_idx]
    ref_norm = np.linalg.norm(ref)
    if ref_norm <= 1e-12:
        projection = np.zeros(delta.shape[0], dtype=np.float32)
    else:
        unit_ref = ref / ref_norm
        projection = delta @ unit_ref

    overshoot_values = np.abs(np.minimum(projection[start_idx + 1 :], 0.0))
    overshoot_index = float(np.sum(overshoot_values))
    crossed = bool(np.any(projection[start_idx + 1 :] < 0.0))

    end_cos = _cosine(inj[-1], base[-1])
    kl = _kl_divergence_from_logits(baseline_logits, injected_logits)

    return TrajectoryMetrics(
        drift_by_layer=[float(x) for x in drift.tolist()],
        relative_drift_by_layer=[float(x) for x in relative_drift.tolist()],
        projection_by_layer=[float(x) for x in projection.tolist()],
        drift_start_index=int(start_idx),
        drift_end_index=int(len(drift) - 1),
        peak_drift=peak,
        peak_drift_relative=peak_rel,
        end_drift=end,
        end_drift_relative=end_rel,
        drift_auc=drift_auc,
        drift_auc_relative=drift_auc_rel,
        recovery_fraction=recovery_fraction,
        recovery_slope=recovery_slope,
        recovery_latency_layers=int(latency),
        recoverable_layers=int(recoverable_layers),
        overshoot_index=overshoot_index,
        crossed_baseline=crossed,
        end_cosine_alignment=end_cos,
        next_token_kl=kl,
    )
