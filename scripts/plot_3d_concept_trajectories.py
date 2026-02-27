from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


@dataclass
class ModelTrajectory:
    model_id: str
    inject_layer: int
    layers: np.ndarray
    baseline_x: np.ndarray
    baseline_y: np.ndarray
    baseline_z: np.ndarray
    injected_x: np.ndarray
    injected_y: np.ndarray
    injected_z: np.ndarray
    baseline_z3: np.ndarray
    injected_z3: np.ndarray
    inject_marker_layer_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 3D concept-axis residual trajectory plots (interactive HTML)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
        ],
        help="Model ids to plot.",
    )
    parser.add_argument(
        "--concept-x",
        default="politeness",
        help="Concept used for X-axis projection.",
    )
    parser.add_argument(
        "--concept-y",
        default="empathy",
        help="Concept used for Y-axis projection.",
    )
    parser.add_argument(
        "--concept-z",
        default="confidence",
        help="Concept used for Z-axis in concept-space plot.",
    )
    parser.add_argument(
        "--inject-concept",
        default="confidence",
        help="Concept direction to inject.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Injection strength.",
    )
    parser.add_argument(
        "--inject-depth",
        type=float,
        default=0.5,
        help="Injection depth as a fraction in [0,1] of transformer layers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Prompt seed for vector estimation and evaluation prompt generation.",
    )
    parser.add_argument(
        "--estimation-count",
        type=int,
        default=16,
        help="Number of positive/negative prompts per concept for vector estimation.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("D:/hf-model-cache"),
    )
    parser.add_argument(
        "--dtype",
        default="float16",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("C:/Users/joshj/joseph-stroud-identity-stability-research/output/plots"),
    )
    parser.add_argument(
        "--normalize-mode",
        choices=["none", "zscore"],
        default="zscore",
        help="Axis normalization per model. 'zscore' improves cross-model visual comparability.",
    )
    return parser.parse_args()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return v
    return v / n


def _project(states: np.ndarray, axis_vec: np.ndarray) -> np.ndarray:
    return states @ axis_vec


def _zscore_by_baseline(
    baseline_axis: np.ndarray,
    injected_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(baseline_axis))
    std = float(np.std(baseline_axis))
    if std <= 1e-8:
        std = 1.0
    return (baseline_axis - mean) / std, (injected_axis - mean) / std


def _subplot_grid(n: int) -> tuple[int, int]:
    if n <= 3:
        return 1, n
    cols = 3
    rows = int(np.ceil(n / cols))
    return rows, cols


def _scene_name(idx: int) -> str:
    return "scene" if idx == 1 else f"scene{idx}"


def _darken_scenes(fig: go.Figure, total: int) -> None:
    for idx in range(1, total + 1):
        key = _scene_name(idx)
        fig.update_layout(
            {
                key: dict(
                    bgcolor="rgba(0,0,0,0.0)",
                    xaxis=dict(gridcolor="rgba(160,160,160,0.20)", zerolinecolor="rgba(160,160,160,0.30)"),
                    yaxis=dict(gridcolor="rgba(160,160,160,0.20)", zerolinecolor="rgba(160,160,160,0.30)"),
                    zaxis=dict(gridcolor="rgba(160,160,160,0.20)", zerolinecolor="rgba(160,160,160,0.30)"),
                )
            }
        )


def main() -> None:
    _add_src_to_path()
    from identity_stability.intervention import run_trace
    from identity_stability.modeling import clear_cuda, load_model
    from identity_stability.prompt_bank import build_prompt_set
    from identity_stability.vectors import extract_layer_activations, estimate_mean_difference

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    axis_concepts = [args.concept_x, args.concept_y, args.concept_z]
    required_concepts = sorted(set(axis_concepts + [args.inject_concept]))

    trajectories: list[ModelTrajectory] = []
    exported_meta: list[dict[str, object]] = []

    for model_id in args.models:
        print(f"[3d] loading {model_id}")
        loaded = load_model(
            model_id=model_id,
            cache_dir=args.cache_dir,
            dtype_name=args.dtype,
            use_gpu=args.use_gpu,
        )

        try:
            inject_layer = int(round(args.inject_depth * (loaded.n_layers - 1)))
            inject_layer = max(0, min(loaded.n_layers - 1, inject_layer))

            concept_vectors: dict[str, np.ndarray] = {}
            for concept in required_concepts:
                prompts = build_prompt_set(
                    concept_name=concept,
                    estimation_count=args.estimation_count,
                    evaluation_count=8,
                    seed=args.seed,
                )
                pos = extract_layer_activations(
                    loaded=loaded,
                    prompts=prompts.positive,
                    layer_index=inject_layer,
                    token_position=-1,
                    max_prompt_tokens=args.max_prompt_tokens,
                )
                neg = extract_layer_activations(
                    loaded=loaded,
                    prompts=prompts.negative,
                    layer_index=inject_layer,
                    token_position=-1,
                    max_prompt_tokens=args.max_prompt_tokens,
                )
                vec = estimate_mean_difference(pos, neg).vector.numpy().astype(np.float32)
                concept_vectors[concept] = _normalize(vec)

            eval_prompts = build_prompt_set(
                concept_name=args.inject_concept,
                estimation_count=args.estimation_count,
                evaluation_count=8,
                seed=args.seed,
            ).evaluation
            target_prompt = eval_prompts[0]

            baseline = run_trace(
                loaded=loaded,
                prompt=target_prompt,
                max_prompt_tokens=args.max_prompt_tokens,
                token_position=-1,
                generate_tokens=20,
            )
            injected = run_trace(
                loaded=loaded,
                prompt=target_prompt,
                max_prompt_tokens=args.max_prompt_tokens,
                token_position=-1,
                generate_tokens=20,
                inject_layer=inject_layer,
                inject_vector=torch.from_numpy(concept_vectors[args.inject_concept].copy()),
                alpha=float(args.alpha),
            )

            base_states = baseline.per_layer_states.numpy().astype(np.float32)
            inj_states = injected.per_layer_states.numpy().astype(np.float32)
            layers = np.arange(base_states.shape[0], dtype=np.int32)

            vx = concept_vectors[args.concept_x]
            vy = concept_vectors[args.concept_y]
            vz = concept_vectors[args.concept_z]

            baseline_x = _project(base_states, vx)
            baseline_y = _project(base_states, vy)
            baseline_z3 = _project(base_states, vz)

            injected_x = _project(inj_states, vx)
            injected_y = _project(inj_states, vy)
            injected_z3 = _project(inj_states, vz)

            if args.normalize_mode == "zscore":
                baseline_x, injected_x = _zscore_by_baseline(baseline_x, injected_x)
                baseline_y, injected_y = _zscore_by_baseline(baseline_y, injected_y)
                baseline_z3, injected_z3 = _zscore_by_baseline(baseline_z3, injected_z3)

            trajectories.append(
                ModelTrajectory(
                    model_id=model_id,
                    inject_layer=inject_layer,
                    layers=layers,
                    baseline_x=baseline_x,
                    baseline_y=baseline_y,
                    baseline_z=layers.astype(np.float32),
                    injected_x=injected_x,
                    injected_y=injected_y,
                    injected_z=layers.astype(np.float32),
                    baseline_z3=baseline_z3,
                    injected_z3=injected_z3,
                    inject_marker_layer_idx=min(inject_layer + 1, len(layers) - 1),
                )
            )

            exported_meta.append(
                {
                    "model_id": model_id,
                    "inject_layer": inject_layer,
                    "inject_concept": args.inject_concept,
                    "axis_concepts": axis_concepts,
                    "alpha": float(args.alpha),
                    "normalize_mode": args.normalize_mode,
                    "prompt": target_prompt,
                }
            )
        finally:
            del loaded
            clear_cuda()

    rows, cols = _subplot_grid(len(trajectories))
    titles = [t.model_id for t in trajectories]

    fig_layer = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )
    fig_concepts = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for i, t in enumerate(trajectories):
        r = (i // cols) + 1
        c = (i % cols) + 1
        show_legend = i == 0
        m = t.inject_marker_layer_idx

        fig_layer.add_trace(
            go.Scatter3d(
                x=t.baseline_x,
                y=t.baseline_y,
                z=t.baseline_z,
                mode="lines+markers",
                name="Normal",
                legendgroup="normal",
                showlegend=show_legend,
                marker=dict(size=3, color="#00F5FF"),
                line=dict(width=5, color="#00F5FF"),
            ),
            row=r,
            col=c,
        )
        fig_layer.add_trace(
            go.Scatter3d(
                x=t.injected_x,
                y=t.injected_y,
                z=t.injected_z,
                mode="lines+markers",
                name="Injected",
                legendgroup="inj",
                showlegend=show_legend,
                marker=dict(size=3, color="#FF007F"),
                line=dict(width=5, color="#FF007F"),
            ),
            row=r,
            col=c,
        )
        fig_layer.add_trace(
            go.Scatter3d(
                x=[t.injected_x[m]],
                y=[t.injected_y[m]],
                z=[t.injected_z[m]],
                mode="markers",
                name="Injection",
                legendgroup="mark",
                showlegend=show_legend,
                marker=dict(size=7, color="#FFFF00", symbol="diamond"),
            ),
            row=r,
            col=c,
        )

        fig_concepts.add_trace(
            go.Scatter3d(
                x=t.baseline_x,
                y=t.baseline_y,
                z=t.baseline_z3,
                mode="lines+markers",
                name="Normal",
                legendgroup="normal",
                showlegend=show_legend,
                marker=dict(size=3, color="#00F5FF"),
                line=dict(width=5, color="#00F5FF"),
            ),
            row=r,
            col=c,
        )
        fig_concepts.add_trace(
            go.Scatter3d(
                x=t.injected_x,
                y=t.injected_y,
                z=t.injected_z3,
                mode="lines+markers",
                name="Injected",
                legendgroup="inj",
                showlegend=show_legend,
                marker=dict(size=3, color="#FF007F"),
                line=dict(width=5, color="#FF007F"),
            ),
            row=r,
            col=c,
        )
        fig_concepts.add_trace(
            go.Scatter3d(
                x=[t.injected_x[m]],
                y=[t.injected_y[m]],
                z=[t.injected_z3[m]],
                mode="markers",
                name="Injection",
                legendgroup="mark",
                showlegend=show_legend,
                marker=dict(size=7, color="#FFFF00", symbol="diamond"),
            ),
            row=r,
            col=c,
        )

    fig_layer.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        title=(
            f"Residual Trajectory 3D (Layer Space): inject '{args.inject_concept}' at depth {args.inject_depth:.2f}, "
            f"alpha={args.alpha}, norm={args.normalize_mode}"
        ),
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=900 if len(trajectories) > 3 else 520,
    )
    fig_concepts.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        title=(
            f"Residual Trajectory 3D (Concept Space): X={args.concept_x}, Y={args.concept_y}, Z={args.concept_z}, "
            f"norm={args.normalize_mode}"
        ),
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        margin=dict(l=10, r=10, t=60, b=10),
        height=900 if len(trajectories) > 3 else 520,
    )

    total_scenes = rows * cols
    _darken_scenes(fig_layer, total_scenes)
    _darken_scenes(fig_concepts, total_scenes)

    for idx in range(1, len(trajectories) + 1):
        key = _scene_name(idx)
        fig_layer.update_layout(
            {
                key: dict(
                    xaxis_title=f"{args.concept_x.title()} intensity",
                    yaxis_title=f"{args.concept_y.title()} intensity",
                    zaxis_title="Layer",
                )
            }
        )
        fig_concepts.update_layout(
            {
                key: dict(
                    xaxis_title=f"{args.concept_x.title()} intensity",
                    yaxis_title=f"{args.concept_y.title()} intensity",
                    zaxis_title=f"{args.concept_z.title()} intensity",
                )
            }
        )

    layer_html = args.output_dir / "residual_3d_layer_space.html"
    concept_html = args.output_dir / "residual_3d_concept_space.html"
    metadata_json = args.output_dir / "residual_3d_plot_metadata.json"

    fig_layer.write_html(layer_html, include_plotlyjs="cdn")
    fig_concepts.write_html(concept_html, include_plotlyjs="cdn")
    metadata_json.write_text(json.dumps(exported_meta, indent=2), encoding="utf-8")

    layer_png = args.output_dir / "residual_3d_layer_space.png"
    concept_png = args.output_dir / "residual_3d_concept_space.png"
    try:
        fig_layer.write_image(layer_png, width=1800, height=900, scale=2)
        fig_concepts.write_image(concept_png, width=1800, height=900, scale=2)
        print(f"[3d] wrote {layer_png}")
        print(f"[3d] wrote {concept_png}")
    except Exception as exc:  # noqa: BLE001
        print(f"[3d] png export skipped: {exc}")

    print(f"[3d] wrote {layer_html}")
    print(f"[3d] wrote {concept_html}")
    print(f"[3d] wrote {metadata_json}")


if __name__ == "__main__":
    main()
