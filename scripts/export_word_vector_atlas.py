from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export concept word-centroid vector atlas stats.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model ids.",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        required=True,
        help="Concept names.",
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
        default=Path("C:/Users/joshj/joseph-stroud-identity-stability-research/output/word_vector_atlas"),
    )
    return parser.parse_args()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    _add_src_to_path()
    from identity_stability.modeling import clear_cuda, load_model
    from identity_stability.prompt_bank import get_concept_words
    from identity_stability.vectors import estimate_word_centroid_vector

    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        print(f"[atlas] loading {model_id}")
        loaded = load_model(
            model_id=model_id,
            cache_dir=args.cache_dir,
            dtype_name=args.dtype,
            use_gpu=args.use_gpu,
        )
        try:
            concept_vecs: dict[str, np.ndarray] = {}
            rows = []
            for concept in args.concepts:
                pos, neg = get_concept_words(concept)
                vec_est = estimate_word_centroid_vector(
                    loaded=loaded,
                    positive_words=pos,
                    negative_words=neg,
                )
                vec = vec_est.vector.numpy().astype(np.float32)
                concept_vecs[concept] = vec
                rows.append(
                    {
                        "model_id": model_id,
                        "concept": concept,
                        "norm": float(np.linalg.norm(vec)),
                        "pos_words": ",".join(pos),
                        "neg_words": ",".join(neg),
                        "dim": int(vec.shape[0]),
                    }
                )

            base = model_id.replace("/", "__")
            model_dir = args.output_dir / base
            model_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(rows).to_csv(model_dir / "word_vector_norms.csv", index=False)

            concepts = list(concept_vecs)
            mat = np.zeros((len(concepts), len(concepts)), dtype=np.float32)
            for i, ci in enumerate(concepts):
                for j, cj in enumerate(concepts):
                    mat[i, j] = _cos(concept_vecs[ci], concept_vecs[cj])

            sim_df = pd.DataFrame(mat, index=concepts, columns=concepts)
            sim_df.to_csv(model_dir / "concept_cosine_matrix.csv")

            nearest_rows = []
            for i, ci in enumerate(concepts):
                order = np.argsort(-mat[i])
                top = [concepts[k] for k in order if concepts[k] != ci][:5]
                nearest_rows.append(
                    {
                        "concept": ci,
                        "top_neighbors": ",".join(top),
                    }
                )
            pd.DataFrame(nearest_rows).to_csv(model_dir / "concept_neighbors.csv", index=False)

            with (model_dir / "vectors_metadata.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_id": model_id,
                        "concept_count": len(concepts),
                        "concepts": concepts,
                    },
                    f,
                    indent=2,
                )
            print(f"[atlas] wrote {model_dir}")
        finally:
            del loaded
            clear_cuda()


if __name__ == "__main__":
    main()
