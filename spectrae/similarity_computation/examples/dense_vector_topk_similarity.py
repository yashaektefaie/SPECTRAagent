"""Compute top-k train neighbors for dense vector similarity.

This script is a local, dependency-light prototype for SPECTRA's generic
pairwise similarity contract. It is exact over the provided train/eval files;
for very large dense embeddings, use the same output contract with FAISS,
HNSW, or another ANN index.
"""

import argparse
import csv
import math
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_columns(df: pd.DataFrame, explicit: str, prefix: str) -> List[str]:
    if explicit:
        columns = [column.strip() for column in explicit.split(",") if column.strip()]
    else:
        columns = [column for column in df.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError("No vector columns found")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError("Missing vector columns: %s" % ", ".join(missing))
    return columns


def _vectors(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    matrix = df.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Vector columns contain non-finite values")
    return matrix


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _top_indices(scores: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    if top_k <= 0 or top_k >= scores.shape[0]:
        indices = np.arange(scores.shape[0])
    else:
        indices = np.argpartition(-scores, top_k - 1)[:top_k]
    order = np.argsort(-scores[indices])
    indices = indices[order]
    return indices, scores[indices]


def _score_chunk(
    train_matrix: np.ndarray,
    eval_matrix: np.ndarray,
    metric: str,
    distance_scale: float,
    raw_cosine: bool,
) -> np.ndarray:
    if metric in {"cosine", "dot"}:
        scores = eval_matrix @ train_matrix.T
        if metric == "cosine" and not raw_cosine:
            scores = (scores + 1.0) / 2.0
        return scores
    if metric == "euclidean":
        diff = eval_matrix[:, None, :] - train_matrix[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        return np.exp(-distances / distance_scale)
    raise ValueError("Unsupported metric: %s" % metric)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--train-id-col", default="")
    parser.add_argument("--vector-cols", default="")
    parser.add_argument("--vector-prefix", default="emb_")
    parser.add_argument("--metric", choices=("cosine", "dot", "euclidean"), default="cosine")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--raw-cosine", action="store_true")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)
    train_id_col = args.train_id_col or args.id_col
    if train_id_col not in train_df.columns:
        raise ValueError("Train id column not found: %s" % train_id_col)
    if args.id_col not in eval_df.columns:
        raise ValueError("Eval id column not found: %s" % args.id_col)
    if args.distance_scale <= 0:
        raise ValueError("--distance-scale must be positive")

    vector_cols = _parse_columns(train_df, args.vector_cols, args.vector_prefix)
    eval_missing = [column for column in vector_cols if column not in eval_df.columns]
    if eval_missing:
        raise ValueError("Eval file is missing vector columns: %s" % ", ".join(eval_missing))

    train_matrix = _vectors(train_df, vector_cols)
    eval_matrix = _vectors(eval_df, vector_cols)
    if args.metric == "cosine":
        train_matrix = _normalize_rows(train_matrix)
        eval_matrix = _normalize_rows(eval_matrix)

    train_ids = train_df[train_id_col].astype(str).to_numpy()
    eval_ids = eval_df[args.id_col].astype(str).to_numpy()
    chunk_size = max(1, args.chunk_size)

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "train_id", "similarity", "rank", "method"],
        )
        writer.writeheader()
        for start in range(0, eval_matrix.shape[0], chunk_size):
            stop = min(start + chunk_size, eval_matrix.shape[0])
            scores = _score_chunk(
                train_matrix=train_matrix,
                eval_matrix=eval_matrix[start:stop],
                metric=args.metric,
                distance_scale=args.distance_scale,
                raw_cosine=args.raw_cosine,
            )
            for local_index, row_scores in enumerate(scores):
                indices, values = _top_indices(row_scores, args.top_k)
                eval_id = eval_ids[start + local_index]
                for rank, (train_index, similarity) in enumerate(zip(indices, values), start=1):
                    if not math.isfinite(float(similarity)):
                        continue
                    writer.writerow(
                        {
                            "sample_id": eval_id,
                            "train_id": train_ids[train_index],
                            "similarity": float(similarity),
                            "rank": rank,
                            "method": "exact_dense_%s_topk" % args.metric,
                        }
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
