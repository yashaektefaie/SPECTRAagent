"""Compute top-k entropic optimal-transport similarity for histogram rows."""

import argparse
import csv
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_columns(df: pd.DataFrame, explicit: str, prefix: str) -> List[str]:
    if explicit:
        columns = [column.strip() for column in explicit.split(",") if column.strip()]
    else:
        columns = [column for column in df.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError("No histogram columns found")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError("Missing histogram columns: %s" % ", ".join(missing))
    return columns


def _histograms(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    matrix = df.loc[:, list(columns)].to_numpy(dtype=float)
    matrix[matrix < 0] = 0.0
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0.0] = 1.0
    return matrix / totals


def _cost_matrix(size: int) -> np.ndarray:
    indices = np.arange(size, dtype=float)
    return np.abs(indices[:, None] - indices[None, :])


def _sinkhorn_cost(left: np.ndarray, right: np.ndarray, kernel: np.ndarray, cost: np.ndarray, iterations: int) -> float:
    u = np.ones_like(left)
    v = np.ones_like(right)
    eps = 1e-12
    for _ in range(iterations):
        u = left / np.maximum(kernel @ v, eps)
        v = right / np.maximum(kernel.T @ u, eps)
    plan = (u[:, None] * kernel) * v[None, :]
    return float(np.sum(plan * cost))


def _top(scored: List[Tuple[int, float]], top_k: int) -> List[Tuple[int, float]]:
    scored.sort(key=lambda item: item[1], reverse=True)
    if top_k > 0:
        return scored[:top_k]
    return scored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--train-id-col", default="")
    parser.add_argument("--histogram-cols", default="")
    parser.add_argument("--histogram-prefix", default="bin_")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--similarity-scale", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    if args.epsilon <= 0 or args.similarity_scale <= 0:
        raise ValueError("--epsilon and --similarity-scale must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)
    train_id_col = args.train_id_col or args.id_col
    if train_id_col not in train_df.columns:
        raise ValueError("Train id column not found: %s" % train_id_col)
    if args.id_col not in eval_df.columns:
        raise ValueError("Eval id column not found: %s" % args.id_col)
    columns = _parse_columns(train_df, args.histogram_cols, args.histogram_prefix)
    eval_missing = [column for column in columns if column not in eval_df.columns]
    if eval_missing:
        raise ValueError("Eval file is missing histogram columns: %s" % ", ".join(eval_missing))

    train_matrix = _histograms(train_df, columns)
    eval_matrix = _histograms(eval_df, columns)
    cost = _cost_matrix(train_matrix.shape[1])
    kernel = np.exp(-cost / args.epsilon)
    train_ids = train_df[train_id_col].astype(str).tolist()

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "train_id", "similarity", "rank", "method", "sinkhorn_cost"],
        )
        writer.writeheader()
        for eval_index, row in eval_df.iterrows():
            scored = []
            for train_index, train_hist in enumerate(train_matrix):
                distance = _sinkhorn_cost(
                    eval_matrix[eval_index],
                    train_hist,
                    kernel,
                    cost,
                    args.iterations,
                )
                similarity = float(np.exp(-distance / args.similarity_scale))
                scored.append((train_index, similarity, distance))
            scored.sort(key=lambda item: item[1], reverse=True)
            if args.top_k > 0:
                scored = scored[: args.top_k]
            for rank, (train_index, similarity, distance) in enumerate(scored, start=1):
                writer.writerow(
                    {
                        "sample_id": str(row[args.id_col]),
                        "train_id": train_ids[train_index],
                        "similarity": similarity,
                        "rank": rank,
                        "method": "sinkhorn_histogram_topk",
                        "sinkhorn_cost": distance,
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
