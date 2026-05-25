"""Approximate cosine candidate generation with random hyperplane LSH."""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

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
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _signature(matrix: np.ndarray, projections: np.ndarray) -> List[str]:
    signs = matrix @ projections.T >= 0
    return ["".join("1" if bit else "0" for bit in row) for row in signs]


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
    parser.add_argument("--vector-cols", default="")
    parser.add_argument("--vector-prefix", default="emb_")
    parser.add_argument("--num-bits", type=int, default=16)
    parser.add_argument("--num-tables", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--fallback-exact", action="store_true")
    args = parser.parse_args()

    if args.num_bits <= 0 or args.num_tables <= 0:
        raise ValueError("--num-bits and --num-tables must be positive")

    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)
    train_id_col = args.train_id_col or args.id_col
    if train_id_col not in train_df.columns:
        raise ValueError("Train id column not found: %s" % train_id_col)
    if args.id_col not in eval_df.columns:
        raise ValueError("Eval id column not found: %s" % args.id_col)
    columns = _parse_columns(train_df, args.vector_cols, args.vector_prefix)
    eval_missing = [column for column in columns if column not in eval_df.columns]
    if eval_missing:
        raise ValueError("Eval file is missing vector columns: %s" % ", ".join(eval_missing))

    train_matrix = _vectors(train_df, columns)
    eval_matrix = _vectors(eval_df, columns)
    rng = np.random.default_rng(args.seed)
    train_ids = train_df[train_id_col].astype(str).to_numpy()
    eval_ids = eval_df[args.id_col].astype(str).to_numpy()

    bucket_tables: List[Dict[str, List[int]]] = []
    eval_sigs_by_table = []
    for _ in range(args.num_tables):
        projections = rng.normal(size=(args.num_bits, train_matrix.shape[1]))
        train_sigs = _signature(train_matrix, projections)
        eval_sigs = _signature(eval_matrix, projections)
        table: Dict[str, List[int]] = defaultdict(list)
        for index, signature in enumerate(train_sigs):
            table[signature].append(index)
        bucket_tables.append(table)
        eval_sigs_by_table.append(eval_sigs)

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "train_id", "similarity", "rank", "method", "candidate_source"],
        )
        writer.writeheader()
        for eval_index, eval_id in enumerate(eval_ids):
            candidate_indices = set()
            for table, eval_sigs in zip(bucket_tables, eval_sigs_by_table):
                candidate_indices.update(table.get(eval_sigs[eval_index], []))
            candidate_source = "random_hyperplane_lsh"
            if not candidate_indices and args.fallback_exact:
                candidate_indices = set(range(train_matrix.shape[0]))
                candidate_source = "fallback_exact"

            scored = _top(
                [
                    (train_index, float(eval_matrix[eval_index] @ train_matrix[train_index]))
                    for train_index in candidate_indices
                ],
                args.top_k,
            )
            for rank, (train_index, similarity) in enumerate(scored, start=1):
                writer.writerow(
                    {
                        "sample_id": eval_id,
                        "train_id": train_ids[train_index],
                        "similarity": (similarity + 1.0) / 2.0,
                        "rank": rank,
                        "method": "random_hyperplane_lsh_exact_cosine_rerank",
                        "candidate_source": candidate_source,
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
