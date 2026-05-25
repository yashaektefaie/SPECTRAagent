#!/usr/bin/env python3
"""Create pairwise_similarity.csv from regularized Mahalanobis distance."""

import argparse
import csv
import math

import numpy as np


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def feature_columns(rows, id_columns, prefix, explicit):
    if explicit:
        return [column.strip() for column in explicit.split(",") if column.strip()]
    if prefix:
        return [column for column in (rows[0].keys() if rows else []) if column.startswith(prefix)]
    ignored = set(id_columns)
    return [column for column in (rows[0].keys() if rows else []) if column not in ignored]


def matrix(rows, columns):
    return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--feature-cols-prefix", default="")
    parser.add_argument("--feature-cols", default="")
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.regularization < 0.0 or args.scale <= 0.0:
        raise SystemExit("--regularization must be non-negative and --scale must be positive")

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    columns = feature_columns(
        train_rows or eval_rows,
        [args.train_id_col, args.eval_id_col],
        args.feature_cols_prefix,
        args.feature_cols,
    )
    if not columns:
        raise SystemExit("No feature columns found")

    train_matrix = matrix(train_rows, columns)
    if len(columns) == 1:
        variance = float(np.var(train_matrix[:, 0])) + args.regularization
        inverse_covariance = np.asarray([[1.0 / variance]])
    else:
        covariance = np.cov(train_matrix, rowvar=False)
        covariance = covariance + args.regularization * np.eye(len(columns))
        inverse_covariance = np.linalg.pinv(covariance)

    train_vectors = [(row[args.train_id_col], vector) for row, vector in zip(train_rows, train_matrix)]
    eval_matrix = matrix(eval_rows, columns)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row, eval_vector in zip(eval_rows, eval_matrix):
            for train_id, train_vector in train_vectors:
                delta = eval_vector - train_vector
                distance = math.sqrt(max(0.0, float(delta.T @ inverse_covariance @ delta)))
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_id,
                        "similarity": math.exp(-distance / args.scale),
                    }
                )


if __name__ == "__main__":
    main()
