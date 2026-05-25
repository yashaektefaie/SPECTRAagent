#!/usr/bin/env python3
"""Create pairwise_similarity.csv from Pearson correlation over vector columns."""

import argparse
import csv
import math


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def vector_columns(rows, id_columns, prefix, explicit):
    if explicit:
        return [column.strip() for column in explicit.split(",") if column.strip()]
    if prefix:
        columns = rows[0].keys() if rows else []
        return [column for column in columns if column.startswith(prefix)]
    ignored = set(id_columns)
    return [column for column in (rows[0].keys() if rows else []) if column not in ignored]


def as_vector(row, columns):
    return [float(row[column]) for column in columns]


def pearson(left, right):
    if len(left) != len(right):
        raise ValueError("Vectors must have equal length")
    if not left:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    left_norm = math.sqrt(sum(a * a for a in left_centered))
    right_norm = math.sqrt(sum(b * b for b in right_centered))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0 if left == right else 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--feature-cols-prefix", default="")
    parser.add_argument("--feature-cols", default="")
    parser.add_argument(
        "--raw-correlation",
        action="store_true",
        help="Write Pearson r directly instead of mapping r from [-1, 1] to [0, 1].",
    )
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    columns = vector_columns(
        train_rows or eval_rows,
        [args.train_id_col, args.eval_id_col],
        args.feature_cols_prefix,
        args.feature_cols,
    )
    if not columns:
        raise SystemExit("No feature columns found")
    train_vectors = [(row[args.train_id_col], as_vector(row, columns)) for row in train_rows]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_vector = as_vector(eval_row, columns)
            for train_id, train_vector in train_vectors:
                correlation = pearson(eval_vector, train_vector)
                similarity = correlation if args.raw_correlation else (correlation + 1.0) / 2.0
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_id,
                        "similarity": similarity,
                    }
                )


if __name__ == "__main__":
    main()

