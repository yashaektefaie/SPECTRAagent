#!/usr/bin/env python3
"""Create pairwise_similarity.csv from cosine similarity over vector columns."""

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


def cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--embedding-cols-prefix", default="")
    parser.add_argument("--embedding-cols", default="")
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    columns = vector_columns(
        train_rows or eval_rows,
        [args.train_id_col, args.eval_id_col],
        args.embedding_cols_prefix,
        args.embedding_cols,
    )
    if not columns:
        raise SystemExit("No embedding columns found")
    train_vectors = [(row[args.train_id_col], as_vector(row, columns)) for row in train_rows]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_vector = as_vector(eval_row, columns)
            for train_id, train_vector in train_vectors:
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_id,
                        "similarity": cosine(eval_vector, train_vector),
                    }
                )


if __name__ == "__main__":
    main()
