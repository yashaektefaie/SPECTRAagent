#!/usr/bin/env python3
"""Create pairwise_similarity.csv from compositional Bray-Curtis similarity."""

import argparse
import csv


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value_columns(rows, id_columns, prefix, explicit):
    if explicit:
        return [column.strip() for column in explicit.split(",") if column.strip()]
    if prefix:
        return [column for column in (rows[0].keys() if rows else []) if column.startswith(prefix)]
    ignored = set(id_columns)
    return [column for column in (rows[0].keys() if rows else []) if column not in ignored]


def vector(row, columns):
    return [max(0.0, float(row[column])) for column in columns]


def braycurtis_similarity(left, right):
    denominator = sum(left) + sum(right)
    if denominator == 0.0:
        return 1.0
    distance = sum(abs(a - b) for a, b in zip(left, right)) / denominator
    return max(0.0, min(1.0, 1.0 - distance))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--feature-cols-prefix", default="")
    parser.add_argument("--feature-cols", default="")
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    columns = value_columns(
        train_rows or eval_rows,
        [args.train_id_col, args.eval_id_col],
        args.feature_cols_prefix,
        args.feature_cols,
    )
    if not columns:
        raise SystemExit("No compositional feature columns found")

    train_vectors = [(row[args.train_id_col], vector(row, columns)) for row in train_rows]
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_vector = vector(eval_row, columns)
            for train_id, train_vector in train_vectors:
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_id,
                        "similarity": braycurtis_similarity(eval_vector, train_vector),
                    }
                )


if __name__ == "__main__":
    main()
