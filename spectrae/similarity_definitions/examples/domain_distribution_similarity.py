#!/usr/bin/env python3
"""Create pairwise_similarity.csv from domain-level feature distribution similarity."""

import argparse
import csv
import math
from collections import defaultdict


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


def squared_distance(left, right):
    return sum((a - b) ** 2 for a, b in zip(left, right))


def rbf(left, right, gamma):
    return math.exp(-gamma * squared_distance(left, right))


def mean_kernel(left_vectors, right_vectors, gamma):
    if not left_vectors or not right_vectors:
        return 0.0
    total = 0.0
    count = 0
    for left in left_vectors:
        for right in right_vectors:
            total += rbf(left, right, gamma)
            count += 1
    return total / count


def mmd2(left_vectors, right_vectors, gamma):
    return max(
        0.0,
        mean_kernel(left_vectors, left_vectors, gamma)
        + mean_kernel(right_vectors, right_vectors, gamma)
        - 2.0 * mean_kernel(left_vectors, right_vectors, gamma),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--domain-col", required=True)
    parser.add_argument("--feature-cols-prefix", default="")
    parser.add_argument("--feature-cols", default="")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.gamma <= 0.0 or args.scale <= 0.0:
        raise SystemExit("--gamma and --scale must be positive")

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    columns = vector_columns(
        train_rows or eval_rows,
        [args.train_id_col, args.eval_id_col, args.domain_col],
        args.feature_cols_prefix,
        args.feature_cols,
    )
    if not columns:
        raise SystemExit("No feature columns found")

    domain_vectors = defaultdict(list)
    for row in train_rows + eval_rows:
        domain_vectors[row[args.domain_col]].append(as_vector(row, columns))

    domain_similarity = {}
    domains = sorted(domain_vectors)
    for left in domains:
        for right in domains:
            distance = mmd2(domain_vectors[left], domain_vectors[right], args.gamma)
            domain_similarity[(left, right)] = math.exp(-distance / args.scale)

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_domain = eval_row[args.domain_col]
            for train_row in train_rows:
                train_domain = train_row[args.domain_col]
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_row[args.train_id_col],
                        "similarity": domain_similarity[(eval_domain, train_domain)],
                    }
                )


if __name__ == "__main__":
    main()
