#!/usr/bin/env python3
"""Create pairwise_similarity.csv from Jensen-Shannon histogram similarity."""

import argparse
import csv
import math


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def histogram_columns(rows, id_columns, prefix, explicit):
    if explicit:
        return [column.strip() for column in explicit.split(",") if column.strip()]
    if prefix:
        return [column for column in (rows[0].keys() if rows else []) if column.startswith(prefix)]
    ignored = set(id_columns)
    return [column for column in (rows[0].keys() if rows else []) if column not in ignored]


def normalize(values):
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total == 0.0:
        return [1.0 / len(clipped)] * len(clipped)
    return [value / total for value in clipped]


def kl_divergence(left, right):
    total = 0.0
    for left_value, right_value in zip(left, right):
        if left_value > 0.0 and right_value > 0.0:
            total += left_value * math.log(left_value / right_value)
    return total


def js_similarity(left, right):
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right)]
    divergence = 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint)
    return max(0.0, 1.0 - divergence / math.log(2.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--hist-cols-prefix", default="")
    parser.add_argument("--hist-cols", default="")
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    columns = histogram_columns(
        train_rows or eval_rows,
        [args.train_id_col, args.eval_id_col],
        args.hist_cols_prefix,
        args.hist_cols,
    )
    if not columns:
        raise SystemExit("No histogram columns found")
    train_histograms = [
        (row[args.train_id_col], normalize([row[column] for column in columns]))
        for row in train_rows
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_histogram = normalize([eval_row[column] for column in columns])
            for train_id, train_histogram in train_histograms:
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_id,
                        "similarity": js_similarity(eval_histogram, train_histogram),
                    }
                )


if __name__ == "__main__":
    main()
