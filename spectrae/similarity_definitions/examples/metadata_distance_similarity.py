#!/usr/bin/env python3
"""Create pairwise_similarity.csv from mixed categorical/numeric metadata."""

import argparse
import csv
import math


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def component_similarity(left, right):
    left_num = as_float(left)
    right_num = as_float(right)
    if left_num is not None and right_num is not None:
        return 1.0 / (1.0 + abs(left_num - right_num))
    return 1.0 if str(left) == str(right) else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--columns", nargs="+", required=True)
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            for train_row in train_rows:
                values = [
                    component_similarity(eval_row.get(column, ""), train_row.get(column, ""))
                    for column in args.columns
                ]
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_row[args.train_id_col],
                        "similarity": sum(values) / len(values),
                    }
                )


if __name__ == "__main__":
    main()

