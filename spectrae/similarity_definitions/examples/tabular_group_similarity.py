#!/usr/bin/env python3
"""Create pairwise_similarity.csv from shared categorical group membership."""

import argparse
import csv


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--group-col", action="append", required=True)
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_key = tuple(eval_row.get(column, "") for column in args.group_col)
            for train_row in train_rows:
                train_key = tuple(train_row.get(column, "") for column in args.group_col)
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_row[args.train_id_col],
                        "similarity": 1.0 if eval_key == train_key else 0.0,
                    }
                )


if __name__ == "__main__":
    main()

