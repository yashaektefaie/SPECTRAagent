#!/usr/bin/env python3
"""Create pairwise_similarity.csv from delimited set annotations."""

import argparse
import csv


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_set(value, delimiter, lowercase):
    tokens = [token.strip() for token in str(value).split(delimiter)]
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return {token for token in tokens if token}


def jaccard(left, right):
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--set-col", required=True)
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--case-sensitive", action="store_true")
    args = parser.parse_args()

    lowercase = not args.case_sensitive
    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    train_sets = [
        (
            row[args.train_id_col],
            parse_set(row.get(args.set_col, ""), args.delimiter, lowercase),
        )
        for row in train_rows
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_set = parse_set(eval_row.get(args.set_col, ""), args.delimiter, lowercase)
            for train_id, train_set in train_sets:
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_id,
                        "similarity": jaccard(eval_set, train_set),
                    }
                )


if __name__ == "__main__":
    main()
