#!/usr/bin/env python3
"""Create pairwise_similarity.csv from normalized sequence identity."""

import argparse
import csv


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def levenshtein(left, right):
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left_char == right_char else 1),
                )
            )
        previous = current
    return previous[-1]


def normalized_identity(left, right):
    left = str(left)
    right = str(right)
    if not left and not right:
        return 1.0
    if len(left) == len(right):
        mismatches = sum(1 for a, b in zip(left, right) if a != b)
        return 1.0 - (mismatches / max(1, len(left)))
    distance = levenshtein(left, right)
    return 1.0 - (distance / max(len(left), len(right), 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--sequence-col", default="sequence")
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            for train_row in train_rows:
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_row[args.train_id_col],
                        "similarity": normalized_identity(
                            eval_row[args.sequence_col],
                            train_row[args.sequence_col],
                        ),
                    }
                )


if __name__ == "__main__":
    main()

