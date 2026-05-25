#!/usr/bin/env python3
"""Create pairwise_similarity.csv from global alignment identity."""

import argparse
import csv


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def global_alignment_identity(left, right, gap_penalty=-1, match_score=1, mismatch_penalty=-1):
    left = str(left)
    right = str(right)
    if not left and not right:
        return 1.0

    n = len(left)
    m = len(right)
    scores = [[0] * (m + 1) for _ in range(n + 1)]
    moves = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        scores[i][0] = i * gap_penalty
        moves[i][0] = "up"
    for j in range(1, m + 1):
        scores[0][j] = j * gap_penalty
        moves[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal_score = match_score if left[i - 1] == right[j - 1] else mismatch_penalty
            choices = [
                (scores[i - 1][j - 1] + diagonal_score, "diag"),
                (scores[i - 1][j] + gap_penalty, "up"),
                (scores[i][j - 1] + gap_penalty, "left"),
            ]
            scores[i][j], moves[i][j] = max(choices, key=lambda value: value[0])

    i = n
    j = m
    matches = 0
    aligned = 0
    while i > 0 or j > 0:
        move = moves[i][j]
        if move == "diag":
            matches += 1 if left[i - 1] == right[j - 1] else 0
            aligned += 1
            i -= 1
            j -= 1
        elif move == "up":
            aligned += 1
            i -= 1
        else:
            aligned += 1
            j -= 1

    return matches / max(1, aligned)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--gap-penalty", type=int, default=-1)
    parser.add_argument("--match-score", type=int, default=1)
    parser.add_argument("--mismatch-penalty", type=int, default=-1)
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
                        "similarity": global_alignment_identity(
                            eval_row[args.sequence_col],
                            train_row[args.sequence_col],
                            gap_penalty=args.gap_penalty,
                            match_score=args.match_score,
                            mismatch_penalty=args.mismatch_penalty,
                        ),
                    }
                )


if __name__ == "__main__":
    main()
