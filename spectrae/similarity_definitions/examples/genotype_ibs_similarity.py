#!/usr/bin/env python3
"""Create pairwise_similarity.csv from biallelic genotype dosage IBS similarity."""

import argparse
import csv


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def marker_columns(rows, id_columns, prefix, explicit):
    if explicit:
        return [column.strip() for column in explicit.split(",") if column.strip()]
    if prefix:
        columns = rows[0].keys() if rows else []
        return [column for column in columns if column.startswith(prefix)]
    ignored = set(id_columns)
    return [column for column in (rows[0].keys() if rows else []) if column not in ignored]


def dosage(value):
    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "NAN", "NULL"}:
        return None
    return float(text)


def ibs_similarity(left_row, right_row, columns):
    similarities = []
    for column in columns:
        left = dosage(left_row[column])
        right = dosage(right_row[column])
        if left is None or right is None:
            continue
        similarities.append(max(0.0, 1.0 - abs(left - right) / 2.0))
    if not similarities:
        return 0.0
    return sum(similarities) / len(similarities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--marker-cols-prefix", default="")
    parser.add_argument("--marker-cols", default="")
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    columns = marker_columns(
        train_rows or eval_rows,
        [args.train_id_col, args.eval_id_col],
        args.marker_cols_prefix,
        args.marker_cols,
    )
    if not columns:
        raise SystemExit("No marker columns found")

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            for train_row in train_rows:
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_row[args.train_id_col],
                        "similarity": ibs_similarity(eval_row, train_row, columns),
                    }
                )


if __name__ == "__main__":
    main()

