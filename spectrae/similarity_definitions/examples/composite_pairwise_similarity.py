#!/usr/bin/env python3
"""Combine multiple pairwise_similarity.csv files into one similarity axis."""

import argparse
import csv


def parse_input(value):
    if ":" not in value:
        return value, 1.0
    path, weight = value.rsplit(":", 1)
    return path, float(weight)


def read_similarity_file(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["sample_id"], row["train_id"])
            rows[key] = float(row["similarity"])
    return rows


def combine(values, weights, method):
    if method == "min":
        return min(values)
    if method == "max":
        return max(values)
    if method == "noisy_or":
        product = 1.0
        for value in values:
            product *= 1.0 - value
        return 1.0 - product
    if method == "mean":
        return sum(values) / len(values)
    if method == "weighted_mean":
        total_weight = sum(weights)
        if total_weight == 0.0:
            raise ValueError("Sum of weights is zero")
        return sum(value * weight for value, weight in zip(values, weights)) / total_weight
    raise ValueError("Unknown method: %s" % method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--method",
        choices=["mean", "weighted_mean", "min", "max", "noisy_or"],
        default="mean",
    )
    args = parser.parse_args()

    parsed_inputs = [parse_input(value) for value in args.inputs]
    matrices = [read_similarity_file(path) for path, _weight in parsed_inputs]
    weights = [weight for _path, weight in parsed_inputs]
    if not matrices:
        raise SystemExit("No input matrices supplied")

    keys = set(matrices[0])
    for matrix in matrices[1:]:
        if set(matrix) != keys:
            raise SystemExit("All input files must contain the same sample_id/train_id pairs")

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for sample_id, train_id in sorted(keys):
            values = [matrix[(sample_id, train_id)] for matrix in matrices]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "train_id": train_id,
                    "similarity": combine(values, weights, args.method),
                }
            )


if __name__ == "__main__":
    main()
