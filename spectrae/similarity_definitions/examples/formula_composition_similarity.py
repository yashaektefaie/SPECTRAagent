#!/usr/bin/env python3
"""Create pairwise_similarity.csv from simple chemical formula composition."""

import argparse
import csv
import math
import re


FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_formula(formula):
    counts = {}
    for element, amount in FORMULA_TOKEN.findall(formula):
        counts[element] = counts.get(element, 0.0) + (float(amount) if amount else 1.0)
    if not counts:
        raise ValueError("Could not parse formula: %s" % formula)
    total = sum(counts.values())
    return {element: value / total for element, value in counts.items()}


def cosine(left, right):
    elements = set(left) | set(right)
    dot = sum(left.get(element, 0.0) * right.get(element, 0.0) for element in elements)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def l1_overlap(left, right):
    elements = set(left) | set(right)
    distance = sum(abs(left.get(element, 0.0) - right.get(element, 0.0)) for element in elements)
    return max(0.0, 1.0 - 0.5 * distance)


def composition_similarity(left, right, metric):
    if metric == "cosine":
        return cosine(left, right)
    if metric == "l1_overlap":
        return l1_overlap(left, right)
    raise ValueError("Unknown metric: %s" % metric)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--formula-col", default="formula")
    parser.add_argument("--metric", choices=["cosine", "l1_overlap"], default="cosine")
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    train_vectors = [
        (row[args.train_id_col], parse_formula(row[args.formula_col])) for row in train_rows
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_vector = parse_formula(eval_row[args.formula_col])
            for train_id, train_vector in train_vectors:
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_id,
                        "similarity": composition_similarity(eval_vector, train_vector, args.metric),
                    }
                )


if __name__ == "__main__":
    main()
