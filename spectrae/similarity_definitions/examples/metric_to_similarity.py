#!/usr/bin/env python3
"""Convert precomputed pairwise metrics to SPECTRA pairwise similarities."""

import argparse
import csv
import math


def parse_metric(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Metric is not finite: %s" % value)
    return parsed


def distance_to_similarity(distance, method, scale):
    if method == "exp":
        return math.exp(-distance / scale)
    if method == "inverse":
        return 1.0 / (1.0 + distance / scale)
    if method == "linear":
        return max(0.0, 1.0 - distance / scale)
    raise ValueError("Unknown distance method: %s" % method)


def clip01(value):
    return max(0.0, min(1.0, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--metric-col", default="metric")
    parser.add_argument("--metric-kind", choices=["similarity", "distance"], required=True)
    parser.add_argument("--distance-method", choices=["exp", "inverse", "linear"], default="exp")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--no-clip", action="store_true")
    args = parser.parse_args()

    if args.scale <= 0.0:
        raise SystemExit("--scale must be positive")

    with open(args.input, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for row in rows:
            metric = parse_metric(row[args.metric_col])
            if args.metric_kind == "distance":
                similarity = distance_to_similarity(metric, args.distance_method, args.scale)
            else:
                similarity = metric
            if not args.no_clip:
                similarity = clip01(similarity)
            writer.writerow(
                {
                    "sample_id": row[args.sample_id_col],
                    "train_id": row[args.train_id_col],
                    "similarity": similarity,
                }
            )


if __name__ == "__main__":
    main()
