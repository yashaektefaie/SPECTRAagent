#!/usr/bin/env python3
"""Create pairwise_similarity.csv from dynamic time warping distance."""

import argparse
import csv
import math


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_series(value, delimiter):
    return [float(token) for token in str(value).split(delimiter) if token.strip()]


def dtw_distance(left, right, window=None):
    n = len(left)
    m = len(right)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float("inf")
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))
    previous = [float("inf")] * (m + 1)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = [float("inf")] * (m + 1)
        start = max(1, i - window)
        stop = min(m, i + window)
        for j in range(start, stop + 1):
            cost = abs(left[i - 1] - right[j - 1])
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return previous[m] / (n + m)


def similarity(distance, scale, method):
    if not math.isfinite(distance):
        return 0.0
    if method == "exp":
        return math.exp(-distance / scale)
    if method == "inverse":
        return 1.0 / (1.0 + distance / scale)
    raise ValueError("Unknown method: %s" % method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--series-col", default="series")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--method", choices=["exp", "inverse"], default="exp")
    args = parser.parse_args()

    if args.scale <= 0.0:
        raise SystemExit("--scale must be positive")
    window = args.window if args.window > 0 else None

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    train_series = [
        (row[args.train_id_col], parse_series(row[args.series_col], args.delimiter))
        for row in train_rows
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_series = parse_series(eval_row[args.series_col], args.delimiter)
            for train_id, train_values in train_series:
                distance = dtw_distance(eval_series, train_values, window=window)
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_id,
                        "similarity": similarity(distance, args.scale, args.method),
                    }
                )


if __name__ == "__main__":
    main()
