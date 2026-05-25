#!/usr/bin/env python3
"""Create pairwise_similarity.csv from point-cloud Chamfer distance."""

import argparse
import csv
import math


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_points(value, point_delimiter=";", coord_delimiter=":"):
    points = []
    for point_text in str(value).split(point_delimiter):
        point_text = point_text.strip()
        if not point_text:
            continue
        coords = [float(token) for token in point_text.split(coord_delimiter)]
        points.append(coords)
    return points


def squared_distance(left, right):
    return sum((a - b) ** 2 for a, b in zip(left, right))


def mean_nearest(source, target):
    if not source and not target:
        return 0.0
    if not source or not target:
        return float("inf")
    return sum(min(squared_distance(point, other) for other in target) for point in source) / len(source)


def chamfer_distance(left, right):
    return 0.5 * (mean_nearest(left, right) + mean_nearest(right, left))


def similarity(distance, scale):
    if not math.isfinite(distance):
        return 0.0
    return math.exp(-distance / scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--points-col", default="points")
    parser.add_argument("--point-delimiter", default=";")
    parser.add_argument("--coord-delimiter", default=":")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.scale <= 0.0:
        raise SystemExit("--scale must be positive")

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    train_points = [
        (
            row[args.train_id_col],
            parse_points(row[args.points_col], args.point_delimiter, args.coord_delimiter),
        )
        for row in train_rows
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_points = parse_points(
                eval_row[args.points_col],
                args.point_delimiter,
                args.coord_delimiter,
            )
            for train_id, train_cloud in train_points:
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_id,
                        "similarity": similarity(chamfer_distance(eval_points, train_cloud), args.scale),
                    }
                )


if __name__ == "__main__":
    main()
