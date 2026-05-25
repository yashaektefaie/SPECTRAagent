#!/usr/bin/env python3
"""Create pairwise_similarity.csv from latitude/longitude distance."""

import argparse
import csv
import math


EARTH_RADIUS_KM = 6371.0088


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def radians(value):
    return math.radians(float(value))


def haversine_km(left, right, lat_col, lon_col):
    lat1 = radians(left[lat_col])
    lon1 = radians(left[lon_col])
    lat2 = radians(right[lat_col])
    lon2 = radians(right[lon_col])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def similarity(distance_km, scale_km, kernel):
    if kernel == "exp":
        return math.exp(-distance_km / scale_km)
    if kernel == "inverse":
        return 1.0 / (1.0 + distance_km / scale_km)
    if kernel == "linear":
        return max(0.0, 1.0 - distance_km / scale_km)
    raise ValueError("Unknown kernel: %s" % kernel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--lat-col", default="lat")
    parser.add_argument("--lon-col", default="lon")
    parser.add_argument("--scale-km", type=float, default=1000.0)
    parser.add_argument("--kernel", choices=["exp", "inverse", "linear"], default="exp")
    args = parser.parse_args()

    if args.scale_km <= 0.0:
        raise SystemExit("--scale-km must be positive")

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            for train_row in train_rows:
                distance_km = haversine_km(eval_row, train_row, args.lat_col, args.lon_col)
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_row[args.train_id_col],
                        "similarity": similarity(distance_km, args.scale_km, args.kernel),
                    }
                )


if __name__ == "__main__":
    main()
