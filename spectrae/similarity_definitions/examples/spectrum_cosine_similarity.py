#!/usr/bin/env python3
"""Create pairwise_similarity.csv from sparse MS/MS peak-list cosine similarity."""

import argparse
import csv
import math


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_peaks(value, peak_delimiter=";", mz_intensity_delimiter=":"):
    peaks = []
    for peak_text in str(value).split(peak_delimiter):
        peak_text = peak_text.strip()
        if not peak_text:
            continue
        mz_text, intensity_text = peak_text.split(mz_intensity_delimiter, 1)
        peaks.append((float(mz_text), max(0.0, float(intensity_text))))
    return sorted(peaks)


def normalize(peaks):
    norm = math.sqrt(sum(intensity * intensity for _, intensity in peaks))
    if norm == 0.0:
        return [(mz, 0.0) for mz, _ in peaks]
    return [(mz, intensity / norm) for mz, intensity in peaks]


def peak_cosine(left, right, tolerance):
    left = normalize(left)
    right = normalize(right)
    i = 0
    j = 0
    score = 0.0
    while i < len(left) and j < len(right):
        left_mz, left_intensity = left[i]
        right_mz, right_intensity = right[j]
        delta = left_mz - right_mz
        if abs(delta) <= tolerance:
            score += left_intensity * right_intensity
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return max(0.0, min(1.0, score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-id-col", default="train_id")
    parser.add_argument("--eval-id-col", default="sample_id")
    parser.add_argument("--peaks-col", default="peaks")
    parser.add_argument("--peak-delimiter", default=";")
    parser.add_argument("--mz-intensity-delimiter", default=":")
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    if args.tolerance < 0.0:
        raise SystemExit("--tolerance must be non-negative")

    train_rows = read_rows(args.train)
    eval_rows = read_rows(args.eval)
    train_peaks = [
        (
            row[args.train_id_col],
            parse_peaks(row[args.peaks_col], args.peak_delimiter, args.mz_intensity_delimiter),
        )
        for row in train_rows
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "similarity"])
        writer.writeheader()
        for eval_row in eval_rows:
            eval_peaks = parse_peaks(
                eval_row[args.peaks_col],
                args.peak_delimiter,
                args.mz_intensity_delimiter,
            )
            for train_id, peaks in train_peaks:
                writer.writerow(
                    {
                        "sample_id": eval_row[args.eval_id_col],
                        "train_id": train_id,
                        "similarity": peak_cosine(eval_peaks, peaks, args.tolerance),
                    }
                )


if __name__ == "__main__":
    main()
