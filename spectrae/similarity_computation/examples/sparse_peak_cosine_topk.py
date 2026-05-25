"""Compute top-k sparse cosine similarity for peak-list spectra."""

import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd


def _parse_peaks(value: object, peak_delimiter: str, mz_intensity_delimiter: str, bin_width: float) -> Dict[int, float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    bins: Dict[int, float] = defaultdict(float)
    for item in str(value).split(peak_delimiter):
        item = item.strip()
        if not item:
            continue
        mz_text, intensity_text = item.split(mz_intensity_delimiter, 1)
        mz = float(mz_text)
        intensity = float(intensity_text)
        if intensity > 0:
            bins[int(round(mz / bin_width))] += intensity
    norm = math.sqrt(sum(value * value for value in bins.values()))
    if norm > 0:
        for key in list(bins):
            bins[key] /= norm
    return dict(bins)


def _sparse_cosine(left: Dict[int, float], right: Dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def _top(scored: List[Tuple[int, float]], top_k: int) -> List[Tuple[int, float]]:
    scored.sort(key=lambda item: item[1], reverse=True)
    if top_k > 0:
        return scored[:top_k]
    return scored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--train-id-col", default="")
    parser.add_argument("--peak-col", required=True)
    parser.add_argument("--peak-delimiter", default=";")
    parser.add_argument("--mz-intensity-delimiter", default=":")
    parser.add_argument("--bin-width", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")

    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)
    train_id_col = args.train_id_col or args.id_col
    for frame_name, frame, id_col in (
        ("train", train_df, train_id_col),
        ("eval", eval_df, args.id_col),
    ):
        if id_col not in frame.columns:
            raise ValueError("%s id column not found: %s" % (frame_name, id_col))
        if args.peak_col not in frame.columns:
            raise ValueError("%s peak column not found: %s" % (frame_name, args.peak_col))

    train_ids = train_df[train_id_col].astype(str).tolist()
    train_spectra = [
        _parse_peaks(value, args.peak_delimiter, args.mz_intensity_delimiter, args.bin_width)
        for value in train_df[args.peak_col]
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "train_id", "similarity", "rank", "method"],
        )
        writer.writeheader()
        for _, row in eval_df.iterrows():
            eval_spectrum = _parse_peaks(
                row[args.peak_col],
                args.peak_delimiter,
                args.mz_intensity_delimiter,
                args.bin_width,
            )
            scored = _top(
                [
                    (train_index, _sparse_cosine(eval_spectrum, train_spectrum))
                    for train_index, train_spectrum in enumerate(train_spectra)
                ],
                args.top_k,
            )
            for rank, (train_index, similarity) in enumerate(scored, start=1):
                writer.writerow(
                    {
                        "sample_id": str(row[args.id_col]),
                        "train_id": train_ids[train_index],
                        "similarity": similarity,
                        "rank": rank,
                        "method": "binned_sparse_peak_cosine_topk",
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
