"""Compute top-k Tanimoto similarity for binary fingerprints or bit index sets."""

import argparse
import csv
from typing import List, Set

import pandas as pd


def _fingerprint_to_set(value: object, delimiter: str) -> Set[int]:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()
    if not text:
        return set()
    compact = text.replace(" ", "")
    if set(compact) <= {"0", "1"}:
        return {index for index, char in enumerate(compact) if char == "1"}
    return {int(token.strip()) for token in text.split(delimiter) if token.strip()}


def _tanimoto(left: Set[int], right: Set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _top(scored: List[tuple], top_k: int) -> List[tuple]:
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
    parser.add_argument("--fingerprint-col", default="fingerprint")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)
    train_id_col = args.train_id_col or args.id_col
    for frame_name, frame, id_col in (
        ("train", train_df, train_id_col),
        ("eval", eval_df, args.id_col),
    ):
        if id_col not in frame.columns:
            raise ValueError("%s id column not found: %s" % (frame_name, id_col))
        if args.fingerprint_col not in frame.columns:
            raise ValueError("%s fingerprint column not found: %s" % (frame_name, args.fingerprint_col))

    train_ids = train_df[train_id_col].astype(str).tolist()
    train_fps = [
        _fingerprint_to_set(value, args.delimiter)
        for value in train_df[args.fingerprint_col]
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "train_id", "similarity", "rank", "method"],
        )
        writer.writeheader()
        for _, row in eval_df.iterrows():
            eval_id = str(row[args.id_col])
            eval_fp = _fingerprint_to_set(row[args.fingerprint_col], args.delimiter)
            scored = _top(
                [
                    (train_index, _tanimoto(eval_fp, train_fp))
                    for train_index, train_fp in enumerate(train_fps)
                ],
                args.top_k,
            )
            for rank, (train_index, similarity) in enumerate(scored, start=1):
                writer.writerow(
                    {
                        "sample_id": eval_id,
                        "train_id": train_ids[train_index],
                        "similarity": similarity,
                        "rank": rank,
                        "method": "exact_bitset_tanimoto_topk",
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
