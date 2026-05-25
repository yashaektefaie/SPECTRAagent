"""Compute top-k generalized Jaccard similarity for sparse weighted features."""

import argparse
import csv
from typing import Dict, List, Tuple

import pandas as pd


def _parse_weighted(value: object, item_delimiter: str, key_value_delimiter: str) -> Dict[str, float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    result: Dict[str, float] = {}
    for item in str(value).split(item_delimiter):
        item = item.strip()
        if not item:
            continue
        if key_value_delimiter in item:
            key, raw_weight = item.split(key_value_delimiter, 1)
            weight = float(raw_weight)
        else:
            key = item
            weight = 1.0
        if weight > 0:
            result[key.strip()] = result.get(key.strip(), 0.0) + weight
    return result


def _weighted_jaccard(left: Dict[str, float], right: Dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator else 0.0


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
    parser.add_argument("--feature-col", required=True)
    parser.add_argument("--item-delimiter", default=";")
    parser.add_argument("--key-value-delimiter", default=":")
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
        if args.feature_col not in frame.columns:
            raise ValueError("%s feature column not found: %s" % (frame_name, args.feature_col))

    train_ids = train_df[train_id_col].astype(str).tolist()
    train_features = [
        _parse_weighted(value, args.item_delimiter, args.key_value_delimiter)
        for value in train_df[args.feature_col]
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "train_id", "similarity", "rank", "method"],
        )
        writer.writeheader()
        for _, row in eval_df.iterrows():
            eval_id = str(row[args.id_col])
            eval_features = _parse_weighted(
                row[args.feature_col],
                args.item_delimiter,
                args.key_value_delimiter,
            )
            scored = _top(
                [
                    (train_index, _weighted_jaccard(eval_features, train_feature))
                    for train_index, train_feature in enumerate(train_features)
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
                        "method": "exact_weighted_jaccard_topk",
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
