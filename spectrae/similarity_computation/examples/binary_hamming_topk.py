"""Compute top-k Hamming similarity for binary codes."""

import argparse
import csv
from typing import List, Set, Tuple

import pandas as pd


def _code_to_set(value: object, delimiter: str) -> Tuple[Set[int], int]:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()
    if not text:
        return set(), 0
    compact = text.replace(" ", "")
    if set(compact) <= {"0", "1"}:
        return {index for index, char in enumerate(compact) if char == "1"}, len(compact)
    bits = {int(token.strip()) for token in text.split(delimiter) if token.strip()}
    return bits, max(bits) + 1 if bits else 0


def _hamming_similarity(left: Set[int], right: Set[int], bit_length: int) -> float:
    if bit_length <= 0:
        return 1.0
    distance = len(left ^ right)
    return max(0.0, 1.0 - distance / bit_length)


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
    parser.add_argument("--code-col", required=True)
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--bit-length", type=int, default=0)
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
        if args.code_col not in frame.columns:
            raise ValueError("%s code column not found: %s" % (frame_name, args.code_col))

    train_codes = [_code_to_set(value, args.delimiter) for value in train_df[args.code_col]]
    eval_codes = [_code_to_set(value, args.delimiter) for value in eval_df[args.code_col]]
    inferred_length = max([length for _, length in train_codes + eval_codes] or [0])
    bit_length = args.bit_length or inferred_length
    if bit_length <= 0:
        raise ValueError("Could not infer bit length; pass --bit-length")

    train_ids = train_df[train_id_col].astype(str).tolist()
    eval_ids = eval_df[args.id_col].astype(str).tolist()

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "train_id", "similarity", "rank", "method", "hamming_distance"],
        )
        writer.writeheader()
        for eval_id, (eval_bits, _) in zip(eval_ids, eval_codes):
            scored = _top(
                [
                    (train_index, _hamming_similarity(eval_bits, train_bits, bit_length))
                    for train_index, (train_bits, _) in enumerate(train_codes)
                ],
                args.top_k,
            )
            for rank, (train_index, similarity) in enumerate(scored, start=1):
                train_bits = train_codes[train_index][0]
                writer.writerow(
                    {
                        "sample_id": eval_id,
                        "train_id": train_ids[train_index],
                        "similarity": similarity,
                        "rank": rank,
                        "method": "exact_binary_hamming_topk",
                        "hamming_distance": len(eval_bits ^ train_bits),
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
