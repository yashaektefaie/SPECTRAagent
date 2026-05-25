"""Create SPECTRA pairwise candidates by blocking on metadata keys."""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd


def _parse_columns(value: str) -> List[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise ValueError("--block-cols must name at least one column")
    return columns


def _tokens(value: object, delimiter: str) -> Set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {token.strip() for token in str(value).split(delimiter) if token.strip()}


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _block_key(row: pd.Series, columns: List[str]) -> Tuple[str, ...]:
    return tuple(str(row[column]) for column in columns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--train-id-col", default="")
    parser.add_argument("--block-cols", required=True)
    parser.add_argument("--token-col", default="")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)
    train_id_col = args.train_id_col or args.id_col
    block_cols = _parse_columns(args.block_cols)
    for frame_name, frame, id_col in (
        ("train", train_df, train_id_col),
        ("eval", eval_df, args.id_col),
    ):
        missing = [column for column in block_cols if column not in frame.columns]
        if missing:
            raise ValueError("%s missing block columns: %s" % (frame_name, ", ".join(missing)))
        if id_col not in frame.columns:
            raise ValueError("%s id column not found: %s" % (frame_name, id_col))
        if args.token_col and args.token_col not in frame.columns:
            raise ValueError("%s token column not found: %s" % (frame_name, args.token_col))

    blocks: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for index, row in train_df.iterrows():
        blocks[_block_key(row, block_cols)].append(index)

    train_ids = train_df[train_id_col].astype(str).tolist()
    train_tokens = (
        [_tokens(value, args.delimiter) for value in train_df[args.token_col]]
        if args.token_col
        else []
    )

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "train_id", "similarity", "rank", "method", "candidate_source"],
        )
        writer.writeheader()
        for _, row in eval_df.iterrows():
            eval_id = str(row[args.id_col])
            candidate_indices = blocks.get(_block_key(row, block_cols), [])
            eval_tokens = _tokens(row[args.token_col], args.delimiter) if args.token_col else set()
            scored = []
            for train_index in candidate_indices:
                similarity = (
                    _jaccard(eval_tokens, train_tokens[train_index])
                    if args.token_col
                    else 1.0
                )
                scored.append((train_index, similarity))
            scored.sort(key=lambda item: item[1], reverse=True)
            if args.top_k > 0:
                scored = scored[: args.top_k]
            for rank, (train_index, similarity) in enumerate(scored, start=1):
                writer.writerow(
                    {
                        "sample_id": eval_id,
                        "train_id": train_ids[train_index],
                        "similarity": similarity,
                        "rank": rank,
                        "method": "blocking_exact_candidate_similarity",
                        "candidate_source": "metadata_block",
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
