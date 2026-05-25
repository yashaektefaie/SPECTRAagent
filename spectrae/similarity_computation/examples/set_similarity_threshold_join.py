"""Train-eval set similarity join with prefix-filter candidate generation."""

import argparse
import csv
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd


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


def _prefix(tokens: Set[str], order: Dict[str, int], threshold: float) -> List[str]:
    ordered = sorted(tokens, key=lambda token: (order.get(token, 10**12), token))
    prefix_length = max(1, len(ordered) - int(threshold * len(ordered)) + 1)
    return ordered[:prefix_length]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--train-id-col", default="")
    parser.add_argument("--set-col", required=True)
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    if args.threshold <= 0 or args.threshold > 1:
        raise ValueError("--threshold must be in (0, 1]")

    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)
    train_id_col = args.train_id_col or args.id_col
    for frame_name, frame, id_col in (
        ("train", train_df, train_id_col),
        ("eval", eval_df, args.id_col),
    ):
        if id_col not in frame.columns:
            raise ValueError("%s id column not found: %s" % (frame_name, id_col))
        if args.set_col not in frame.columns:
            raise ValueError("%s set column not found: %s" % (frame_name, args.set_col))

    train_sets = [_tokens(value, args.delimiter) for value in train_df[args.set_col]]
    eval_sets = [_tokens(value, args.delimiter) for value in eval_df[args.set_col]]
    token_counts = Counter(token for token_set in train_sets + eval_sets for token in token_set)
    token_order = {token: index for index, (token, _) in enumerate(token_counts.most_common()[::-1])}

    inverted: Dict[str, List[int]] = defaultdict(list)
    for train_index, token_set in enumerate(train_sets):
        for token in _prefix(token_set, token_order, args.threshold):
            inverted[token].append(train_index)

    train_ids = train_df[train_id_col].astype(str).tolist()
    eval_ids = eval_df[args.id_col].astype(str).tolist()

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "train_id", "similarity", "rank", "method", "candidate_source"],
        )
        writer.writeheader()
        for eval_id, token_set in zip(eval_ids, eval_sets):
            candidates = set()
            for token in _prefix(token_set, token_order, args.threshold):
                candidates.update(inverted.get(token, []))
            scored: List[Tuple[int, float]] = []
            for train_index in candidates:
                similarity = _jaccard(token_set, train_sets[train_index])
                if similarity >= args.threshold:
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
                        "method": "prefix_filter_exact_jaccard_join",
                        "candidate_source": "prefix_filter",
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
