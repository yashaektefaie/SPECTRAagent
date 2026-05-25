"""Find train-eval string pairs within an edit-distance threshold."""

import argparse
import csv
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd


def _qgrams(value: str, q: int) -> Set[str]:
    padded = "#" * (q - 1) + value + "$" * (q - 1)
    if len(padded) < q:
        return {padded}
    return {padded[index : index + q] for index in range(len(padded) - q + 1)}


def _edit_distance(left: str, right: str, max_distance: int) -> int:
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = current[0]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def _prefix(tokens: Set[str], order: Dict[str, int], max_distance: int) -> List[str]:
    ordered = sorted(tokens, key=lambda token: (order.get(token, 10**12), token))
    prefix_length = min(len(ordered), max_distance + 1)
    return ordered[:prefix_length] if ordered else []


def _similarity(distance: int, left: str, right: str) -> float:
    denominator = max(len(left), len(right), 1)
    return max(0.0, 1.0 - distance / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--train-id-col", default="")
    parser.add_argument("--text-col", required=True)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--max-distance", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--fallback-exact", action="store_true")
    args = parser.parse_args()

    if args.q <= 0 or args.max_distance < 0:
        raise ValueError("--q must be positive and --max-distance must be nonnegative")

    train_df = pd.read_csv(args.train)
    eval_df = pd.read_csv(args.eval)
    train_id_col = args.train_id_col or args.id_col
    for frame_name, frame, id_col in (
        ("train", train_df, train_id_col),
        ("eval", eval_df, args.id_col),
    ):
        if id_col not in frame.columns:
            raise ValueError("%s id column not found: %s" % (frame_name, id_col))
        if args.text_col not in frame.columns:
            raise ValueError("%s text column not found: %s" % (frame_name, args.text_col))

    train_texts = train_df[args.text_col].fillna("").astype(str).tolist()
    eval_texts = eval_df[args.text_col].fillna("").astype(str).tolist()
    train_qgrams = [_qgrams(value, args.q) for value in train_texts]
    eval_qgrams = [_qgrams(value, args.q) for value in eval_texts]
    counts = Counter(token for token_set in train_qgrams + eval_qgrams for token in token_set)
    order = {token: index for index, (token, _) in enumerate(counts.most_common()[::-1])}

    inverted: Dict[str, List[int]] = defaultdict(list)
    for train_index, tokens in enumerate(train_qgrams):
        for token in _prefix(tokens, order, args.max_distance):
            inverted[token].append(train_index)

    train_ids = train_df[train_id_col].astype(str).tolist()
    eval_ids = eval_df[args.id_col].astype(str).tolist()

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "train_id", "similarity", "rank", "method", "edit_distance", "candidate_source"],
        )
        writer.writeheader()
        for eval_id, eval_text, tokens in zip(eval_ids, eval_texts, eval_qgrams):
            candidates = set()
            for token in _prefix(tokens, order, args.max_distance):
                candidates.update(inverted.get(token, []))
            candidate_source = "qgram_prefix_filter"
            if not candidates and args.fallback_exact:
                candidates = set(range(len(train_texts)))
                candidate_source = "fallback_exact"

            scored: List[Tuple[int, float, int]] = []
            for train_index in candidates:
                distance = _edit_distance(eval_text, train_texts[train_index], args.max_distance)
                if distance <= args.max_distance:
                    scored.append(
                        (
                            train_index,
                            _similarity(distance, eval_text, train_texts[train_index]),
                            distance,
                        )
                    )
            scored.sort(key=lambda item: (item[1], -item[2]), reverse=True)
            if args.top_k > 0:
                scored = scored[: args.top_k]
            for rank, (train_index, similarity, distance) in enumerate(scored, start=1):
                writer.writerow(
                    {
                        "sample_id": eval_id,
                        "train_id": train_ids[train_index],
                        "similarity": similarity,
                        "rank": rank,
                        "method": "qgram_filter_exact_edit_join",
                        "edit_distance": distance,
                        "candidate_source": candidate_source,
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
