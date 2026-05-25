"""Generate train-eval Jaccard candidates with MinHash LSH and exact reranking."""

import argparse
import csv
import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


MAX_HASH = (1 << 64) - 1


def _tokens(value: object, delimiter: str) -> Set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {token.strip() for token in str(value).split(delimiter) if token.strip()}


def _hash_token(token: str, seed: int, hash_index: int) -> int:
    payload = ("%d|%d|%s" % (seed, hash_index, token)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _signature(tokens: Set[str], num_hashes: int, seed: int) -> Tuple[int, ...]:
    if not tokens:
        return tuple([MAX_HASH] * num_hashes)
    values: List[int] = []
    for hash_index in range(num_hashes):
        values.append(min(_hash_token(token, seed, hash_index) for token in tokens))
    return tuple(values)


def _bands(signature: Sequence[int], rows_per_band: int) -> Iterable[Tuple[int, Tuple[int, ...]]]:
    band_index = 0
    for start in range(0, len(signature), rows_per_band):
        chunk = tuple(signature[start : start + rows_per_band])
        if len(chunk) == rows_per_band:
            yield band_index, chunk
        band_index += 1


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--train-id-col", default="")
    parser.add_argument("--set-col", required=True)
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--num-hashes", type=int, default=128)
    parser.add_argument("--rows-per-band", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--fallback-exact", action="store_true")
    args = parser.parse_args()

    if args.num_hashes <= 0 or args.rows_per_band <= 0:
        raise ValueError("Hash and band sizes must be positive")
    if args.num_hashes % args.rows_per_band != 0:
        raise ValueError("--num-hashes must be divisible by --rows-per-band")

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

    train_ids = train_df[train_id_col].astype(str).tolist()
    train_sets = [_tokens(value, args.delimiter) for value in train_df[args.set_col]]
    eval_ids = eval_df[args.id_col].astype(str).tolist()
    eval_sets = [_tokens(value, args.delimiter) for value in eval_df[args.set_col]]

    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
    for index, token_set in enumerate(train_sets):
        signature = _signature(token_set, args.num_hashes, args.seed)
        for bucket_key in _bands(signature, args.rows_per_band):
            buckets[bucket_key].append(index)

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "train_id", "similarity", "rank", "method", "candidate_source"],
        )
        writer.writeheader()
        for eval_id, token_set in zip(eval_ids, eval_sets):
            signature = _signature(token_set, args.num_hashes, args.seed)
            candidate_indices = set()
            for bucket_key in _bands(signature, args.rows_per_band):
                candidate_indices.update(buckets.get(bucket_key, []))
            candidate_source = "minhash_lsh"
            if not candidate_indices and args.fallback_exact:
                candidate_indices = set(range(len(train_sets)))
                candidate_source = "fallback_exact"

            scored = [
                (train_index, _jaccard(token_set, train_sets[train_index]))
                for train_index in candidate_indices
            ]
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
                        "method": "minhash_lsh_exact_jaccard_rerank",
                        "candidate_source": candidate_source,
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
