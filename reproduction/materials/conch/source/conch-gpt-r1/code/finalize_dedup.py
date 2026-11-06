#!/usr/bin/env python3
"""Materialize the frozen deduplication exclusions from completed diagnostics."""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["nct_crc_100k", "crc_val_7k", "stanford_crc_val_small", "kather2016_5k"]


def main() -> None:
    by_sha = defaultdict(list)
    for dataset in DATASETS:
        with (ROOT / "data" / "manifests" / f"{dataset}.blind.csv").open() as f:
            for row in csv.DictReader(f):
                by_sha[row["sha256"]].append((dataset, row["tile_id"]))
    group_rows, exclusions = [], []
    for digest, members in by_sha.items():
        if len(members) < 2:
            continue
        gid = hashlib.sha256((digest + repr(sorted(members))).encode()).hexdigest()[:16]
        for dataset, tile_id in sorted(members):
            group_rows.append({"group_id": gid, "sha256": digest, "dataset": dataset, "tile_id": tile_id})
        by_dataset = defaultdict(list)
        for dataset, tile_id in members:
            by_dataset[dataset].append(tile_id)
        for dataset, ids in by_dataset.items():
            if dataset != "nct_crc_100k" and len(ids) > 1:
                for tile_id in sorted(ids)[1:]:
                    exclusions.append({"dataset": dataset, "tile_id": tile_id, "reason": "within_cohort_exact_duplicate_keep_lexicographic_first"})
    near = pd.read_csv(ROOT / "results" / "near_duplicate_pairs.csv")
    for row in near.itertuples(index=False):
        exclusions.append({"dataset": row.target_dataset, "tile_id": row.target_tile_id, "reason": f"reference_dhash_hamming_{row.hamming_distance}"})
    pd.DataFrame(group_rows, columns=["group_id", "sha256", "dataset", "tile_id"]).to_csv(ROOT / "results" / "exact_duplicate_groups.csv", index=False)
    pd.DataFrame(exclusions, columns=["dataset", "tile_id", "reason"]).drop_duplicates(["dataset", "tile_id"]).to_csv(ROOT / "results" / "dedup_exclusions.csv", index=False)
    print(pd.DataFrame(exclusions).groupby(["dataset", "reason"]).size())


if __name__ == "__main__":
    main()
