#!/usr/bin/env python3
"""Exact SHA-256 and dHash near-duplicate diagnostics without target labels."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["nct_crc_100k", "crc_val_7k", "stanford_crc_val_small", "kather2016_5k"]


class BKTree:
    def __init__(self):
        self.root = None

    @staticmethod
    def dist(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def add(self, value: int, tile_id: str) -> None:
        node = (value, [tile_id], {})
        if self.root is None:
            self.root = node
            return
        current = self.root
        while True:
            d = self.dist(value, current[0])
            if d == 0:
                current[1].append(tile_id)
                return
            if d not in current[2]:
                current[2][d] = node
                return
            current = current[2][d]

    def query(self, value: int, radius: int = 4):
        if self.root is None:
            return []
        out, stack = [], [self.root]
        while stack:
            node = stack.pop()
            d = self.dist(value, node[0])
            if d <= radius:
                out.extend((tile_id, d) for tile_id in node[1])
            stack.extend(child for edge, child in node[2].items() if d - radius <= edge <= d + radius)
        return out


def main() -> None:
    exact = defaultdict(list)
    manifests = {}
    for dataset in DATASETS:
        with (ROOT / "data" / "manifests" / f"{dataset}.blind.csv").open() as f:
            rows = list(csv.DictReader(f))
        manifests[dataset] = rows
        for row in rows:
            exact[row["sha256"]].append((dataset, row["tile_id"]))
    exact_groups = [v for v in exact.values() if len(v) > 1]

    ref = pd.read_csv(ROOT / "features" / "nct_crc_100k.prospective.csv", usecols=["tile_id", "dhash64"], dtype=str)
    tree = BKTree()
    for row in ref.itertuples(index=False):
        tree.add(int(row.dhash64, 16), row.tile_id)
    near_rows = []
    for dataset in ("crc_val_7k", "stanford_crc_val_small", "kather2016_5k"):
        frame = pd.read_csv(ROOT / "features" / f"{dataset}.prospective.csv", usecols=["tile_id", "dhash64"], dtype=str)
        for row in frame.itertuples(index=False):
            hits = tree.query(int(row.dhash64, 16), radius=4)
            if hits:
                best = min(hits, key=lambda x: x[1])
                near_rows.append({"target_dataset": dataset, "target_tile_id": row.tile_id, "reference_tile_id": best[0], "hamming_distance": best[1], "n_hits_le4": len(hits)})
    pd.DataFrame(near_rows, columns=["target_dataset", "target_tile_id", "reference_tile_id", "hamming_distance", "n_hits_le4"]).to_csv(ROOT / "results" / "near_duplicate_pairs.csv", index=False)
    result = {
        "exact_duplicate_groups_all_datasets": len(exact_groups),
        "exact_cross_dataset_groups": sum(len({d for d, _ in group}) > 1 for group in exact_groups),
        "dhash_cross_reference_pairs_radius_le_4": len(near_rows),
        "method": "exact file SHA-256 plus 64-bit difference-hash BK-tree search at Hamming radius 4",
    }
    (ROOT / "results" / "deduplication_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
