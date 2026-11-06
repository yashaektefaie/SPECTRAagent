#!/usr/bin/env python3
"""Post-freeze class/level and recoverable source-cluster diagnostics."""

import csv
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["crc_val_7k", "stanford_crc_val_small", "kather2016_5k"]


def main() -> None:
    state = json.loads((ROOT / "audit" / "state.json").read_text())
    if not state["target_labels_accessible"]:
        raise SystemExit("controller guard: post-freeze only")
    maps = json.loads((ROOT / "audit" / "label_maps.json").read_text())
    out, clusters = [], []
    for dataset in DATASETS:
        with (ROOT / "data" / "manifests" / f"{dataset}.sealed.tsv").open() as f:
            sealed = list(csv.DictReader(f, delimiter="\t"))
        labels = pd.DataFrame({
            "tile_id": [r["tile_id"] for r in sealed],
            "raw_label": [r["raw_label"] for r in sealed],
            "label": [maps[dataset].get(r["raw_label"]) for r in sealed],
        })
        axis = pd.read_csv(ROOT / "features" / f"{dataset}.axis.csv")
        frame = axis.merge(labels, on="tile_id")
        for keys, group in frame.groupby(["raw_label", "label", "axis_level"], dropna=False):
            out.append({"dataset": dataset, "raw_label": keys[0], "mapped_label": keys[1], "level": int(keys[2]), "n": len(group)})
        if dataset == "kather2016_5k":
            ids = []
            for r in sealed:
                match = re.search(r"CRC-Prim-HE-[^_/]+", r["archive_member"])
                ids.append(match.group(0) if match else "unresolved")
            clusters.append({"dataset": dataset, "recoverable_cluster": "slide-like filename token", "n_unique": len(set(ids)), "notes": "available for cluster sensitivity; not a verified patient id"})
        else:
            clusters.append({"dataset": dataset, "recoverable_cluster": "none", "n_unique": None, "notes": "published cohort count is known, but the released tile filename does not map tiles to patient/WSI"})
    pd.DataFrame(out).to_csv(ROOT / "results" / "level_class_counts.csv", index=False)
    pd.DataFrame(clusters).to_csv(ROOT / "results" / "cluster_diagnostics.csv", index=False)
    print(pd.DataFrame(out).groupby(["dataset", "level"]).n.sum())
    print(pd.DataFrame(clusters).to_string(index=False))


if __name__ == "__main__":
    main()
