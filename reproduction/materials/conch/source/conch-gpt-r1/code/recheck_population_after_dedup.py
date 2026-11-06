#!/usr/bin/env python3
"""Apply the same frozen count gate after deterministic deduplication."""

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MINIMUM = 50


def main() -> None:
    state = json.loads((ROOT / "audit" / "state.json").read_text())
    if state["target_outputs_accessible"]:
        raise SystemExit("must run before target scoring")
    population_path = ROOT / "audit" / "analysis_population.json"
    population = json.loads(population_path.read_text())
    maps = json.loads((ROOT / "audit" / "label_maps.json").read_text())
    exclusions = pd.read_csv(ROOT / "results" / "dedup_exclusions.csv")
    rows = []
    for dataset in population["datasets"]:
        with (ROOT / "data" / "manifests" / f"{dataset}.sealed.tsv").open() as f:
            sealed = list(csv.DictReader(f, delimiter="\t"))
        labels = pd.DataFrame({"tile_id": [r["tile_id"] for r in sealed], "label": [maps[dataset].get(r["raw_label"]) for r in sealed]})
        frame = pd.read_csv(ROOT / "features" / f"{dataset}.axis.csv").merge(labels, on="tile_id")
        frame = frame[~frame.tile_id.isin(exclusions[exclusions.dataset == dataset].tile_id)]
        table = frame[frame.label.notna()].pivot_table(index="label", columns="axis_level", values="tile_id", aggfunc="count", fill_value=0).reindex(columns=[1,2,3,4,5], fill_value=0)
        eligible = sorted(table.index[table.min(axis=1) >= MINIMUM].tolist())
        population["datasets"][dataset]["primary_classes"] = eligible
        population["datasets"][dataset]["post_dedup_class_level_minima"] = {str(c): int(table.loc[c].min()) for c in table.index}
        for label in table.index:
            for level in table.columns:
                rows.append({"dataset": dataset, "label": label, "level": int(level), "n": int(table.loc[label, level])})
    population["deduplication_applied_before_target_outputs"] = True
    population_path.write_text(json.dumps(population, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "post_dedup_level_class_counts.csv", index=False)
    print(json.dumps(population, indent=2))


if __name__ == "__main__":
    main()
