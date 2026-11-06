#!/usr/bin/env python3
"""Outcome-blind mechanical application of the frozen class-count gate."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MINIMUM = 50


def main() -> None:
    state = json.loads((ROOT / "audit" / "state.json").read_text())
    if state["target_outputs_accessible"]:
        raise SystemExit("population gate must be finalized before CONCH outputs")
    counts = pd.read_csv(ROOT / "results" / "level_class_counts.csv")
    result = {
        "created_before_target_outputs": True,
        "rule": "primary classes must be prospectively mapped and have at least 50 tiles in every frozen level",
        "minimum_per_class_level": MINIMUM,
        "datasets": {},
    }
    for dataset, frame in counts[counts.mapped_label.notna()].groupby("dataset"):
        table = frame.pivot_table(index="mapped_label", columns="level", values="n", aggfunc="sum", fill_value=0)
        eligible = sorted(table.index[(table.reindex(columns=[1, 2, 3, 4, 5], fill_value=0).min(axis=1) >= MINIMUM)].tolist())
        result["datasets"][dataset] = {
            "primary_classes": eligible,
            "secondary_classes": sorted(table.index.tolist()),
            "class_level_minima": {str(c): int(table.loc[c].min()) for c in table.index},
        }
    output = ROOT / "audit" / "analysis_population.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
