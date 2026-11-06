#!/usr/bin/env python3
"""Validate packaged ESMFold2 curve inputs without using upstream run roots."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_global_curves() -> list[dict]:
    audit = load_module("global_hypothesis_audit", ROOT / "scripts" / "global_hypothesis_audit.py")
    configs = [
        (
            "all_scored",
            ROOT / "data" / "all_esmfold2_scored_evaluations.csv",
            ROOT / "tables" / "global_hypothesis_level_curves_all_evaluations.csv",
        ),
        (
            "primary_exact_sequence",
            ROOT / "data" / "primary_exact_sequence_dedup.csv",
            ROOT / "tables" / "global_hypothesis_level_curves_primary_sequence.csv",
        ),
    ]
    checks = []
    for label, data_path, curve_path in configs:
        source = pd.read_csv(data_path, low_memory=False)
        computed = audit.evaluate_hypotheses(source, label)[1]
        stored = pd.read_csv(curve_path, low_memory=False)
        keys = ["axis", "level"]
        values = [
            "n",
            "stress_mean",
            "ca_lddt_mean",
            "ca_lddt_median",
            "gdt_ts_mean",
            "ca_rmsd_median",
            "baseline_ca_lddt_mean",
        ]
        stored = stored.sort_values(keys).reset_index(drop=True)
        computed = computed.sort_values(keys).reset_index(drop=True)
        same_keys = stored[keys].astype(str).equals(computed[keys].astype(str))
        deltas = {}
        for column in values:
            left = pd.to_numeric(stored[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(computed[column], errors="coerce").to_numpy(float)
            finite = np.isfinite(left) & np.isfinite(right)
            deltas[column] = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        checks.append(
            {
                "table": curve_path.relative_to(ROOT).as_posix(),
                "stored_rows": len(stored),
                "computed_rows": len(computed),
                "same_keys": same_keys,
                "max_abs_deltas": deltas,
                "passes": same_keys and len(stored) == len(computed) and max(deltas.values()) < 1e-12,
            }
        )
    return checks


def main() -> None:
    redraw = load_module("redraw_reported_curves", ROOT / "scripts" / "redraw_reported_curves.py")
    curve_tables = redraw.curve_tables()
    global_checks = compare_global_curves()
    result = {
        "packaged_curve_tables": len(curve_tables),
        "global_curve_tables_recomputed": len(global_checks),
        "all_pass": bool(curve_tables) and all(check["passes"] for check in global_checks),
        "global_checks": global_checks,
    }
    print(json.dumps(result, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
