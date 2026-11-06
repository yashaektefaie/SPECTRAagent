#!/usr/bin/env python3
"""Recompute all packaged STATE SPC aggregates from per-example rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "spectra_ready_dataset"
ARTIFACTS = ROOT / "artifacts"


def compare(aggregate: str, source_files: list[str], level_column: str) -> dict:
    expected = pd.read_csv(ARTIFACTS / aggregate)
    source = pd.concat([pd.read_csv(READY / name, low_memory=False) for name in source_files], ignore_index=True)
    mismatches = []
    max_delta = 0.0
    for _, row in expected.iterrows():
        mask = (
            (source["spectra_subset"].astype(str) == str(row["spectra_subset"]))
            & (source["representation"].astype(str) == str(row["representation"]))
            & (source[level_column].astype(str) == str(row["level"]))
        )
        group = source.loc[mask]
        if len(group) != int(row["n_rows"]):
            mismatches.append({"level": row["level"], "field": "n_rows", "expected": int(row["n_rows"]), "actual": len(group)})
        for source_column, aggregate_column in (
            ("overlap_at_N", "overlap_at_N_mean"),
            ("de_sig_genes_recall", "de_sig_genes_recall_mean"),
            ("pearson_delta", "pearson_delta_mean"),
        ):
            expected_value = float(row[aggregate_column])
            actual_value = float(pd.to_numeric(group[source_column], errors="coerce").mean())
            delta = abs(expected_value - actual_value) if np.isfinite(expected_value) and np.isfinite(actual_value) else 0.0
            max_delta = max(max_delta, delta)
            if delta > 1e-12:
                mismatches.append(
                    {
                        "level": row["level"],
                        "field": aggregate_column,
                        "expected": expected_value,
                        "actual": actual_value,
                        "delta": delta,
                    }
                )
    return {
        "aggregate": aggregate,
        "source_files": source_files,
        "aggregate_rows": len(expected),
        "max_abs_metric_delta": max_delta,
        "mismatches": len(mismatches),
        "passes": not mismatches,
        "examples": mismatches[:3],
    }


def main() -> None:
    checks = [
        compare("spc_parse_fewshot_ligand_family_support.csv", ["parse_fewshot_ligand_family_support_examples.csv"], "parse_family_support_level"),
        compare("spc_replogle_best_refined_axis.csv", ["state_replogle_reactome_pathway_log_train_support_v2_examples.csv"], "reactome_pathway_train_support_log_mean_level"),
        compare("spc_replogle_fewshot_reactome_support.csv", ["replogle_fewshot_reactome_support_examples.csv"], "reactome_support_level"),
        compare("spc_replogle_zeroshot_reactome_support_confirmation.csv", ["replogle_zeroshot_reactome_support_confirmation_examples.csv"], "zeroshot_norm_support_level_from_fewshot_freeze"),
        compare("spc_state_replogle_hgnc_group_log_train_support_v2.csv", ["state_replogle_hgnc_group_log_train_support_v2_examples.csv"], "hgnc_group_train_support_log_mean_level"),
        compare("spc_state_replogle_reactome_pathway_log_train_support_v2.csv", ["state_replogle_reactome_pathway_log_train_support_v2_examples.csv"], "reactome_pathway_train_support_log_mean_level"),
        compare("spc_state_replogle_reactome_pathway_mean_train_support_v2.csv", ["state_replogle_reactome_pathway_mean_train_support_v2_examples.csv"], "reactome_pathway_train_support_mean_level"),
        compare("spc_state_replogle_reactome_pathway_min_train_support_v2.csv", ["state_replogle_reactome_pathway_min_train_support_v2_examples.csv"], "reactome_pathway_train_support_min_level"),
        compare("spc_state_replogle_symbol_family_support_control.csv", ["state_replogle_symbol_family_support_control_examples.csv"], "symbol_family_candidate_train_count_level"),
    ]
    for collection in ("go_bp", "go_cc", "go_mf", "hgnc_gene_group", "kegg", "reactome", "wikipathways"):
        checks.append(
            compare(
                f"spc_multi_pathway_{collection}.csv",
                [f"multi_pathway_{collection}_fewshot_examples.csv", f"multi_pathway_{collection}_zeroshot_examples.csv"],
                "support_level",
            )
        )
    result = {
        "curve_tables_checked": len(checks),
        "all_pass": all(check["passes"] for check in checks),
        "max_abs_metric_delta": max(check["max_abs_metric_delta"] for check in checks),
        "checks": checks,
    }
    print(json.dumps(result, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
