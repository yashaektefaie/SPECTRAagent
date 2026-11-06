#!/usr/bin/env python3
"""Cross-protocol confirmation for the refined Replogle Reactome support axis."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import state_spectra_loop as base  # noqa: E402


ROOT = Path("./spectra_runs/state_generalizability_continued_20260603")
ART = ROOT / "artifacts"
READY = ROOT / "spectra_ready_dataset"

PRIMARY_METRIC = "overlap_at_N"


def safe_mean(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def support_stats(feature_set: set[str], feature_index: dict[str, set[str]], train_n: int) -> dict[str, float]:
    if not feature_set or train_n <= 0:
        return {"log_mean": 0.0, "norm_log_mean": 0.0, "supported_fraction": 0.0}
    counts = np.array([len(feature_index.get(f, set())) for f in feature_set], dtype=float)
    log_mean = float(np.mean(np.log1p(counts)))
    return {
        "log_mean": log_mean,
        "norm_log_mean": log_mean / float(np.log1p(train_n)),
        "supported_fraction": float(np.mean(counts > 0)),
    }


def summarize(df: pd.DataFrame, level_col: str, sim_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = {"L0_no_annotation": 0, "L1_low": 1, "L2_mid": 2, "L3_high": 3, "L4_very_high": 4}
    rows = []
    for (subset, rep, level), sub in df.groupby(["spectra_subset", "representation", level_col]):
        rows.append(
            {
                "spectra_subset": subset,
                "representation": rep,
                "level": level,
                "level_order": order.get(str(level), -1),
                "n_rows": int(len(sub)),
                "n_unique_contexts": int(sub["context"].nunique()),
                "n_unique_perturbations": int(sub["gene_symbol_canonical"].nunique()),
                "similarity_mean": safe_mean(sub[sim_col]),
                "overlap_at_N_mean": safe_mean(sub["overlap_at_N"]),
                "de_sig_genes_recall_mean": safe_mean(sub["de_sig_genes_recall"]),
                "pearson_delta_mean": safe_mean(sub["pearson_delta"]),
            }
        )
    levels = pd.DataFrame(rows).sort_values(["spectra_subset", "representation", "level_order"])
    trend_rows = []
    for (subset, rep), sub in levels.groupby(["spectra_subset", "representation"]):
        valid = sub[sub["level_order"] >= 0].sort_values("level_order")
        if len(valid) < 2:
            continue
        low = valid.iloc[0]
        high = valid.iloc[-1]
        trend_rows.append(
            {
                "spectra_subset": subset,
                "representation": rep,
                "n_levels": int(len(valid)),
                "n_rows": int(valid["n_rows"].sum()),
                "similarity_monotone_increasing": bool(valid["similarity_mean"].is_monotonic_increasing),
                "spearman_level_vs_overlap_at_N": float(valid["level_order"].corr(valid["overlap_at_N_mean"], method="spearman")),
                "low_level": low["level"],
                "high_level": high["level"],
                "low_overlap_at_N_mean": float(low["overlap_at_N_mean"]),
                "high_overlap_at_N_mean": float(high["overlap_at_N_mean"]),
                "high_minus_low_overlap_at_N": float(high["overlap_at_N_mean"] - low["overlap_at_N_mean"]),
            }
        )
    return levels, pd.DataFrame(trend_rows)


def main() -> None:
    df = base.load_results()
    hgnc, alias = base.load_hgnc()
    reactome = base.load_reactome(hgnc, alias)

    fewshot = pd.read_csv(READY / "state_replogle_reactome_pathway_log_train_support_v2_examples.csv", low_memory=False)
    fewshot["norm_support"] = fewshot["reactome_pathway_train_support_log_mean"] / np.log1p(
        pd.to_numeric(fewshot["context_train_gene_count"], errors="coerce").replace(0, np.nan)
    )
    disc = fewshot[
        (fewshot["spectra_subset"] == "discovery_hepg2_k562")
        & (fewshot["reactome_pathway_count"] > 0)
        & fewshot["norm_support"].notna()
    ]
    cutpoints = [float(x) for x in disc["norm_support"].quantile([0.25, 0.50, 0.75]).tolist()]
    labels = ["L1_low", "L2_mid", "L3_high", "L4_very_high"]

    z = df[(df["dataset_family"] == "Replogle-Nadig") & (df["protocol"] == "zeroshot")].copy()
    z["gene_symbol_canonical"] = z["gene_symbol"].map(lambda x: alias.get(base.norm_symbol(x), base.norm_symbol(x)))
    all_by_context = {
        str(ctx): {alias.get(base.norm_symbol(g), base.norm_symbol(g)) for g in sub["gene_symbol"].dropna() if base.norm_symbol(g) != "NON-TARGETING"}
        for ctx, sub in z.groupby("context")
    }
    all_contexts = set(all_by_context)
    rows = []
    for context, sub in z.groupby("context"):
        train_contexts = all_contexts - {str(context)}
        train_genes = set().union(*(all_by_context[c] for c in train_contexts))
        idx = base.build_feature_index(reactome, train_genes)
        for row_index, row in sub.iterrows():
            gene = row["gene_symbol_canonical"]
            feats = reactome.get(gene, set())
            stats = support_stats(feats, idx, len(train_genes))
            rows.append(
                {
                    "row_index": row_index,
                    "zeroshot_train_contexts": ",".join(sorted(train_contexts)),
                    "zeroshot_train_gene_count": len(train_genes),
                    "reactome_pathway_count": len(feats),
                    "zeroshot_reactome_support_log_mean": stats["log_mean"],
                    "zeroshot_reactome_support_norm_log_mean": stats["norm_log_mean"],
                    "zeroshot_reactome_supported_fraction": stats["supported_fraction"],
                }
            )
    feat = pd.DataFrame(rows).set_index("row_index")
    z = z.join(feat)
    z = z[z["gene_symbol_canonical"] != "NON-TARGETING"].copy()

    def assign(v: object) -> str:
        if pd.isna(v):
            return "L0_no_annotation"
        fv = float(v)
        for cp, lab in zip(cutpoints, labels):
            if fv <= cp:
                return lab
        return "L4_very_high"

    z["zeroshot_norm_support_level_from_fewshot_freeze"] = z["zeroshot_reactome_support_norm_log_mean"].map(assign)
    z.loc[z["reactome_pathway_count"] <= 0, "zeroshot_norm_support_level_from_fewshot_freeze"] = "L0_no_annotation"
    z["spectra_subset"] = "zeroshot_cross_protocol_confirmation"
    z.to_csv(READY / "replogle_zeroshot_reactome_support_confirmation_examples.csv", index=False)
    freeze = {
        "axis_id": "STATE_REPLOGLE_REACTOME_SUPPORT_CROSS_PROTOCOL_CONFIRMATION",
        "status": "applied_fewshot_discovery_normalized_cutpoints_to_zeroshot_protocol",
        "similarity": "Mean log(1 + training genes sharing each Reactome pathway), normalized by train-set size",
        "fewshot_discovery_contexts_for_cutpoints": ["hepg2", "k562"],
        "zeroshot_confirmation_contexts": sorted(all_contexts),
        "cutpoints_from_fewshot_discovery_norm_support": cutpoints,
        "target_metric": PRIMARY_METRIC,
    }
    (ART / "replogle_zeroshot_reactome_support_confirmation_freeze.json").write_text(json.dumps(freeze, indent=2))
    levels, trends = summarize(
        z,
        "zeroshot_norm_support_level_from_fewshot_freeze",
        "zeroshot_reactome_support_norm_log_mean",
    )
    levels.to_csv(ART / "spc_replogle_zeroshot_reactome_support_confirmation.csv", index=False)
    trends.to_csv(ART / "spc_replogle_zeroshot_reactome_support_confirmation_trends.csv", index=False)
    decision = {
        "axis_id": freeze["axis_id"],
        "decision": "supports_axis" if (trends["high_minus_low_overlap_at_N"] > 0).all() else "does_not_support_axis",
        "note": "This is a harder cross-protocol check; context zeroshot performance can be dominated by cell-context novelty.",
        "trends": trends.to_dict(orient="records"),
    }
    (ART / "replogle_zeroshot_reactome_support_confirmation_decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
