#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "spectra_ready_dataset" / "candidate_features.csv"
RESULTS = ROOT / "target_model_results.csv"
RESULT_COLUMNS = [
    "target_id",
    "status",
    "error",
    "wall_sec",
    "ptm",
    "plddt_mean",
    "plddt_ca_mean",
    "alignment_identity",
    "alignment_coverage",
    "aligned_ca",
    "ca_lddt",
    "ca_rmsd",
    "gdt_ts",
    "ca_median_error",
    "ca_p90_error",
    "baseline_ca_lddt",
    "baseline_gdt_ts",
    "baseline_ca_rmsd",
]


def read_results(path: Path) -> pd.DataFrame:
    first = path.read_text().splitlines()[0] if path.exists() else ""
    if first.startswith("target_id,"):
        return pd.read_csv(path)
    return pd.read_csv(path, names=RESULT_COLUMNS, skip_blank_lines=True)


def qbin(df: pd.DataFrame, col: str, n: int = 5, reverse: bool = False) -> pd.Series:
    ranks = df[col].rank(method="first")
    bins = pd.qcut(ranks, q=n, labels=False, duplicates="drop")
    if reverse:
        bins = bins.max() - bins
    return bins.astype(int)


def summarize_axis(df: pd.DataFrame, axis_name: str, similarity_col: str, level_col: str) -> dict:
    rows = []
    for level, g in df.groupby(level_col):
        rows.append(
            {
                "axis": axis_name,
                "level": int(level),
                "n": int(len(g)),
                "similarity_mean": float(g[similarity_col].mean()),
                "similarity_min": float(g[similarity_col].min()),
                "similarity_max": float(g[similarity_col].max()),
                "seq_len_mean": float(g["seq_len"].mean()),
                "difficulty_mean": float(g["difficulty"].mean()),
                "resolution_mean": float(g["resolution"].mean()),
                "target_ca_lddt_mean": float(g["ca_lddt"].mean()),
                "target_ca_lddt_median": float(g["ca_lddt"].median()),
                "target_gdt_ts_mean": float(g["gdt_ts"].mean()),
                "target_ca_rmsd_median": float(g["ca_rmsd"].median()),
                "baseline_ca_lddt_mean": float(g["baseline_ca_lddt"].mean()),
                "baseline_gdt_ts_mean": float(g["baseline_gdt_ts"].mean()),
                "status_ok_rate": float((g["status"] == "ok").mean()),
                "wall_sec_mean": float(g["wall_sec"].mean()),
            }
        )
    spc = pd.DataFrame(rows).sort_values("level")
    sim_decreases = bool(spc["similarity_mean"].is_monotonic_decreasing)
    perf = spc["target_ca_lddt_mean"].to_numpy()
    similarity = spc["similarity_mean"].to_numpy()
    rho = spearmanr(similarity, perf, nan_policy="omit")
    return {
        "axis": axis_name,
        "similarity_col": similarity_col,
        "level_col": level_col,
        "n_total": int(len(df)),
        "n_levels": int(spc["level"].nunique()),
        "similarity_decreases_with_level": sim_decreases,
        "target_ca_lddt_range": [float(np.nanmin(perf)), float(np.nanmax(perf))],
        "target_ca_lddt_drop_high_to_low_similarity": float(perf[0] - perf[-1]),
        "spearman_similarity_vs_ca_lddt_rho": float(rho.correlation) if rho.correlation == rho.correlation else None,
        "spearman_similarity_vs_ca_lddt_p": float(rho.pvalue) if rho.pvalue == rho.pvalue else None,
        "spc": spc,
    }


def main() -> None:
    feat = pd.read_csv(FEATURES)
    res = read_results(RESULTS)
    df = feat.merge(res, on="target_id", how="inner")
    df = df[(df["status"] == "ok") & df["ca_lddt"].notna()].copy()

    axes = []
    # Frozen before target scoring: higher MSA depth means greater evolutionary-family support.
    df["axis_msa_depth_level"] = qbin(df, "msa_log10_nonquery_rows", n=5, reverse=True)
    axes.append(summarize_axis(df, "frozen_msa_depth", "msa_log10_nonquery_rows", "axis_msa_depth_level"))

    # Predeclared secondary boundary axes, scored only after the frozen primary axis.
    df["axis_length_level"] = qbin(df, "seq_len", n=5, reverse=False)
    axes.append(summarize_axis(df, "predeclared_length_boundary", "seq_len", "axis_length_level"))
    df["axis_difficulty_level"] = df["difficulty"].astype(int)
    axes.append(summarize_axis(df, "predeclared_cameo_difficulty", "difficulty", "axis_difficulty_level"))
    df["axis_msa_top_identity_level"] = qbin(df, "msa_top_nonquery_identity", n=5, reverse=True)
    axes.append(summarize_axis(df, "predeclared_top_homolog_identity", "msa_top_nonquery_identity", "axis_msa_top_identity_level"))

    out_rows = []
    ledger = []
    for item in axes:
        spc = item.pop("spc")
        out_rows.append(spc)
        decision = "negative_or_non_explanatory"
        drop = item["target_ca_lddt_drop_high_to_low_similarity"]
        if item["n_levels"] >= 3 and item["similarity_decreases_with_level"] and drop >= 0.08:
            decision = "promising_degradation"
        if item["axis"] == "predeclared_length_boundary":
            # For length, level increases as length increases rather than similarity decreasing.
            # Treat as a boundary curve if performance worsens by level.
            if drop >= 0.08 or spc["target_ca_lddt_mean"].iloc[0] - spc["target_ca_lddt_mean"].iloc[-1] >= 0.08:
                decision = "promising_boundary"
        ledger.append({**item, "decision": decision})
    curve = pd.concat(out_rows, ignore_index=True)
    curve.to_csv(ROOT / "spectral_performance_curve.csv", index=False)

    for item in ledger:
        for k, v in list(item.items()):
            if isinstance(v, np.generic):
                item[k] = v.item()
    (ROOT / "axis_ledger.json").write_text(json.dumps(ledger, indent=2))

    state = {
        "dataset": str(FEATURES),
        "target_results": str(RESULTS),
        "evaluated_ok": int(len(df)),
        "primary_frozen_axis": "frozen_msa_depth",
        "axis_statuses": [{"axis": x["axis"], "decision": x["decision"], "drop": x["target_ca_lddt_drop_high_to_low_similarity"]} for x in ledger],
        "closure_gate": "pending_auditor_review",
    }
    (ROOT / "spectra_loop_state.json").write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
