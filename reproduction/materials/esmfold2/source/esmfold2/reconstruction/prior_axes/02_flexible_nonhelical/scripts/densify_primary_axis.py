#!/usr/bin/env python3
"""Densify the primary flexible/non-helical sequence axis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_CSV = ROOT / "spectra_ready_dataset" / "prospective_features.followup.csv"
RESULTS_CSV = ROOT / "target_model_results.csv"
OUT_DECILES = ROOT / "primary_axis_decile_curve.csv"
OUT_THRESHOLDS = ROOT / "primary_axis_threshold_sweep.csv"
OUT_SPC = ROOT / "spectral_performance_curve.csv"
OUT_JSON = ROOT / "primary_axis_densification.json"
AXIS = "flexible_nonhelical_composite_rank"
CONTROLS = ["seq_len", "msa_log10_nonquery_rows", "msa_top_nonquery_identity", "difficulty"]


def fold_for_target(target_id: str) -> int:
    return int(hashlib.sha256(target_id.encode()).hexdigest(), 16) % 3


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * 0.0


def ols_binary_coef(df: pd.DataFrame, y_col: str, stress_col: str) -> float:
    sub = df[[y_col, stress_col] + CONTROLS].copy()
    for col in sub.columns:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 30 or sub[stress_col].nunique() < 2:
        return float("nan")
    y = sub[y_col].to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(len(sub)), sub[stress_col].to_numpy(dtype=float)]
        + [zscore(sub[c]).to_numpy(dtype=float) for c in CONTROLS]
    )
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def make_decile_curve(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    work = df.copy()
    q = pd.qcut(work[AXIS], q=10, labels=False, duplicates="drop")
    work["stress_rank"] = q.astype(float)
    rows = []
    for rank, g in work.groupby("stress_rank"):
        rows.append(
            {
                "subset": subset,
                "stress_rank": float(rank),
                "n": int(len(g)),
                "feature_median": float(g[AXIS].median()),
                "ca_lddt": float(g["ca_lddt"].mean()),
                "gdt_ts": float(g["gdt_ts"].mean()),
                "ca_rmsd": float(g["ca_rmsd"].mean()),
                "baseline_ca_lddt": float(g["baseline_ca_lddt"].mean()),
                "seq_len_median": float(g["seq_len"].median()),
                "msa_log10_nonquery_rows_median": float(g["msa_log10_nonquery_rows"].median()),
                "difficulty_mean": float(g["difficulty"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("stress_rank")


def threshold_sweep(df: pd.DataFrame, subset: str) -> list[dict[str, float | int | str]]:
    rows = []
    for q in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        threshold = float(df[AXIS].quantile(q))
        work = df.copy()
        work["top_tail"] = (work[AXIS] >= threshold).astype(int)
        top = work[work["top_tail"].eq(1)]
        lower = work[work["top_tail"].eq(0)]
        if len(top) < 8 or len(lower) < 20:
            continue
        rows.append(
            {
                "subset": subset,
                "tail_quantile": q,
                "threshold": threshold,
                "n_top": int(len(top)),
                "n_lower": int(len(lower)),
                "ca_lddt_top_minus_lower": float(top["ca_lddt"].mean() - lower["ca_lddt"].mean()),
                "gdt_ts_top_minus_lower": float(top["gdt_ts"].mean() - lower["gdt_ts"].mean()),
                "ca_rmsd_top_minus_lower": float(top["ca_rmsd"].mean() - lower["ca_rmsd"].mean()),
                "baseline_ca_lddt_top_minus_lower": float(top["baseline_ca_lddt"].mean() - lower["baseline_ca_lddt"].mean()),
                "seq_len_top_median": float(top["seq_len"].median()),
                "seq_len_lower_median": float(lower["seq_len"].median()),
                "msa_top_median": float(top["msa_log10_nonquery_rows"].median()),
                "msa_lower_median": float(lower["msa_log10_nonquery_rows"].median()),
                "broad_adjusted_top_tail_coef": ols_binary_coef(work, "ca_lddt", "top_tail"),
            }
        )
    return rows


def main() -> None:
    features = pd.read_csv(FEATURE_CSV)
    results = pd.read_csv(RESULTS_CSV)
    df = features.merge(results, on="target_id", how="inner", validate="one_to_one")
    df = df[df["status"].eq("ok")].copy()
    df["discovery_fold"] = df["target_id"].map(fold_for_target)
    subsets = {
        "full_138": df,
        "confirmation_folds1_2": df[df["discovery_fold"].isin([1, 2])],
    }
    deciles = []
    thresholds = []
    for subset, work in subsets.items():
        deciles.append(make_decile_curve(work, subset))
        thresholds.extend(threshold_sweep(work, subset))
    decile_df = pd.concat(deciles, ignore_index=True)
    threshold_df = pd.DataFrame(thresholds)
    decile_df.to_csv(OUT_DECILES, index=False)
    threshold_df.to_csv(OUT_THRESHOLDS, index=False)
    # Terminal SPC uses the 7-level frozen axis from the full evaluated pool.
    full_spc = pd.read_csv(ROOT / "followup_axis_curves.csv")
    full_spc = full_spc[
        full_spc["subset"].eq("full_138")
        & full_spc["axis_id"].eq("seq_intrinsic_flexible_nonhelical_composite_7")
    ].copy()
    full_spc.to_csv(OUT_SPC, index=False)
    report = {
        "primary_axis": "seq_intrinsic_flexible_nonhelical_composite_7",
        "axis_feature": AXIS,
        "decile_curve_csv": str(OUT_DECILES.relative_to(ROOT)),
        "threshold_sweep_csv": str(OUT_THRESHOLDS.relative_to(ROOT)),
        "terminal_spc_csv": str(OUT_SPC.relative_to(ROOT)),
        "shape": "localized high-tail/threshold degradation, strongest above the top 15-20% of the flexible/non-helical composite",
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\\nDECILES")
    print(decile_df.to_string(index=False))
    print("\\nTHRESHOLDS")
    print(threshold_df.to_string(index=False))


if __name__ == "__main__":
    main()
