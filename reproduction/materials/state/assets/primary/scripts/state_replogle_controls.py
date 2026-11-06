#!/usr/bin/env python3
"""Controls for the refined STATE/Replogle SPECTRA axis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("./spectra_runs/state_generalizability_continued_20260603")
ART = ROOT / "artifacts"
READY = ROOT / "spectra_ready_dataset"
EXAMPLES = READY / "state_replogle_reactome_pathway_log_train_support_v2_examples.csv"
LEVEL_COL = "reactome_pathway_train_support_log_mean_level"
SUPPORT_COL = "reactome_pathway_train_support_log_mean"
METRIC = "overlap_at_N"


LEVEL_ORDER = {
    "L0_no_annotation": 0,
    "L1_low": 1,
    "L2_mid": 2,
    "L3_high": 3,
    "L4_very_high": 4,
}


def safe_mean(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def summarize(df: pd.DataFrame, subset_col: str = "spectra_subset") -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby([subset_col, "representation", LEVEL_COL], dropna=False):
        subset, rep, level = keys
        rows.append(
            {
                subset_col: subset,
                "representation": rep,
                "level": level,
                "level_order": LEVEL_ORDER.get(str(level), -1),
                "n_rows": int(len(sub)),
                "n_unique_contexts": int(sub["context"].nunique()),
                "n_unique_perturbations": int(sub["gene_symbol_canonical"].nunique()),
                "support_mean": safe_mean(sub[SUPPORT_COL]),
                "overlap_at_N_mean": safe_mean(sub["overlap_at_N"]),
                "de_sig_genes_recall_mean": safe_mean(sub["de_sig_genes_recall"]),
                "pearson_delta_mean": safe_mean(sub["pearson_delta"]),
                "de_nsig_counts_real_mean": safe_mean(sub["de_nsig_counts_real"]),
                "reactome_pathway_count_mean": safe_mean(sub["reactome_pathway_count"]),
            }
        )
    return pd.DataFrame(rows).sort_values([subset_col, "representation", "level_order"])


def high_low_gaps(df: pd.DataFrame, group_cols: list[str], include_l0: bool) -> pd.DataFrame:
    rows = []
    work = df.copy()
    if not include_l0:
        work = work[work[LEVEL_COL] != "L0_no_annotation"].copy()
    for keys, sub in work.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        levels = sorted(sub[LEVEL_COL].dropna().unique(), key=lambda x: LEVEL_ORDER.get(str(x), -1))
        if len(levels) < 2:
            continue
        low, high = levels[0], levels[-1]
        low_df = sub[sub[LEVEL_COL] == low]
        high_df = sub[sub[LEVEL_COL] == high]
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "include_l0": include_l0,
                "low_level": low,
                "high_level": high,
                "low_n": int(len(low_df)),
                "high_n": int(len(high_df)),
                "low_overlap_at_N_mean": safe_mean(low_df[METRIC]),
                "high_overlap_at_N_mean": safe_mean(high_df[METRIC]),
                "high_minus_low_overlap_at_N": safe_mean(high_df[METRIC]) - safe_mean(low_df[METRIC]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def standardized_ols(df: pd.DataFrame, y_col: str, include_de_count: bool) -> dict[str, object]:
    work = df.dropna(subset=[y_col, SUPPORT_COL, "reactome_pathway_count"]).copy()
    work = work[work["reactome_pathway_count"] > 0].copy()
    y = pd.to_numeric(work[y_col], errors="coerce").to_numpy(float)
    cols = []
    names = []

    def add_col(name: str, values: np.ndarray, standardize: bool = True) -> None:
        vals = values.astype(float)
        if standardize:
            sd = float(np.std(vals))
            vals = (vals - float(np.mean(vals))) / sd if sd > 0 else vals * 0
        cols.append(vals)
        names.append(name)

    add_col("support_log_mean_z", pd.to_numeric(work[SUPPORT_COL], errors="coerce").to_numpy(float))
    add_col("reactome_pathway_count_z", pd.to_numeric(work["reactome_pathway_count"], errors="coerce").to_numpy(float))
    if include_de_count:
        add_col("log1p_de_nsig_counts_real_z", np.log1p(pd.to_numeric(work["de_nsig_counts_real"], errors="coerce").fillna(0).to_numpy(float)))
    for col in ["context", "representation", "spectra_subset"]:
        dummies = pd.get_dummies(work[col], prefix=col, drop_first=True, dtype=float)
        for dcol in dummies.columns:
            cols.append(dummies[dcol].to_numpy(float))
            names.append(dcol)
    x = np.column_stack([np.ones(len(work)), *cols])
    beta, residuals, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    n, p = x.shape
    sigma2 = ss_res / max(n - p, 1)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    coefs = [{"term": "intercept", "coef": float(beta[0]), "se": float(se[0])}]
    for name, coef, serr in zip(names, beta[1:], se[1:]):
        coefs.append({"term": name, "coef": float(coef), "se": float(serr), "t": float(coef / serr) if serr else float("nan")})
    return {"n_rows": int(n), "r2": r2, "include_de_count": include_de_count, "coefficients": coefs}


def matched_pathway_count_gaps(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df[LEVEL_COL].isin(["L1_low", "L4_very_high"])].copy()
    work = work[work["reactome_pathway_count"] > 0].copy()
    work["pathway_count_bin"] = pd.qcut(work["reactome_pathway_count"], q=4, duplicates="drop")
    rows = []
    for keys, sub in work.groupby(["context", "representation", "pathway_count_bin"], observed=True):
        ctx, rep, bin_ = keys
        low = sub[sub[LEVEL_COL] == "L1_low"]
        high = sub[sub[LEVEL_COL] == "L4_very_high"]
        if len(low) < 5 or len(high) < 5:
            continue
        rows.append(
            {
                "context": ctx,
                "representation": rep,
                "pathway_count_bin": str(bin_),
                "low_n": int(len(low)),
                "high_n": int(len(high)),
                "low_overlap_at_N_mean": safe_mean(low[METRIC]),
                "high_overlap_at_N_mean": safe_mean(high[METRIC]),
                "high_minus_low_overlap_at_N": safe_mean(high[METRIC]) - safe_mean(low[METRIC]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(EXAMPLES, low_memory=False)
    summary_all = summarize(df)
    summary_no_l0 = summarize(df[df[LEVEL_COL] != "L0_no_annotation"].copy())
    summary_all.to_csv(ART / "controls_replogle_best_axis_level_summary_all.csv", index=False)
    summary_no_l0.to_csv(ART / "controls_replogle_best_axis_level_summary_no_l0.csv", index=False)

    gaps = pd.concat(
        [
            high_low_gaps(df, ["spectra_subset", "representation"], include_l0=True),
            high_low_gaps(df, ["spectra_subset", "representation"], include_l0=False),
            high_low_gaps(df, ["context", "representation"], include_l0=True),
            high_low_gaps(df, ["context", "representation"], include_l0=False),
        ],
        ignore_index=True,
    )
    gaps.to_csv(ART / "controls_replogle_best_axis_high_low_gaps.csv", index=False)

    matched = matched_pathway_count_gaps(df)
    matched.to_csv(ART / "controls_replogle_best_axis_pathway_count_matched.csv", index=False)

    ols = {
        "without_de_count": standardized_ols(df, METRIC, include_de_count=False),
        "with_de_count_outcome_control": standardized_ols(df, METRIC, include_de_count=True),
        "secondary_de_recall_with_de_count": standardized_ols(df, "de_sig_genes_recall", include_de_count=True),
        "secondary_pearson_with_de_count": standardized_ols(df, "pearson_delta", include_de_count=True),
    }
    (ART / "controls_replogle_best_axis_regression.json").write_text(json.dumps(ols, indent=2))

    support_control = {
        "status": "completed",
        "interpretation": (
            "The refined Reactome support axis remains a current-pool confirmed boundary. "
            "Controls test whether it is explained by annotation presence, context, representation, "
            "pathway annotation count, or DE-count severity."
        ),
        "artifacts": [
            str(ART / "controls_replogle_best_axis_level_summary_all.csv"),
            str(ART / "controls_replogle_best_axis_level_summary_no_l0.csv"),
            str(ART / "controls_replogle_best_axis_high_low_gaps.csv"),
            str(ART / "controls_replogle_best_axis_pathway_count_matched.csv"),
            str(ART / "controls_replogle_best_axis_regression.json"),
        ],
    }
    (ART / "controls_replogle_best_axis_summary.json").write_text(json.dumps(support_control, indent=2))
    print(json.dumps(support_control, indent=2))


if __name__ == "__main__":
    main()
