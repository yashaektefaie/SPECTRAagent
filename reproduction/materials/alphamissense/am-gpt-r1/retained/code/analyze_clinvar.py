#!/usr/bin/env python3
"""Analyze a frozen ClinVar SPC with gene-cluster uncertainty and controls."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260714
LEVELS = [0, 1, 2, 3]


def check_access(partition: str) -> None:
    state = json.loads((ROOT / "controller/state.json").read_text())
    if partition == "discovery" and not state.get("target_outcomes_accessible"):
        raise SystemExit("Discovery outcomes are sealed")
    if partition == "confirmation" and not state.get("confirmation_accessible"):
        raise SystemExit("Confirmation outcomes are sealed")


def weighted_auc_sorted(labels: np.ndarray, scores: np.ndarray,
                        row_weights: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    w = row_weights[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    pos = np.add.reduceat(w * (y == 1), starts)
    neg = np.add.reduceat(w * (y == 0), starts)
    p, n = pos.sum(), neg.sum()
    if p <= 0 or n <= 0:
        return np.nan
    neg_before = np.cumsum(neg) - neg
    return float(np.sum(pos * (neg_before + 0.5 * neg)) / (p * n))


def ece(y: np.ndarray, score: np.ndarray) -> float:
    bins = np.minimum((score * 10).astype(int), 9)
    value = 0.0
    for b in range(10):
        keep = bins == b
        if keep.any():
            value += keep.mean() * abs(y[keep].mean() - score[keep].mean())
    return float(value)


def calibration(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    z = np.log(np.clip(score, 1e-6, 1 - 1e-6) /
               np.clip(1 - score, 1e-6, 1 - 1e-6))
    try:
        fit = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit()
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        return np.nan, np.nan


def level_metrics(g: pd.DataFrame) -> dict:
    y = g.label.to_numpy(int)
    s = g.am_pathogenicity.to_numpy(float)
    ref = g.blosum_pathogenicity.to_numpy(float)
    intercept, slope = calibration(y, s)
    called = (s < 0.34) | (s > 0.564)
    correct_call = ((s < 0.34) & (y == 0)) | ((s > 0.564) & (y == 1))
    return {
        "axis_level": int(g.axis_level.iloc[0]), "axis_name": g.axis_name.iloc[0],
        "rows": len(g), "genes": g.gene.nunique(), "proteins": g.uniprot_id.nunique(),
        "pathogenic": int(y.sum()), "benign": int((1 - y).sum()),
        "prevalence": y.mean(), "median_d_splice_bp": g.d_splice_bp.median(),
        "q25_d_splice_bp": g.d_splice_bp.quantile(.25),
        "q75_d_splice_bp": g.d_splice_bp.quantile(.75),
        "min_d_splice_bp": g.d_splice_bp.min(), "max_d_splice_bp": g.d_splice_bp.max(),
        "am_auroc": roc_auc_score(y, s),
        "blosum_auroc": roc_auc_score(y, ref),
        "am_auprc": average_precision_score(y, s),
        "brier": np.mean((s - y) ** 2), "ece_10": ece(y, s),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "sensitivity_gt_0564": np.mean(s[y == 1] > 0.564),
        "specificity_at_0564": np.mean(s[y == 0] <= 0.564),
        "sensitivity_at_034": np.mean(s[y == 1] >= 0.34),
        "specificity_lt_034": np.mean(s[y == 0] < 0.34),
        "ambiguous_fraction": np.mean((s >= 0.34) & (s <= 0.564)),
        "called_fraction": called.mean(),
        "called_accuracy": correct_call.sum() / called.sum() if called.any() else np.nan,
    }


def cluster_bootstrap(df: pd.DataFrame, reps: int = 2000) -> pd.DataFrame:
    genes = sorted(df.gene.unique())
    gene_to_i = {g: i for i, g in enumerate(genes)}
    arrays = {}
    for lv in LEVELS:
        g = df[df.axis_level == lv]
        arrays[lv] = {
            "y": g.label.to_numpy(int),
            "am": g.am_pathogenicity.to_numpy(float),
            "ref": g.blosum_pathogenicity.to_numpy(float),
            "gi": g.gene.map(gene_to_i).to_numpy(int),
        }
    rng = np.random.default_rng(SEED)
    out = []
    n = len(genes)
    for b in range(reps):
        counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        row = {"replicate": b}
        for lv, a in arrays.items():
            w = counts[a["gi"]]
            row[f"am_auc_{lv}"] = weighted_auc_sorted(a["y"], a["am"], w)
            row[f"ref_auc_{lv}"] = weighted_auc_sorted(a["y"], a["ref"], w)
        row["d_am"] = row["am_auc_0"] - row["am_auc_3"]
        row["d_ref"] = row["ref_auc_0"] - row["ref_auc_3"]
        row["contrast_am_minus_ref"] = row["d_am"] - row["d_ref"]
        out.append(row)
    return pd.DataFrame(out)


def ci(values: pd.Series) -> tuple[float, float]:
    return tuple(np.nanquantile(values, [.025, .975]))


def matched_extremes(df: pd.DataFrame) -> dict:
    x = df[df.axis_level.isin([0, 3])].copy()
    selected = []
    strata_used = 0
    for _, g in x.groupby(["gene", "label", "ref_aa", "alt_aa"], sort=True):
        far = g[g.axis_level == 0].sort_values("variant_key")
        near = g[g.axis_level == 3].sort_values("variant_key")
        n = min(len(far), len(near))
        if n:
            selected.extend([far.iloc[:n], near.iloc[:n]])
            strata_used += 1
    if not selected:
        return {"rows": 0, "strata": 0}
    m = pd.concat(selected, ignore_index=True)
    result = {"rows": len(m), "strata": strata_used, "genes": m.gene.nunique()}
    for lv, g in m.groupby("axis_level"):
        result[f"level_{lv}_rows"] = len(g)
        result[f"level_{lv}_auc"] = roc_auc_score(g.label, g.am_pathogenicity)
        result[f"level_{lv}_ref_auc"] = roc_auc_score(g.label, g.blosum_pathogenicity)
    result["am_degradation"] = result.get("level_0_auc", np.nan) - result.get("level_3_auc", np.nan)
    result["ref_degradation"] = result.get("level_0_ref_auc", np.nan) - result.get("level_3_ref_auc", np.nan)
    return result


def macro_gene_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (gene, lv), g in df.groupby(["gene", "axis_level"]):
        if g.label.nunique() == 2 and min(g.label.value_counts()) >= 2:
            rows.append({"gene": gene, "axis_level": lv, "n": len(g),
                         "am_auc": roc_auc_score(g.label, g.am_pathogenicity),
                         "blosum_auc": roc_auc_score(g.label, g.blosum_pathogenicity)})
    return pd.DataFrame(rows)


def confound_regression(df: pd.DataFrame) -> dict:
    x = df.copy()
    x["score_z"] = (x.am_pathogenicity - x.am_pathogenicity.mean()) / x.am_pathogenicity.std()
    x["level_z"] = x.axis_level.astype(float)
    x["log_length"] = np.log(x.protein_length)
    formula = ("label ~ score_z * level_z + blosum62 + position_fraction + log_length + "
               "review_stars + C(ref_aa) + C(alt_aa) + C(strand) + C(terminal_exon)")
    fit = smf.glm(formula, data=x, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": x.gene}, maxiter=100)
    names = ["score_z", "level_z", "score_z:level_z", "blosum62",
             "position_fraction", "log_length", "review_stars"]
    return {name: {"coefficient": float(fit.params[name]),
                   "se_cluster_gene": float(fit.bse[name]),
                   "p_value": float(fit.pvalues[name])} for name in names}


def plot_spc(table: pd.DataFrame, boot: pd.DataFrame, partition: str) -> None:
    names = table.axis_name.tolist()
    auc = table.am_auroc.to_numpy()
    lo, hi = [], []
    ref_lo, ref_hi = [], []
    for lv in LEVELS:
        a, b = ci(boot[f"am_auc_{lv}"]); lo.append(a); hi.append(b)
        a, b = ci(boot[f"ref_auc_{lv}"]); ref_lo.append(a); ref_hi.append(b)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(4)
    ax.errorbar(x, auc, yerr=[auc - np.array(lo), np.array(hi) - auc], marker="o",
                capsize=3, lw=2, label="AlphaMissense")
    ref = table.blosum_auroc.to_numpy()
    ax.errorbar(x, ref, yerr=[ref - np.array(ref_lo), np.array(ref_hi) - ref], marker="s",
                capsize=3, lw=1.5, label="BLOSUM62")
    ax.set_xticks(x, names); ax.set_ylabel("AUROC (95% gene-cluster bootstrap CI)")
    ax.set_xlabel("Frozen splice-distance level: far → near")
    ax.set_ylim(max(0.45, min(ref_lo) - .04), min(1, max(hi) + .03))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(ROOT / f"artifacts/plots/clinvar_{partition}_spc.{ext}", dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("partition", choices=["discovery", "confirmation"])
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    args = p.parse_args()
    check_access(args.partition)
    path = (ROOT / "data/processed/clinvar_discovery_scored.parquet" if args.partition == "discovery"
            else ROOT / "data/processed/clinvar_confirmation_outcomes_sealed.parquet")
    df = pd.read_parquet(path)
    table = pd.DataFrame([level_metrics(g) for _, g in df.groupby("axis_level")]).sort_values("axis_level")
    boot = cluster_bootstrap(df, args.bootstrap_reps)
    for lv in LEVELS:
        table.loc[table.axis_level == lv, "am_auc_ci_low"] = ci(boot[f"am_auc_{lv}"])[0]
        table.loc[table.axis_level == lv, "am_auc_ci_high"] = ci(boot[f"am_auc_{lv}"])[1]
        table.loc[table.axis_level == lv, "blosum_auc_ci_low"] = ci(boot[f"ref_auc_{lv}"])[0]
        table.loc[table.axis_level == lv, "blosum_auc_ci_high"] = ci(boot[f"ref_auc_{lv}"])[1]
    d_am = float(table.loc[table.axis_level == 0, "am_auroc"].iloc[0] -
                 table.loc[table.axis_level == 3, "am_auroc"].iloc[0])
    d_ref = float(table.loc[table.axis_level == 0, "blosum_auroc"].iloc[0] -
                  table.loc[table.axis_level == 3, "blosum_auroc"].iloc[0])
    contrast = d_am - d_ref
    category = "larger" if contrast > .03 else "smaller" if contrast < -.03 else "comparable"
    macro = macro_gene_level(df)
    macro_summary = (macro.groupby("axis_level").agg(
        genes=("gene", "nunique"), mean_am_auc=("am_auc", "mean"),
        median_am_auc=("am_auc", "median"), mean_blosum_auc=("blosum_auc", "mean"))
        .reset_index() if len(macro) else pd.DataFrame())
    summary = {
        "partition": args.partition, "rows": len(df), "genes": df.gene.nunique(),
        "proteins": df.uniprot_id.nunique(), "bootstrap_replicates": args.bootstrap_reps,
        "d_am_far_minus_splice": d_am, "d_am_ci95": ci(boot.d_am),
        "d_blosum_far_minus_splice": d_ref, "d_blosum_ci95": ci(boot.d_ref),
        "contrast_am_minus_blosum": contrast,
        "contrast_ci95": ci(boot.contrast_am_minus_ref),
        "reference_gate_call": category,
        "matched_extreme_sensitivity": matched_extremes(df),
        "confound_regression": confound_regression(df),
    }
    prefix = ROOT / f"artifacts/tables/clinvar_{args.partition}"
    table.to_csv(str(prefix) + "_spc.csv", index=False)
    boot.to_parquet(str(prefix) + "_bootstrap.parquet", index=False)
    macro.to_csv(str(prefix) + "_per_gene_level_auc.csv", index=False)
    macro_summary.to_csv(str(prefix) + "_macro_gene_spc.csv", index=False)
    (Path(str(prefix) + "_summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    df.to_csv(ROOT / f"artifacts/tables/clinvar_{args.partition}_per_variant.csv.gz",
              index=False, compression="gzip")
    plot_spc(table, boot, args.partition)
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
