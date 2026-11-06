#!/usr/bin/env python3
"""Analyze the protein-disjoint ProteinGym DMS mechanism-control panels."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260714


def metrics(g: pd.DataFrame) -> dict:
    y = g.dms_deleterious.to_numpy(int)
    return {
        "rows": len(g), "proteins": g.entry_name.nunique(), "assays": g.dms_id.nunique(),
        "deleterious": int(y.sum()), "fit": int((1-y).sum()),
        "median_d_splice_bp": g.d_splice_bp_median.median(),
        "am_auroc": roc_auc_score(y, g.am_pathogenicity),
        "blosum_auroc": roc_auc_score(y, g.blosum_pathogenicity),
        "am_spearman_negative_fitness": spearmanr(g.am_pathogenicity, -g.dms_score).statistic,
        "median_plddt": g.af2_plddt.median(),
        "low_confidence_fraction": g.af2_low_confidence_lt50.mean(),
        "median_msa_neff_l": g.msa_neff_l.median(),
    }


def weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    y, s, w = labels[order], scores[order], weights[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    pos = np.add.reduceat(w * (y == 1), starts)
    neg = np.add.reduceat(w * (y == 0), starts)
    p, n = pos.sum(), neg.sum()
    if p <= 0 or n <= 0:
        return np.nan
    before = np.cumsum(neg) - neg
    return float(np.sum(pos * (before + .5 * neg)) / (p * n))


def protein_bootstrap(df: pd.DataFrame, reps: int = 2000) -> pd.DataFrame:
    proteins = sorted(df.entry_name.unique())
    pmap = {p: i for i, p in enumerate(proteins)}
    arrays = {}
    for lv, g in df.groupby("axis_level"):
        arrays[int(lv)] = {
            "y": g.dms_deleterious.to_numpy(int),
            "am": g.am_pathogenicity.to_numpy(float),
            "ref": g.blosum_pathogenicity.to_numpy(float),
            "pi": g.entry_name.map(pmap).to_numpy(int),
        }
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(reps):
        counts = np.bincount(rng.integers(0, len(proteins), len(proteins)),
                             minlength=len(proteins)).astype(float)
        row = {"replicate": b}
        for lv, a in arrays.items():
            weights = counts[a["pi"]]
            row[f"am_auc_{lv}"] = weighted_auc(a["y"], a["am"], weights)
            row[f"ref_auc_{lv}"] = weighted_auc(a["y"], a["ref"], weights)
        if "am_auc_0" in row and "am_auc_3" in row:
            row["d_am"] = row["am_auc_0"] - row["am_auc_3"]
            row["d_ref"] = row["ref_auc_0"] - row["ref_auc_3"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    state = json.loads((ROOT / "controller/state.json").read_text())
    if not state.get("confirmation_accessible"):
        raise SystemExit("DMS confirmation outcomes are sealed")
    all_summary = {}
    for part in ["discovery", "confirmation"]:
        df = pd.read_parquet(ROOT / f"data/processed/proteingym_dms_{part}_scored.parquet")
        table = []
        for lv, g in df.groupby("axis_level"):
            row = {"axis_level": int(lv), "axis_name": g.axis_name.iloc[0]}
            row.update(metrics(g)); table.append(row)
        table = pd.DataFrame(table).sort_values("axis_level")
        boot = protein_bootstrap(df)
        for lv in table.axis_level:
            values = boot[f"am_auc_{lv}"].dropna()
            table.loc[table.axis_level.eq(lv), "am_auc_ci_low"] = values.quantile(.025)
            table.loc[table.axis_level.eq(lv), "am_auc_ci_high"] = values.quantile(.975)
        d = float(table.loc[table.axis_level.eq(0), "am_auroc"].iloc[0] -
                  table.loc[table.axis_level.eq(3), "am_auroc"].iloc[0])
        d_ci = [float(boot.d_am.quantile(.025)), float(boot.d_am.quantile(.975))]
        summary = {
            "partition": part, "rows": len(df), "proteins": df.entry_name.nunique(),
            "assays": df.dms_id.nunique(), "far_minus_splice_am_auc": d,
            "far_minus_splice_am_auc_ci95_protein_bootstrap": d_ci,
            "interpretation": "protein-effect DMS control degrades" if d > 0 else
                              "protein-effect DMS control does not degrade",
        }
        all_summary[part] = summary
        prefix = ROOT / f"artifacts/tables/dms_{part}"
        table.to_csv(str(prefix) + "_spc.csv", index=False)
        boot.to_parquet(str(prefix) + "_bootstrap.parquet", index=False)
        df.to_csv(str(prefix) + "_per_variant.csv.gz", index=False, compression="gzip")
        Path(str(prefix) + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        x = table.axis_level.to_numpy()
        y = table.am_auroc.to_numpy()
        ax.errorbar(x, y, yerr=[y-table.am_auc_ci_low.to_numpy(),
                                table.am_auc_ci_high.to_numpy()-y], marker="o", capsize=3,
                    label="AlphaMissense vs DMS")
        ax.plot(x, table.blosum_auroc, marker="s", label="BLOSUM62 vs DMS")
        ax.set_xticks(x, table.axis_name); ax.set_ylabel("AUROC for DMS-deleterious variants")
        ax.set_xlabel("Frozen genomic splice-distance level"); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout()
        for ext in ["png", "svg"]:
            fig.savefig(ROOT / f"artifacts/plots/dms_{part}_spc.{ext}", dpi=180)
        plt.close(fig)
        print(table.to_string(index=False)); print(json.dumps(summary, indent=2))
    (ROOT / "artifacts/tables/dms_control_summary.json").write_text(
        json.dumps(all_summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
