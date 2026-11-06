#!/usr/bin/env python
"""Cross-validate our fresh-inference pipeline against the OFFICIAL cell-eval
per-perturbation metrics (weights/fewshot_k562/eval_last.ckpt/k562_results.csv).

We compute pearson_delta per unit two ways:
  A) deltas vs predicted-control mean  (cell-eval variant)
  B) deltas vs real-control mean       (our pre-declared primary variant)
and report agreement with the official table.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def unit_means(x, perts):
    df = pd.DataFrame({"pert": perts})
    out = {}
    idx = df.groupby("pert").indices
    for k, ii in idx.items():
        out[k] = x[ii].mean(axis=0)
    return out

def main():
    z = np.load("results/preds_test_k562.npz")
    preds, reals = z["preds"], z["reals"]
    pert = z["pert"].astype(str)
    pm = unit_means(preds, pert)
    rm = unit_means(reals, pert)
    ctrl = "non-targeting"
    assert ctrl in pm and ctrl in rm, "control cells missing from test split"
    pctrl, rctrl = pm[ctrl], rm[ctrl]

    off = pd.read_csv("weights/fewshot_k562/eval_last.ckpt/k562_results.csv")
    off = off.set_index("perturbation")
    rows = []
    for g in off.index:
        if g not in pm:
            rows.append((g, np.nan, np.nan, off.loc[g, "pearson_delta"]))
            continue
        d_r = rm[g] - rctrl
        d_pA = pm[g] - pctrl   # cell-eval: pred delta vs pred control
        d_pB = pm[g] - rctrl   # ours: pred delta vs real control
        pdA = pearsonr(d_pA, d_r)[0] if d_r.std() > 0 else np.nan
        pdB = pearsonr(d_pB, d_r)[0] if d_r.std() > 0 else np.nan
        rows.append((g, pdA, pdB, off.loc[g, "pearson_delta"]))
    df = pd.DataFrame(rows, columns=["gene", "pdA_predctrl", "pdB_realctrl", "official"])
    df.to_csv("results/crosscheck_pearson_delta.csv", index=False)
    for c in ["pdA_predctrl", "pdB_realctrl"]:
        sub = df.dropna(subset=[c, "official"])
        r_p = pearsonr(sub[c], sub["official"])[0]
        r_s = spearmanr(sub[c], sub["official"])[0]
        mad = np.abs(sub[c] - sub["official"]).mean()
        print(f"{c}: n={len(sub)} pearson={r_p:.4f} spearman={r_s:.4f} "
              f"mean|diff|={mad:.4f} mean_ours={sub[c].mean():.4f} "
              f"mean_official={sub['official'].mean():.4f}")

if __name__ == "__main__":
    main()
