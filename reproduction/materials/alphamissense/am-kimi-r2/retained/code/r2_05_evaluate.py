#!/usr/bin/env python
"""r2 step 05: evaluate frozen axis A1.
Per-level AUC for: AM (mean/min/max), R1 Grantham, R2 gene-mean phyloP, diagnostic phyloP-in-level.
Bootstrap CIs (variant-level + gene-cluster), gene-stratified AUCs, confound table.
Outputs: results/r2_a1_per_level.tsv, results/r2_a1_tests.txt
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(20260722)
df = pd.read_csv("tables/r2_eval_table.tsv", sep="\t")
df = df.dropna(subset=["phyloP"]).copy()
bins = [-21, 0, 2, 4, 6, 11]
labels = ["L1","L2","L3","L4","L5"]
df["lvl"] = pd.cut(df.phyloP, bins, right=False, labels=labels)

PREDICTORS = {"AM_mean":"am_mean", "AM_min":"am_min", "AM_max":"am_max",
              "R1_Grantham":"grantham", "R2_genePhyloP":"gene_mean_phyloP",
              "diag_phyloP":"phyloP"}

def auc(y, s):
    m = ~(np.isnan(s))
    y, s = y[m], s[m]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, s)

rows = []
for lv in labels:
    sub = df[df.lvl == lv]
    y = sub.label.values.astype(float)
    r = {"level": lv, "n": len(sub), "n_path": int(y.sum()), "n_ben": int((1-y).sum()),
         "prev": y.mean(), "phyloP_median": sub.phyloP.median(),
         "n_genes": sub.gene.nunique(), "grantham_mean": sub.grantham.mean(),
         "genePhyloP_mean": sub.gene_mean_phyloP.mean()}
    for name, col in PREDICTORS.items():
        r[f"auc_{name}"] = auc(y, sub[col].values.astype(float))
    rows.append(r)
res = pd.DataFrame(rows)
res.to_csv("results/r2_a1_per_level.tsv", sep="\t", index=False)
print(res.to_string(index=False))

# ---- bootstrap: variant-level and gene-cluster, for AM_mean, R1, R2
def boot_delta(sub_all, col, kind="variant", B=2000):
    L1 = sub_all[sub_all.lvl=="L1"]; L5 = sub_all[sub_all.lvl=="L5"]
    ds = []
    if kind == "variant":
        y1, s1 = L1.label.values, L1[col].values
        y5, s5 = L5.label.values, L5[col].values
        ok1 = ~np.isnan(s1); ok5 = ~np.isnan(s5)
        y1, s1, y5, s5 = y1[ok1], s1[ok1], y5[ok5], s5[ok5]
        n1, n5 = len(y1), len(y5)
        for _ in range(B):
            i1 = rng.integers(0, n1, n1); i5 = rng.integers(0, n5, n5)
            a1, a5 = auc(y1[i1], s1[i1]), auc(y5[i5], s5[i5])
            if not (np.isnan(a1) or np.isnan(a5)):
                ds.append(a5 - a1)
    else:  # gene-cluster
        for lv, sign in (("L1",-1),("L5",1)):
            pass
        genes1 = L1.gene.values; genes5 = L5.gene.values
        ug1 = L1.gene.unique(); ug5 = L5.gene.unique()
        L1a = L1.dropna(subset=[col]); L5a = L5.dropna(subset=[col])
        g1 = L1a.gene.values; g5 = L5a.gene.values
        ug1 = np.unique(g1); ug5 = np.unique(g5)
        y1 = L1a.label.values; s1 = L1a[col].values
        y5 = L5a.label.values; s5 = L5a[col].values
        idx1 = {g: np.where(g1==g)[0] for g in ug1}
        idx5 = {g: np.where(g5==g)[0] for g in ug5}
        for _ in range(B):
            b1 = rng.choice(ug1, len(ug1), replace=True)
            b5 = rng.choice(ug5, len(ug5), replace=True)
            i1 = np.concatenate([idx1[g] for g in b1])
            i5 = np.concatenate([idx5[g] for g in b5])
            a1, a5 = auc(y1[i1], s1[i1]), auc(y5[i5], s5[i5])
            if not (np.isnan(a1) or np.isnan(a5)):
                ds.append(a5 - a1)
    ds = np.array(ds)
    return ds.mean(), np.percentile(ds, 2.5), np.percentile(ds, 97.5), (ds <= 0).mean()

out = open("results/r2_a1_tests.txt", "w")
for col, name in [("am_mean","AM_mean"), ("grantham","R1_Grantham"), ("gene_mean_phyloP","R2_genePhyloP")]:
    for kind in ("variant","cluster"):
        m, lo, hi, p0 = boot_delta(df, col, kind, B=2000)
        line = f"deltaAUC(L5-L1) {name:14s} [{kind:8s} boot] = {m:.4f}  95%CI [{lo:.4f},{hi:.4f}]  P(delta<=0)={p0:.4f}"
        print(line); out.write(line+"\n")

# ---- gene-stratified AUC per level (within-gene ranking, weighted mean over genes with both classes)
def stratified_auc(sub, col):
    vals, wts = [], []
    for g, gg in sub.groupby("gene"):
        y = gg.label.values; s = gg[col].values
        m = ~np.isnan(s)
        y, s = y[m], s[m]
        if len(y) >= 6 and 0 < y.sum() < len(y):
            vals.append(roc_auc_score(y, s)); wts.append(min(len(y), 50))
    if not vals: return np.nan, 0
    return np.average(vals, weights=wts), len(vals)

for col, name in [("am_mean","AM_mean"), ("grantham","R1_Grantham")]:
    for lv in labels:
        sub = df[df.lvl==lv]
        a, ng = stratified_auc(sub, col)
        line = f"gene-stratified AUC {name:12s} {lv} = {a:.4f} (genes={ng})"
        print(line); out.write(line+"\n")
out.close()
