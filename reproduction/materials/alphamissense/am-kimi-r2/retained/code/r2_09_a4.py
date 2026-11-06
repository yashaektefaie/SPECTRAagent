#!/usr/bin/env python
"""r2 step 09: evaluate frozen axis A4 (Grantham). Fast (no cluster boot inner np.where;
uses precomputed gene-index lists).
Outputs: results/r2_a4_per_level.tsv, results/r2_a4_paired.tsv, results/r2_a4_tests.txt
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(20260722)
out = open("results/r2_a4_tests.txt", "w")

def safe_auc(y, s):
    m = ~np.isnan(s); y, s = y[m], s[m]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y): return np.nan
    return roc_auc_score(y, s)

df = pd.read_csv("tables/r2_eval_table2.tsv", sep="\t").dropna(subset=["grantham"]).copy()
df["tlvl"] = pd.cut(df.grantham, [-1,60,100,150,216], right=False, labels=["T1","T2","T3","T4"])
LV = ["T1","T2","T3","T4"]
PREDS = {"AM_mean":"am_mean","AM_min":"am_min","AM_max":"am_max",
         "R3_posPhyloP":"phyloP","R2_genePhyloP":"gene_mean_phyloP"}

rows = []
for lv in LV:
    s = df[df.tlvl==lv]; y = s.label.values.astype(float)
    r = {"level": lv, "n": len(s), "n_path": int(y.sum()), "prev": y.mean(),
         "grantham_median": s.grantham.median(), "n_genes": s.gene.nunique(),
         "phyloP_mean": s.phyloP.mean()}
    for name, col in PREDS.items():
        r[f"auc_{name}"] = safe_auc(y, s[col].values.astype(float))
    rows.append(r)
a4 = pd.DataFrame(rows)
a4.to_csv("results/r2_a4_per_level.tsv", sep="\t", index=False)
print(a4.to_string(index=False)); out.write(a4.to_string(index=False)+"\n\n")

def boot_delta(data, col, kind, B=1000):
    A = data[data.tlvl=="T1"].dropna(subset=[col]); Z = data[data.tlvl=="T4"].dropna(subset=[col])
    ds = []
    if kind == "variant":
        yA, sA = A.label.values, A[col].values; yZ, sZ = Z.label.values, Z[col].values
        nA, nZ = len(yA), len(yZ)
        for _ in range(B):
            iA = rng.integers(0, nA, nA); iZ = rng.integers(0, nZ, nZ)
            a, z = safe_auc(yA[iA], sA[iA]), safe_auc(yZ[iZ], sZ[iZ])
            if not (np.isnan(a) or np.isnan(z)): ds.append(z-a)
    else:
        def prep(D):
            g = D.gene.values; ug, inv = np.unique(g, return_inverse=True)
            splits = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
            return D.label.values, D[col].values, ug, splits
        yA0, sA0, ugA, spA = prep(A); yZ0, sZ0, ugZ, spZ = prep(Z)
        for _ in range(B):
            bA = rng.integers(0, len(ugA), len(ugA)); bZ = rng.integers(0, len(ugZ), len(ugZ))
            iA = np.concatenate([spA[i] for i in bA]); iZ = np.concatenate([spZ[i] for i in bZ])
            a, z = safe_auc(yA0[iA], sA0[iA]), safe_auc(yZ0[iZ], sZ0[iZ])
            if not (np.isnan(a) or np.isnan(z)): ds.append(z-a)
    ds = np.array(ds)
    return ds.mean(), np.percentile(ds,2.5), np.percentile(ds,97.5), (ds<=0).mean()

for col, name in [("am_mean","AM_mean"),("phyloP","R3_posPhyloP"),("gene_mean_phyloP","R2_genePhyloP")]:
    for kind in ("variant","cluster"):
        m, lo, hi, p0 = boot_delta(df, col, kind)
        line = f"A4 deltaAUC(T4-T1) {name:14s} [{kind:8s}] = {m:.4f} 95%CI [{lo:.4f},{hi:.4f}] P(d<=0)={p0:.4f}"
        print(line); out.write(line+"\n")

# paired within-gene control
d2 = df[df.tlvl.isin(["T1","T4"])]
cnt = d2.groupby(["gene","tlvl","label"]).size().unstack(fill_value=0)
cnt.columns = ["ben","path"]
ok = cnt[(cnt.ben>=2)&(cnt.path>=2)].reset_index()
piv = ok.pivot_table(index="gene", columns="tlvl", values="path", aggfunc="size")
panel = piv.dropna().index
sub = d2[d2.gene.isin(panel)]
rows = []
for name, col in PREDS.items():
    diffs = []
    for g, gg in sub.groupby("gene"):
        a1 = safe_auc(gg[gg.tlvl=="T1"].label.values, gg[gg.tlvl=="T1"][col].values.astype(float))
        a4 = safe_auc(gg[gg.tlvl=="T4"].label.values, gg[gg.tlvl=="T4"][col].values.astype(float))
        if not (np.isnan(a1) or np.isnan(a4)): diffs.append(a4-a1)
    dv = np.array(diffs)
    if len(dv) == 0:
        rows.append({"predictor": name, "n_genes": 0}); continue
    boots = np.array([rng.choice(dv, len(dv), replace=True).mean() for _ in range(2000)])
    rows.append({"predictor": name, "n_genes": len(dv), "delta_T4-T1": dv.mean(),
                 "ci_lo": np.percentile(boots,2.5), "ci_hi": np.percentile(boots,97.5),
                 "p_le0": (boots<=0).mean()})
a4p = pd.DataFrame(rows)
a4p.to_csv("results/r2_a4_paired.tsv", sep="\t", index=False)
print(a4p.to_string(index=False)); out.write("\nPaired within-gene (T4-T1):\n"+a4p.to_string(index=False)+"\n")
out.close()
