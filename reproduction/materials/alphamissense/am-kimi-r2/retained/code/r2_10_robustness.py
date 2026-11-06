#!/usr/bin/env python
"""r2 step 10: finish C1 robustness (fast cluster bootstrap + paired within-gene),
and B2 bootstrap for the record.
Outputs appended to results/r2_c1_tests.txt and results/r2_b2_tests.txt
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(20260722)

def safe_auc(y, s):
    m = ~np.isnan(s); y, s = y[m], s[m]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y): return np.nan
    return roc_auc_score(y, s)

def prep(D, col):
    g = D.gene.values; ug, inv = np.unique(g, return_inverse=True)
    splits = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    return D.label.values, D[col].values, ug, splits

def cluster_boot_delta(A, Z, col, B=1000):
    A = A.dropna(subset=[col]); Z = Z.dropna(subset=[col])
    yA, sA, ugA, spA = prep(A, col); yZ, sZ, ugZ, spZ = prep(Z, col)
    ds = []
    for _ in range(B):
        bA = rng.integers(0, len(ugA), len(ugA)); bZ = rng.integers(0, len(ugZ), len(ugZ))
        iA = np.concatenate([spA[i] for i in bA]); iZ = np.concatenate([spZ[i] for i in bZ])
        a, z = safe_auc(yA[iA], sA[iA]), safe_auc(yZ[iZ], sZ[iZ])
        if not (np.isnan(a) or np.isnan(z)): ds.append(z - a)
    ds = np.array(ds)
    return ds.mean(), np.percentile(ds,2.5), np.percentile(ds,97.5), (ds<=0).mean()

def paired(df, arm_col, lo, hi, preds, out, tag):
    d2 = df[df[arm_col].isin([lo,hi])]
    cnt = d2.groupby(["gene",arm_col,"label"]).size().unstack(fill_value=0)
    cnt.columns = ["ben","path"]
    ok = cnt[(cnt.ben>=2)&(cnt.path>=2)].reset_index()
    piv = ok.pivot_table(index="gene", columns=arm_col, values="path", aggfunc="size")
    panel = piv.dropna().index
    sub = d2[d2.gene.isin(panel)]
    rows = []
    for name, col in preds.items():
        diffs = []
        for g, gg in sub.groupby("gene"):
            a1 = safe_auc(gg[gg[arm_col]==lo].label.values, gg[gg[arm_col]==lo][col].values.astype(float))
            a4 = safe_auc(gg[gg[arm_col]==hi].label.values, gg[gg[arm_col]==hi][col].values.astype(float))
            if not (np.isnan(a1) or np.isnan(a4)): diffs.append(a4-a1)
        dv = np.array(diffs)
        boots = np.array([rng.choice(dv, len(dv), replace=True).mean() for _ in range(2000)])
        rows.append({"predictor": name, "n_genes": len(dv), "delta": dv.mean(),
                     "ci_lo": np.percentile(boots,2.5), "ci_hi": np.percentile(boots,97.5),
                     "p_le0": (boots<=0).mean()})
    r = pd.DataFrame(rows)
    out.write(f"\n{tag} paired within-gene ({hi}-{lo}), panel={len(panel)} genes:\n"+r.to_string(index=False)+"\n")
    print(tag); print(r.to_string(index=False))
    return r

# ---- C1
df = pd.read_csv("tables/r2_eval_table2.tsv", sep="\t").dropna(subset=["plddt"]).copy()
df["slvl"] = pd.cut(df.plddt, [0,50,70,90,101], right=False, labels=["S1","S2","S3","S4"])
out = open("results/r2_c1_tests.txt", "a")
PREDS = {"AM_mean":"am_mean","R1_Grantham":"grantham","R3_posPhyloP":"phyloP"}
for col, name in [("am_mean","AM_mean"),("grantham","R1_Grantham"),("phyloP","R3_posPhyloP")]:
    m, lo, hi, p0 = cluster_boot_delta(df[df.slvl=="S1"], df[df.slvl=="S4"], col)
    line = f"C1 deltaAUC(S4-S1) {name:13s} [cluster ] = {m:.4f} 95%CI [{lo:.4f},{hi:.4f}] P(d<=0)={p0:.4f}"
    print(line); out.write(line+"\n")
c1p = paired(df, "slvl", "S1", "S4", PREDS, out, "C1")
c1p.to_csv("results/r2_c1_paired.tsv", sep="\t", index=False)
out.close()

# ---- B2 (record only)
df2 = pd.read_csv("tables/r2_eval_table2.tsv", sep="\t").dropna(subset=["gene_mean_phyloP"]).copy()
qs = [1.8001, 2.8121, 3.5615, 4.2782]
df2["glvl"] = pd.cut(df2.gene_mean_phyloP, [-2]+qs+[7], labels=["G1","G2","G3","G4","G5"])
out = open("results/r2_b2_tests.txt", "w")
for col, name in [("am_mean","AM_mean"),("grantham","R1_Grantham"),("phyloP","R3_posPhyloP")]:
    # variant boot
    A = df2[df2.glvl=="G1"].dropna(subset=[col]); Z = df2[df2.glvl=="G5"].dropna(subset=[col])
    yA, sA = A.label.values, A[col].values; yZ, sZ = Z.label.values, Z[col].values
    ds = []
    for _ in range(1000):
        iA = rng.integers(0, len(yA), len(yA)); iZ = rng.integers(0, len(yZ), len(yZ))
        ds.append(safe_auc(yZ[iZ], sZ[iZ]) - safe_auc(yA[iA], sA[iA]))
    ds = np.array(ds)
    line = f"B2 deltaAUC(G5-G1) {name:13s} [variant ] = {ds.mean():.4f} 95%CI [{np.percentile(ds,2.5):.4f},{np.percentile(ds,97.5):.4f}]"
    print(line); out.write(line+"\n")
    m, lo, hi, p0 = cluster_boot_delta(df2[df2.glvl=="G1"], df2[df2.glvl=="G5"], col)
    line = f"B2 deltaAUC(G5-G1) {name:13s} [cluster ] = {m:.4f} 95%CI [{lo:.4f},{hi:.4f}] P(d<=0)={p0:.4f}"
    print(line); out.write(line+"\n")
out.close()
