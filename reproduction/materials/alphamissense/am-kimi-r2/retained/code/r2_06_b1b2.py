#!/usr/bin/env python
"""r2 step 06: evaluate frozen axes B1 (paired within-gene) and B2 (gene-level depth).
Outputs: results/r2_b1_paired.tsv, results/r2_b2_per_level.tsv, results/r2_b1b2_tests.txt
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(20260722)
df = pd.read_csv("tables/r2_eval_table.tsv", sep="\t").dropna(subset=["phyloP"]).copy()
out = open("results/r2_b1b2_tests.txt", "w")

def safe_auc(y, s):
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, s)

# ================= B1: paired within-gene =================
df["arm"] = np.where(df.phyloP < 2, "LOW", np.where(df.phyloP >= 6, "HIGH", "MID"))
cnt = df[df.arm.isin(["LOW","HIGH"])].groupby(["gene","arm","label"]).size().unstack(fill_value=0)
cnt.columns = ["ben","path"]
ok = cnt[(cnt.ben >= 2) & (cnt.path >= 2)].reset_index()
piv = ok.pivot_table(index="gene", columns="arm", values="path", aggfunc="size")
panel = piv.dropna().index
sub = df[df.gene.isin(panel) & df.arm.isin(["LOW","HIGH"])].copy()
print(f"B1 panel: {len(panel)} genes, variants: {len(sub)}")
out.write(f"B1 panel genes={len(panel)} variants={len(sub)}\n")

PREDS = {"AM_mean":"am_mean", "AM_min":"am_min", "AM_max":"am_max",
         "R1_Grantham":"grantham", "diag_phyloP":"phyloP"}
rows = []
for name, col in PREDS.items():
    diffs = []
    for g, gg in sub.groupby("gene"):
        lo = gg[gg.arm=="LOW"]; hi = gg[gg.arm=="HIGH"]
        a_lo = safe_auc(lo.label.values, lo[col].values.astype(float))
        a_hi = safe_auc(hi.label.values, hi[col].values.astype(float))
        if not (np.isnan(a_lo) or np.isnan(a_hi)):
            diffs.append((g, a_hi - a_lo, a_hi, a_lo))
    d = pd.DataFrame(diffs, columns=["gene","diff","auc_high","auc_low"])
    B = 2000
    boots = []
    dv = d["diff"].values
    for _ in range(B):
        boots.append(rng.choice(dv, len(dv), replace=True).mean())
    boots = np.array(boots)
    row = {"predictor": name, "n_genes": len(d),
           "mean_auc_high": d.auc_high.mean(), "mean_auc_low": d.auc_low.mean(),
           "delta": dv.mean(), "ci_lo": np.percentile(boots,2.5), "ci_hi": np.percentile(boots,97.5),
           "p_le0": (boots <= 0).mean(), "median_diff": d["diff"].median()}
    rows.append(row)
b1 = pd.DataFrame(rows)
b1.to_csv("results/r2_b1_paired.tsv", sep="\t", index=False)
print(b1.to_string(index=False))
out.write(b1.to_string(index=False)+"\n")

# ================= B2: gene-level evolutionary depth =================
df2 = df.dropna(subset=["gene_mean_phyloP"]).copy()
qs = [1.8001, 2.8121, 3.5615, 4.2782]
df2["glvl"] = pd.cut(df2.gene_mean_phyloP, [-2]+qs+[7], labels=["G1","G2","G3","G4","G5"])
rows = []
for lv in ["G1","G2","G3","G4","G5"]:
    s = df2[df2.glvl==lv]
    y = s.label.values.astype(float)
    r = {"level": lv, "n": len(s), "n_path": int(y.sum()), "prev": y.mean(),
         "genePhyloP_median": s.gene_mean_phyloP.median(), "n_genes": s.gene.nunique(),
         "cds_len_mean": s.cds_len.mean(), "pos_phyloP_mean": s.phyloP.mean(),
         "grantham_mean": s.grantham.mean()}
    for name, col in {"AM_mean":"am_mean","AM_min":"am_min","AM_max":"am_max",
                      "R1_Grantham":"grantham","R3_posPhyloP":"phyloP"}.items():
        r[f"auc_{name}"] = safe_auc(y, s[col].values.astype(float))
    rows.append(r)
b2 = pd.DataFrame(rows)
b2.to_csv("results/r2_b2_per_level.tsv", sep="\t", index=False)
print(b2.to_string(index=False))
out.write("\n"+b2.to_string(index=False)+"\n")

def boot_delta_b2(data, col, kind="variant", B=2000):
    A = data[data.glvl=="G1"].dropna(subset=[col]); Z = data[data.glvl=="G5"].dropna(subset=[col])
    ds = []
    if kind == "variant":
        yA, sA = A.label.values, A[col].values; yZ, sZ = Z.label.values, Z[col].values
        for _ in range(B):
            iA = rng.integers(0, len(yA), len(yA)); iZ = rng.integers(0, len(yZ), len(yZ))
            a, z = safe_auc(yA[iA], sA[iA]), safe_auc(yZ[iZ], sZ[iZ])
            if not (np.isnan(a) or np.isnan(z)): ds.append(z - a)
    else:
        def cluster_boot(D):
            g = D.gene.values; ug = np.unique(g)
            idx = {x: np.where(g==x)[0] for x in ug}
            bg = rng.choice(ug, len(ug), replace=True)
            ii = np.concatenate([idx[x] for x in bg])
            return D.label.values[ii], D[col].values[ii]
        for _ in range(B):
            yA, sA = cluster_boot(A); yZ, sZ = cluster_boot(Z)
            a, z = safe_auc(yA, sA), safe_auc(yZ, sZ)
            if not (np.isnan(a) or np.isnan(z)): ds.append(z - a)
    ds = np.array(ds)
    return ds.mean(), np.percentile(ds,2.5), np.percentile(ds,97.5), (ds<=0).mean()

for col, name in [("am_mean","AM_mean"),("grantham","R1_Grantham"),("phyloP","R3_posPhyloP")]:
    for kind in ("variant","cluster"):
        m, lo, hi, p0 = boot_delta_b2(df2, col, kind)
        line = f"B2 deltaAUC(G5-G1) {name:13s} [{kind:8s}] = {m:.4f} 95%CI [{lo:.4f},{hi:.4f}] P(d<=0)={p0:.4f}"
        print(line); out.write(line+"\n")
out.close()
