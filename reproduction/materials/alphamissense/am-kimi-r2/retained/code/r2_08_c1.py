#!/usr/bin/env python
"""r2 step 08: evaluate frozen axis C1 (pLDDT) + finish B2 bootstrap record.
Outputs: results/r2_c1_per_level.tsv, results/r2_c1_paired.tsv, results/r2_c1_tests.txt
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(20260722)
out = open("results/r2_c1_tests.txt", "w")

def safe_auc(y, s):
    m = ~np.isnan(s); y, s = y[m], s[m]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y): return np.nan
    return roc_auc_score(y, s)

df = pd.read_csv("tables/r2_eval_table2.tsv", sep="\t").dropna(subset=["plddt"]).copy()
df["slvl"] = pd.cut(df.plddt, [0,50,70,90,101], right=False, labels=["S1","S2","S3","S4"])
LV = ["S1","S2","S3","S4"]
PREDS = {"AM_mean":"am_mean","AM_min":"am_min","AM_max":"am_max",
         "R1_Grantham":"grantham","R3_posPhyloP":"phyloP"}

rows = []
for lv in LV:
    s = df[df.slvl==lv]; y = s.label.values.astype(float)
    r = {"level": lv, "n": len(s), "n_path": int(y.sum()), "prev": y.mean(),
         "plddt_median": s.plddt.median(), "n_genes": s.gene.nunique(),
         "phyloP_mean": s.phyloP.mean(), "grantham_mean": s.grantham.mean()}
    for name, col in PREDS.items():
        r[f"auc_{name}"] = safe_auc(y, s[col].values.astype(float))
    rows.append(r)
c1 = pd.DataFrame(rows)
c1.to_csv("results/r2_c1_per_level.tsv", sep="\t", index=False)
print(c1.to_string(index=False)); out.write(c1.to_string(index=False)+"\n\n")

# missingness by label (spec confound 4)
full = pd.read_csv("tables/r2_eval_table2.tsv", sep="\t")
miss = full.groupby("label").plddt.apply(lambda s: s.isna().mean())
out.write(f"pLDDT missing rate by label: {miss.to_dict()}\n\n")

# bootstrap deltas S4-S1 (variant + gene-cluster)
def boot_delta(data, col, kind, B=2000):
    A = data[data.slvl=="S1"].dropna(subset=[col]); Z = data[data.slvl=="S4"].dropna(subset=[col])
    ds = []
    if kind == "variant":
        yA, sA = A.label.values, A[col].values; yZ, sZ = Z.label.values, Z[col].values
        for _ in range(B):
            iA = rng.integers(0, len(yA), len(yA)); iZ = rng.integers(0, len(yZ), len(yZ))
            a, z = safe_auc(yA[iA], sA[iA]), safe_auc(yZ[iZ], sZ[iZ])
            if not (np.isnan(a) or np.isnan(z)): ds.append(z-a)
    else:
        yA0, sA0, gA = A.label.values, A[col].values, A.gene.values
        yZ0, sZ0, gZ = Z.label.values, Z[col].values, Z.gene.values
        ugA, ugZ = np.unique(gA), np.unique(gZ)
        for _ in range(B):
            def cb(y0, s0, g, ug):
                bg = rng.choice(ug, len(ug), replace=True)
                ii = np.concatenate([np.where(g==x)[0] for x in bg])
                return y0[ii], s0[ii]
            yA, sA = cb(yA0, sA0, gA, ugA); yZ, sZ = cb(yZ0, sZ0, gZ, ugZ)
            a, z = safe_auc(yA, sA), safe_auc(yZ, sZ)
            if not (np.isnan(a) or np.isnan(z)): ds.append(z-a)
    ds = np.array(ds)
    return ds.mean(), np.percentile(ds,2.5), np.percentile(ds,97.5), (ds<=0).mean()

for col, name in [("am_mean","AM_mean"),("grantham","R1_Grantham"),("phyloP","R3_posPhyloP")]:
    for kind in ("variant","cluster"):
        m, lo, hi, p0 = boot_delta(df, col, kind, B=1000)
        line = f"C1 deltaAUC(S4-S1) {name:13s} [{kind:8s}] = {m:.4f} 95%CI [{lo:.4f},{hi:.4f}] P(d<=0)={p0:.4f}"
        print(line); out.write(line+"\n")

# paired within-gene control: genes with >=2 path + >=2 ben in S1 and S4
d2 = df[df.slvl.isin(["S1","S4"])]
cnt = d2.groupby(["gene","slvl","label"]).size().unstack(fill_value=0)
cnt.columns = ["ben","path"]
ok = cnt[(cnt.ben>=2)&(cnt.path>=2)].reset_index()
piv = ok.pivot_table(index="gene", columns="slvl", values="path", aggfunc="size")
panel = piv.dropna().index
sub = d2[d2.gene.isin(panel)]
rows = []
for name, col in PREDS.items():
    diffs = []
    for g, gg in sub.groupby("gene"):
        a1 = safe_auc(gg[gg.slvl=="S1"].label.values, gg[gg.slvl=="S1"][col].values.astype(float))
        a4 = safe_auc(gg[gg.slvl=="S4"].label.values, gg[gg.slvl=="S4"][col].values.astype(float))
        if not (np.isnan(a1) or np.isnan(a4)): diffs.append(a4-a1)
    dv = np.array(diffs)
    boots = np.array([rng.choice(dv, len(dv), replace=True).mean() for _ in range(2000)])
    rows.append({"predictor": name, "n_genes": len(dv), "delta_S4-S1": dv.mean(),
                 "ci_lo": np.percentile(boots,2.5), "ci_hi": np.percentile(boots,97.5),
                 "p_le0": (boots<=0).mean()})
c1p = pd.DataFrame(rows)
c1p.to_csv("results/r2_c1_paired.tsv", sep="\t", index=False)
print(c1p.to_string(index=False)); out.write("\nPaired within-gene (S4-S1):\n"+c1p.to_string(index=False)+"\n")
out.close()
