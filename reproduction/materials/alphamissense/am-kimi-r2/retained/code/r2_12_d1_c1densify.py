#!/usr/bin/env python
"""r2 step 12: evaluate frozen axis D1 (CpG) + C1 densified within-gene curve +
C1 phyloP-stratified robustness.
Outputs: results/r2_d1_per_level.tsv, results/r2_d1_paired.tsv, results/r2_d1_tests.txt,
         results/r2_c1_densified.tsv
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

def boot_delta(A, Z, col, kind, B=1000):
    A = A.dropna(subset=[col]); Z = Z.dropna(subset=[col])
    ds = []
    if kind == "variant":
        yA, sA = A.label.values, A[col].values; yZ, sZ = Z.label.values, Z[col].values
        for _ in range(B):
            iA = rng.integers(0, len(yA), len(yA)); iZ = rng.integers(0, len(yZ), len(yZ))
            a, z = safe_auc(yA[iA], sA[iA]), safe_auc(yZ[iZ], sZ[iZ])
            if not (np.isnan(a) or np.isnan(z)): ds.append(z-a)
    else:
        yA, sA, ugA, spA = prep(A, col); yZ, sZ, ugZ, spZ = prep(Z, col)
        for _ in range(B):
            bA = rng.integers(0, len(ugA), len(ugA)); bZ = rng.integers(0, len(ugZ), len(ugZ))
            iA = np.concatenate([spA[i] for i in bA]); iZ = np.concatenate([spZ[i] for i in bZ])
            a, z = safe_auc(yA[iA], sA[iA]), safe_auc(yZ[iZ], sZ[iZ])
            if not (np.isnan(a) or np.isnan(z)): ds.append(z-a)
    ds = np.array(ds)
    return ds.mean(), np.percentile(ds,2.5), np.percentile(ds,97.5), (ds<=0).mean()

def paired_delta(df, arm_col, lo, hi, col, min_n=2):
    d2 = df[df[arm_col].isin([lo,hi])]
    cnt = d2.groupby(["gene",arm_col,"label"], observed=True).size().unstack(fill_value=0)
    cnt.columns = ["ben","path"]
    ok = cnt[(cnt.ben>=min_n)&(cnt.path>=min_n)].reset_index()
    piv = ok.pivot_table(index="gene", columns=arm_col, values="path", aggfunc="size")
    panel = piv.dropna().index
    sub = d2[d2.gene.isin(panel)]
    diffs = []
    for g, gg in sub.groupby("gene"):
        a1 = safe_auc(gg[gg[arm_col]==lo].label.values, gg[gg[arm_col]==lo][col].values.astype(float))
        a4 = safe_auc(gg[gg[arm_col]==hi].label.values, gg[gg[arm_col]==hi][col].values.astype(float))
        if not (np.isnan(a1) or np.isnan(a4)): diffs.append(a4-a1)
    dv = np.array(diffs)
    if len(dv) < 10: return len(panel), np.nan, np.nan, np.nan, np.nan
    boots = np.array([rng.choice(dv, len(dv), replace=True).mean() for _ in range(2000)])
    return len(panel), dv.mean(), np.percentile(boots,2.5), np.percentile(boots,97.5), (boots<=0).mean()

# ============ D1: CpG ============
df = pd.read_csv("tables/r2_eval_table3.tsv", sep="\t")
out = open("results/r2_d1_tests.txt", "w")
PREDS = {"AM_mean":"am_mean","AM_min":"am_min","AM_max":"am_max",
         "R1_Grantham":"grantham","R3_posPhyloP":"phyloP"}
rows = []
for lv in [0,1,2]:
    s = df[df.cpg_ctx==lv]; y = s.label.values.astype(float)
    r = {"level": f"CTX{lv}", "n": len(s), "n_path": int(y.sum()), "prev": y.mean(),
         "n_genes": s.gene.nunique(), "phyloP_mean": s.phyloP.mean(),
         "grantham_mean": s.grantham.mean(), "grantham_ge100_frac": (s.grantham>=100).mean()}
    for name, col in PREDS.items():
        r[f"auc_{name}"] = safe_auc(y, s[col].values.astype(float))
    rows.append(r)
d1 = pd.DataFrame(rows)
d1.to_csv("results/r2_d1_per_level.tsv", sep="\t", index=False)
print(d1.to_string(index=False)); out.write(d1.to_string(index=False)+"\n\n")

for col, name in [("am_mean","AM_mean"),("grantham","R1_Grantham"),("phyloP","R3_posPhyloP")]:
    for kind in ("variant","cluster"):
        # delta = AUC(CTX0) - AUC(CTX2): boot returns hi-lo with Z=CTX2, A=CTX0 => compute a-z
        A = df[df.cpg_ctx==0]; Z = df[df.cpg_ctx==2]
        ds = []
        for b in range(1000):
            pass
        m, lo, hi, p0 = boot_delta(Z, A, col, kind)  # Z=CTX2 lo? we want CTX0-CTX2 => AUC(A)-AUC(Z)
        line = f"D1 deltaAUC(CTX0-CTX2) {name:12s} [{kind:8s}] = {m:.4f} 95%CI [{lo:.4f},{hi:.4f}] P(d<=0)={p0:.4f}"
        print(line); out.write(line+"\n")

rows = []
for col, name in [("am_mean","AM_mean"),("grantham","R1_Grantham"),("phyloP","R3_posPhyloP")]:
    ng, m, lo, hi, p0 = paired_delta(df, "cpg_ctx", 2, 0, col)  # hi=CTX0 - lo=CTX2
    rows.append({"predictor": name, "n_genes": ng, "delta_CTX0-CTX2": m,
                 "ci_lo": lo, "ci_hi": hi, "p_le0": p0})
d1p = pd.DataFrame(rows)
d1p.to_csv("results/r2_d1_paired.tsv", sep="\t", index=False)
print(d1p.to_string(index=False)); out.write("\nPaired within-gene (CTX0-CTX2):\n"+d1p.to_string(index=False)+"\n")

# Grantham-stratified robustness for AM
for strat, cond in [("grantham<100", df.grantham<100), ("grantham>=100", df.grantham>=100)]:
    s = df[cond]
    a0 = safe_auc(s[s.cpg_ctx==0].label.values, s[s.cpg_ctx==0].am_mean.values.astype(float))
    a2 = safe_auc(s[s.cpg_ctx==2].label.values, s[s.cpg_ctx==2].am_mean.values.astype(float))
    line = f"D1 robustness [{strat}]: AUC CTX0={a0:.4f} CTX2={a2:.4f} delta={a0-a2:.4f} (n0={cond.sum() and (s.cpg_ctx==0).sum()}, n2={(s.cpg_ctx==2).sum()})"
    print(line); out.write(line+"\n")
out.close()

# ============ C1 densified within-gene curve ============
out = open("results/r2_c1_densified.tsv", "w")
dfc = df.dropna(subset=["plddt"]).copy()
dfc["slvl"] = pd.cut(dfc.plddt, [0,50,70,90,101], right=False, labels=["S1","S2","S3","S4"])
# panel: genes with >=2 path + >=2 ben in ALL four levels
cnt = dfc.groupby(["gene","slvl","label"], observed=True).size().unstack(fill_value=0)
cnt.columns = ["ben","path"]
ok = cnt[(cnt.ben>=2)&(cnt.path>=2)].reset_index()
piv = ok.pivot_table(index="gene", columns="slvl", values="path", aggfunc="size")
panel4 = piv.dropna().index
print(f"C1 densified panel (all 4 levels): {len(panel4)} genes")
out.write(f"# panel genes with >=2path+>=2ben in all 4 levels: {len(panel4)}\n")
sub = dfc[dfc.gene.isin(panel4)]
hdr = "gene\tauc_S1_AM\tauc_S2_AM\tauc_S3_AM\tauc_S4_AM\tauc_S1_grantham\tauc_S4_grantham\tauc_S1_phyloP\tauc_S4_phyloP"
out.write(hdr+"\n")
curves = []
for g, gg in sub.groupby("gene"):
    row = [g]
    for lv in ["S1","S2","S3","S4"]:
        row.append(safe_auc(gg[gg.slvl==lv].label.values, gg[gg.slvl==lv].am_mean.values.astype(float)))
    row.append(safe_auc(gg[gg.slvl=="S1"].label.values, gg[gg.slvl=="S1"].grantham.values.astype(float)))
    row.append(safe_auc(gg[gg.slvl=="S4"].label.values, gg[gg.slvl=="S4"].grantham.values.astype(float)))
    row.append(safe_auc(gg[gg.slvl=="S1"].label.values, gg[gg.slvl=="S1"].phyloP.values.astype(float)))
    row.append(safe_auc(gg[gg.slvl=="S4"].label.values, gg[gg.slvl=="S4"].phyloP.values.astype(float)))
    curves.append(row)
    out.write("\t".join(str(round(x,4)) if isinstance(x,float) else str(x) for x in row)+"\n")
cur = pd.DataFrame(curves, columns=hdr.split("\t"))
means = cur.drop(columns=["gene"]).mean()
boots = {}
for col in ["auc_S1_AM","auc_S2_AM","auc_S3_AM","auc_S4_AM"]:
    v = cur[col].values
    b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(2000)])
    boots[col] = (v.mean(), np.percentile(b,2.5), np.percentile(b,97.5))
print("C1 densified within-gene AM curve (mean [CI]):")
for k, v in boots.items(): print(f"  {k}: {v[0]:.4f} [{v[1]:.4f},{v[2]:.4f}]")
    # paired deltas along the curve
for pair in [("auc_S4_AM","auc_S1_AM"),("auc_S4_AM","auc_S2_AM"),("auc_S3_AM","auc_S1_AM")]:
    d = (cur[pair[0]]-cur[pair[1]]).values
    b = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(2000)])
    print(f"  paired {pair[0]}-{pair[1]}: {d.mean():.4f} [{np.percentile(b,2.5):.4f},{np.percentile(b,97.5):.4f}]")
# phyloP-stratified robustness of C1 paired effect
for strat, cond in [("phyloP in [0,6)", (dfc.phyloP>=0)&(dfc.phyloP<6)), ("phyloP>=6", dfc.phyloP>=6), ("phyloP<0", dfc.phyloP<0)]:
    s = dfc[cond]
    ng, m, lo, hi, p0 = paired_delta(s, "slvl", "S1", "S4", "am_mean")
    print(f"C1 paired S4-S1 AM [{strat}]: genes={ng} delta={m:.4f} [{lo:.4f},{hi:.4f}] p_le0={p0:.4f}")
out.close()
