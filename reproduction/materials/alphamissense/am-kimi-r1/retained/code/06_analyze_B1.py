#!/usr/bin/env python3
"""06_analyze_B1.py — frozen axis B1: homology availability (ortholog count).

Builds ORTHO_COUNT per gene from Biomart Compara ortholog presence across 6
species, maps to panel variants via ClinVar GENEINFO Entrez ID, then runs the
frozen-spec analysis (docs/AXIS_SPEC_B1.md).
"""
import gzip, csv
import numpy as np, pandas as pd
from scipy.stats import rankdata

rng = np.random.default_rng(20260722)
SPECIES = ["ptroglodytes", "mmusculus", "ggallus", "drerio", "dmelanogaster", "scerevisiae"]

# ---- 1) vid -> Entrez GeneID from ClinVar VCF (GENEINFO=SYMBOL:EntrezID) ----
panel = pd.read_csv("data/processed/panel_frozen.tsv", sep="\t", low_memory=False)
need = set(panel["variation_id"].astype(str))
vid2entrez = {}
with gzip.open("data/raw/clinvar.vcf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.split("\t", 8)
        vid = p[2]
        if vid in need:
            gi = [kv for kv in p[7].split(";") if kv.startswith("GENEINFO=")]
            if gi:
                vid2entrez[vid] = gi[0][9:].split("|")[0].split(":")[1]
print(f"entrez mapped: {len(vid2entrez)}/{len(need)}")
panel["entrez"] = panel["variation_id"].astype(str).map(vid2entrez)

# ---- 2) ensembl -> entrez, and ortholog presence per species ----
gi = pd.read_csv("data/raw/biomart/gene_info.tsv", sep="\t", dtype=str)
gi.columns = ["ensembl", "entrez", "symbol"]
gi = gi.dropna(subset=["entrez"])
e2e = dict(zip(gi["ensembl"], gi["entrez"]))

ortho = {}  # ensembl -> count of species with >=1 ortholog
for sp in SPECIES:
    d = pd.read_csv(f"data/raw/biomart/ortho_{sp}.tsv", sep="\t", dtype=str)
    d.columns = ["ensembl", "orth"]
    has_orth = set(d.loc[d["orth"].notna() & (d["orth"] != ""), "ensembl"])
    for ens in has_orth:
        ortho[ens] = ortho.get(ens, 0) + 1

# entrez -> max ortho count over its ensembl ids
entrez2count = {}
for ens, ent in e2e.items():
    c = ortho.get(ens, 0)
    if ent not in entrez2count or c > entrez2count[ent]:
        entrez2count[ent] = c

panel["ortho_count"] = panel["entrez"].map(entrez2count)
n_unmatched = panel["ortho_count"].isna().sum()
print(f"variants without entrez->ortho match: {n_unmatched} "
      f"({panel.loc[panel['ortho_count'].isna(),'gene'].nunique()} genes)")
df = panel.dropna(subset=["ortho_count"]).copy()
df["ortho_count"] = df["ortho_count"].astype(int)
print(df["ortho_count"].value_counts().sort_index())

df["am"] = df["am_pathogenicity"].astype(float)
df["label"] = df["label"].astype(int)
df["neg_blosum62"] = -df["blosum62"]
df["grantham"] = -df["neg_grantham"]
SCORES = ["am", "neg_blosum62", "grantham"]

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    r = rankdata(s)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def boot_auc(y, s, B=2000):
    y = np.asarray(y); s = np.asarray(s); n = len(y)
    est = auc(y, s)
    bs = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        bs[b] = auc(y[i], s[i])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return est, lo, hi

def boot_delta(d1, d4, score, B=5000):
    y1, s1 = d1["label"].values, d1[score].values
    y4, s4 = d4["label"].values, d4[score].values
    est = auc(y1, s1) - auc(y4, s4)
    bs = np.empty(B)
    for b in range(B):
        i = rng.integers(0, len(y1), len(y1))
        j = rng.integers(0, len(y4), len(y4))
        bs[b] = auc(y1[i], s1[i]) - auc(y4[j], s4[j])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return est, lo, hi, min(p, 1.0)

# ---- 3) frozen levels: L1=6, L2=4-5, L3=1-3, L4=0 (merge if <1000) ----
def lev(c):
    return "L1(6)" if c == 6 else "L2(4-5)" if c >= 4 else "L3(1-3)" if c >= 1 else "L4(0)"
df["lvl0"] = df["ortho_count"].map(lev)
order0 = ["L1(6)", "L2(4-5)", "L3(1-3)", "L4(0)"]
counts0 = df["lvl0"].value_counts().to_dict()
print("pre-merge level counts:", counts0)
# deterministic predeclared merge: repeatedly fold the smallest <1000 level
# upward (toward L1), decided on paper before assignment
groups = [[l] for l in order0]
def gcount(g):
    return sum(counts0.get(l, 0) for l in g)
while len(groups) > 1 and min(gcount(g) for g in groups) < 1000:
    i = min(range(len(groups)), key=lambda k: gcount(groups[k]))
    j = i - 1 if i > 0 else 1
    groups[j] = groups[j] + groups[i] if j < i else groups[i] + groups[j]
    del groups[i]
name_of = {l: "+".join(g) for g in groups for l in g}
order = ["+".join(g) for g in groups]
df["lvl"] = df["lvl0"].map(name_of)
counts = df["lvl"].value_counts()
print("post-merge levels:", counts.to_dict())

rows = []
levels = {}
for l in order:
    d = df[df["lvl"] == l]
    levels[l] = d
    row = {"axis": "B1_ortho_count", "level": l, "n": len(d),
           "n_pos": int(d["label"].sum()), "pos_rate": d["label"].mean(),
           "feat_mean": d["ortho_count"].mean()}
    for sc in SCORES:
        e, lo, hi = boot_auc(d["label"], d[sc])
        row[f"auc_{sc}"], row[f"auc_{sc}_lo"], row[f"auc_{sc}_hi"] = e, lo, hi
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv("results/spc_B1_ortho.tsv", sep="\t", index=False, float_format="%.4f")
print(res[["level", "n", "n_pos", "pos_rate", "feat_mean", "auc_am",
           "auc_neg_blosum62", "auc_grantham"]].to_string(index=False))

top, bot = levels[order[0]], levels[order[-1]]
gate_rows = []
for sc in SCORES:
    e, lo, hi, p = boot_delta(top, bot, sc)
    gate_rows.append({"axis": "B1_ortho_count", "score": sc, "delta_auc": e,
                      "ci_lo": lo, "ci_hi": hi, "boot_p": p})
    print(f"delta {order[0]}->{order[-1]} [{sc}]: {e:.4f} (95% CI {lo:.4f},{hi:.4f}) p={p:.2g}")
pd.DataFrame(gate_rows).to_csv("results/gate_B1.tsv", sep="\t", index=False, float_format="%.4f")

# ---- 4) controls: contrast within entropy quartile and length quartile ----
ctl = []
for feat, tag in [("entropy", "entropy_q"), ("protein_length", "length_q")]:
    df["_q"] = pd.qcut(df[feat], 4, labels=False)
    for q in sorted(df["_q"].unique()):
        d1 = df[(df["_q"] == q) & (df["lvl"] == order[0])]
        d4 = df[(df["_q"] == q) & (df["lvl"] == order[-1])]
        if len(d1) >= 200 and len(d4) >= 200:
            e, lo, hi, p = boot_delta(d1, d4, "am", B=2000)
            ctl.append({"control": tag, "stratum": q, "n_top": len(d1),
                        "n_bot": len(d4), "delta_am": e, "lo": lo, "hi": hi})
            print(f"control {tag} q{q}: n={len(d1)}/{len(d4)} delta_AM={e:.4f} ({lo:.4f},{hi:.4f})")
star2 = df[df["revstat"].isin(["criteria_provided,_multiple_submitters,_no_conflicts",
                               "reviewed_by_expert_panel", "practice_guideline"])]
d1 = star2[star2["lvl"] == order[0]]; d4 = star2[star2["lvl"] == order[-1]]
if len(d1) >= 200 and len(d4) >= 200:
    e, lo, hi, p = boot_delta(d1, d4, "am", B=2000)
    ctl.append({"control": "ge2star", "stratum": "all", "n_top": len(d1),
                "n_bot": len(d4), "delta_am": e, "lo": lo, "hi": hi})
    print(f"control >=2star: n={len(d1)}/{len(d4)} delta_AM={e:.4f} ({lo:.4f},{hi:.4f})")
pd.DataFrame(ctl).to_csv("results/controls_B1.tsv", sep="\t", index=False, float_format="%.4f")

df[["variation_id", "uniprot_id", "protein_variant", "gene", "entrez", "ortho_count",
    "label", "am", "neg_blosum62", "grantham", "lvl"]].to_csv(
    "data/processed/panel_B1.tsv", sep="\t", index=False, float_format="%.6f")
print("wrote results/spc_B1_ortho.tsv results/gate_B1.tsv results/controls_B1.tsv data/processed/panel_B1.tsv")
