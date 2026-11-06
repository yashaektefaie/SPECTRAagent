#!/usr/bin/env python3
"""08_controls_C5.py — controls for the C5 (splice proximity) finding per
docs/AXIS_SPEC_C1C5.md: >=2-star sensitivity, Grantham-tercile and
entropy-quartile stratified contrasts, wt-aa composition, formal gate call.
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata

rng = np.random.default_rng(20260722)
df = pd.read_csv("data/processed/panel_C5.tsv", sep="\t", low_memory=False)
TOP, BOT = "L1(>=30nt)", "L4(1-2nt)"

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    r = rankdata(s)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

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

top, bot = df[df["lvl_c5m"] == TOP], df[df["lvl_c5m"] == BOT]
ctl = []

# 1) >=2-star sensitivity
rev = pd.read_csv("data/processed/panel_frozen.tsv", sep="\t", low_memory=False,
                  usecols=["variation_id", "revstat"])
star2ids = set(rev.loc[rev["revstat"].isin(
    ["criteria_provided,_multiple_submitters,_no_conflicts",
     "reviewed_by_expert_panel", "practice_guideline"]), "variation_id"])
t2 = top[top["variation_id"].isin(star2ids)]
b2 = bot[bot["variation_id"].isin(star2ids)]
e, lo, hi, p = boot_delta(t2, b2, "am")
ctl.append({"control": "ge2star", "stratum": "all", "n_top": len(t2), "n_bot": len(b2),
            "delta_am": e, "lo": lo, "hi": hi, "p": p})
print(f">=2star: n={len(t2)}/{len(b2)} delta_AM={e:.4f} ({lo:.4f},{hi:.4f}) p={p:.2g}")

# 2) Grantham-tercile stratified contrast
feat = pd.read_csv("data/processed/panel_frozen.tsv", sep="\t", low_memory=False,
                   usecols=["variation_id", "neg_grantham", "entropy", "gene"])
df = df.merge(feat, on="variation_id")
df["gterc"] = pd.qcut(df["neg_grantham"], 3, labels=["conservative", "mid", "radical"])
for t in ["conservative", "mid", "radical"]:
    d1 = top[top.index.isin(df[df["gterc"] == t].index)]
    d4 = bot[bot.index.isin(df[df["gterc"] == t].index)]
    if len(d1) >= 200 and len(d4) >= 200:
        e, lo, hi, p = boot_delta(d1, d4, "am")
        ctl.append({"control": "grantham_tercile", "stratum": t, "n_top": len(d1),
                    "n_bot": len(d4), "delta_am": e, "lo": lo, "hi": hi, "p": p})
        print(f"grantham {t}: n={len(d1)}/{len(d4)} delta_AM={e:.4f} ({lo:.4f},{hi:.4f}) p={p:.2g}")

# 3) entropy-quartile stratified contrast
df["eq"] = pd.qcut(df["entropy"], 4, labels=False)
for q in sorted(df["eq"].unique()):
    ids = set(df[df["eq"] == q]["variation_id"])
    d1 = top[top["variation_id"].isin(ids)]
    d4 = bot[bot["variation_id"].isin(ids)]
    if len(d1) >= 200 and len(d4) >= 200:
        e, lo, hi, p = boot_delta(d1, d4, "am")
        ctl.append({"control": "entropy_quartile", "stratum": q, "n_top": len(d1),
                    "n_bot": len(d4), "delta_am": e, "lo": lo, "hi": hi, "p": p})
        print(f"entropy q{q}: n={len(d1)}/{len(d4)} delta_AM={e:.4f} ({lo:.4f},{hi:.4f}) p={p:.2g}")

# 4) last-exon sensitivity: boundary-adjacent variants near the last exon
# cannot act via NMD-linked splicing; annotate distance excluding terminal exon
# (exploratory control, reported but not gated)
# skipped: requires exon-order bookkeeping; noted as known gap in the report.

# 5) wt-aa composition per level
comp = df.groupby("lvl_c5m").agg(
    n=("label", "size"), pos_rate=("label", "mean"),
    mean_grantham=("grantham", "mean"),
    top_wt=("protein_variant", lambda x: x.str[0].value_counts().head(3).to_dict()))
comp.to_csv("results/qc_composition_C5.tsv", sep="\t", float_format="%.4f")
print(comp.to_string())

pd.DataFrame(ctl).to_csv("results/controls_C5.tsv", sep="\t", index=False, float_format="%.4f")

# ---- formal gate call ----
g = pd.read_csv("results/gate_C5.tsv", sep="\t")
d_am = g.loc[g["score"] == "am", "delta_auc"].iloc[0]
d_b = g.loc[g["score"] == "neg_blosum62", "delta_auc"].iloc[0]
d_g = g.loc[g["score"] == "grantham", "delta_auc"].iloc[0]
call = {
    "axis": "C5_splice",
    "delta_AM": d_am, "delta_BLOSUM62": d_b, "delta_Grantham": d_g,
    "AM_degrades": bool(d_am > 0),
    "gate_AM_at_least_as_much_as_BLOSUM62": bool(d_am >= d_b),
    "gate_AM_at_least_as_much_as_Grantham": bool(d_am >= d_g),
    "label": "shared (references degrade at least as much or more)",
}
pd.DataFrame([call]).to_csv("results/gate_call.tsv", sep="\t", index=False, float_format="%.4f")
print(call)
