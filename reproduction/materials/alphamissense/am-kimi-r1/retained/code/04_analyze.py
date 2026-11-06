#!/usr/bin/env python3
"""04_analyze.py — frozen-spec analysis for axis A1 (local sequence complexity)
plus predeclared secondary axes A2 (terminus distance) and A3 (protein length).

Implements docs/AXIS_SPEC.md: quartile levels, per-level AUROC for
AlphaMissense and both fixed references, bootstrap CIs, L1->L4 gate call,
within-gene control, >=2-star sensitivity, composition/label-prior QC.
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata

rng = np.random.default_rng(20260722)
P = "data/processed/panel_frozen.tsv"
df = pd.read_csv(P, sep="\t", low_memory=False)
df["am"] = df["am_pathogenicity"].astype(float)
df["label"] = df["label"].astype(int)
# put ALL scores on the pathogenicity scale (higher = more pathogenic) so that
# per-level AUCs and degradation deltas are comparable across model/references
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

def run_axis(frame, feat, name, outfile):
    qs = frame[feat].quantile([0.25, 0.5, 0.75]).values
    lv = pd.cut(frame[feat], [-np.inf, qs[0], qs[1], qs[2], np.inf],
                labels=["L1_low", "L2", "L3", "L4_high"])
    # order levels from highest feature (least stressed) to lowest (most stressed)
    order = ["L4_high", "L3", "L2", "L1_low"]
    labels = {"L4_high": "L1(highest)", "L3": "L2", "L2": "L3", "L1_low": "L4(lowest)"}
    rows = []
    levels = {}
    for q in order:
        d = frame[lv == q]
        levels[labels[q]] = d
        row = {"axis": name, "level": labels[q], "n": len(d),
               "n_pos": int(d["label"].sum()),
               "pos_rate": d["label"].mean(),
               "feat_mean": d[feat].mean(), "feat_min": d[feat].min(),
               "feat_max": d[feat].max()}
        for sc in SCORES:
            e, lo, hi = boot_auc(d["label"], d[sc])
            row[f"auc_{sc}"] = e
            row[f"auc_{sc}_lo"] = lo
            row[f"auc_{sc}_hi"] = hi
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(outfile, sep="\t", index=False, float_format="%.4f")
    # gate deltas: highest-feature level vs lowest-feature level
    top, bot = levels["L1(highest)"], levels["L4(lowest)"]
    deltas = {}
    for sc in SCORES:
        deltas[sc] = boot_delta(top, bot, sc)
    return res, deltas, lv, labels

print("=== PRIMARY AXIS A1: windowed entropy (21-aa) ===")
res1, delta1, lv1, lab1 = run_axis(df, "entropy", "A1_entropy", "results/spc_A1_entropy.tsv")
print(res1[["level", "n", "n_pos", "feat_mean", "auc_am", "auc_neg_blosum62", "auc_grantham"]].to_string(index=False))

gate_rows = []
for sc, (e, lo, hi, p) in delta1.items():
    gate_rows.append({"axis": "A1_entropy", "score": sc, "delta_auc": e,
                      "ci_lo": lo, "ci_hi": hi, "boot_p": p})
    print(f"delta AUROC L1->L4 [{sc}]: {e:.4f} (95% CI {lo:.4f},{hi:.4f}) p={p:.2g}")

# GATE CALL: does AM degrade at least as much as each fixed reference?
d_am = delta1["am"][0]
d_r1 = delta1["neg_blosum62"][0]
d_r2 = delta1["grantham"][0]
call_r1 = d_am >= d_r1
call_r2 = d_am >= d_r2
label = ("model-specific-leaning" if (call_r1 and call_r2 and d_am > 1.5 * max(d_r1, d_r2))
         else "shared" if (d_r1 > 0 or d_r2 > 0) else "model-specific")
gate_rows.append({"axis": "A1_entropy", "score": "GATE_CALL",
                  "delta_auc": d_am,
                  "ci_lo": call_r1 and call_r2, "ci_hi": 0, "boot_p": 0})
pd.DataFrame(gate_rows).to_csv("results/gate_A1.tsv", sep="\t", index=False, float_format="%.4f")
print(f"GATE: dAM={d_am:.4f} >= dBLOSUM={d_r1:.4f}? {call_r1}; >= dGrantham={d_r2:.4f}? {call_r2}; label={label}")

print("\n=== SECONDARY AXIS A2: distance to nearest terminus ===")
res2, delta2, _, _ = run_axis(df, "term_dist", "A2_term_dist", "results/spc_A2_term.tsv")
print(res2[["level", "n", "feat_mean", "auc_am", "auc_neg_blosum62", "auc_grantham"]].to_string(index=False))
for sc, (e, lo, hi, p) in delta2.items():
    print(f"delta [{sc}]: {e:.4f} ({lo:.4f},{hi:.4f})")

print("\n=== SECONDARY AXIS A3: protein length ===")
res3, delta3, _, _ = run_axis(df, "protein_length", "A3_length", "results/spc_A3_length.tsv")
print(res3[["level", "n", "feat_mean", "auc_am", "auc_neg_blosum62", "auc_grantham"]].to_string(index=False))
for sc, (e, lo, hi, p) in delta3.items():
    print(f"delta [{sc}]: {e:.4f} ({lo:.4f},{hi:.4f})")

# ---------- controls for A1 ----------
print("\n=== CONTROLS (A1) ===")
df["lvl"] = lv1.map(lab1)
top = df[df["lvl"] == "L1(highest)"]
bot = df[df["lvl"] == "L4(lowest)"]

# 1) within-gene control: genes with >=5 variants in each extreme level
gc = pd.concat([top, bot]).groupby(["gene", "lvl"]).size().unstack(fill_value=0)
genes = gc[(gc >= 5).all(axis=1)].index
sub = df[df["gene"].isin(genes) & df["lvl"].isin(["L1(highest)", "L4(lowest)"])]
print(f"within-gene subset: {len(genes)} genes, {len(sub)} variants")
ctl = []
for lev in ["L1(highest)", "L4(lowest)"]:
    d = sub[sub["lvl"] == lev]
    e, lo, hi = boot_auc(d["label"], d["am"])
    ctl.append({"control": "within_gene", "level": lev, "n": len(d),
                "n_genes": d["gene"].nunique(), "auc_am": e, "lo": lo, "hi": hi})
    print(f"  {lev}: n={len(d)} genes={d['gene'].nunique()} AUC_AM={e:.4f} ({lo:.4f},{hi:.4f})")

# 2) >=2-star sensitivity
star2 = df[df["revstat"].isin(["criteria_provided,_multiple_submitters,_no_conflicts",
                               "reviewed_by_expert_panel", "practice_guideline"])]
for lev, dd in [("L1(highest)", top), ("L4(lowest)", bot)]:
    d = dd[dd.index.isin(star2.index)]
    if len(d) > 500:
        e, lo, hi = boot_auc(d["label"], d["am"])
        ctl.append({"control": "ge2star", "level": lev, "n": len(d),
                    "n_genes": d["gene"].nunique(), "auc_am": e, "lo": lo, "hi": hi})
        print(f"  >=2star {lev}: n={len(d)} AUC_AM={e:.4f} ({lo:.4f},{hi:.4f})")

# 3) substitution-severity composition: AM AUC within radical/conservative halves
med = df["neg_grantham"].median()  # more negative = more radical
for lev, dd in [("L1(highest)", top), ("L4(lowest)", bot)]:
    for nm, mask in [("conservative", dd["neg_grantham"] >= med),
                     ("radical", dd["neg_grantham"] < med)]:
        d = dd[mask]
        e = auc(d["label"], d["am"])
        ctl.append({"control": f"grantham_{nm}", "level": lev, "n": len(d),
                    "n_genes": d["gene"].nunique(), "auc_am": e, "lo": np.nan, "hi": np.nan})
        print(f"  {nm} {lev}: n={len(d)} AUC_AM={e:.4f}")

pd.DataFrame(ctl).to_csv("results/controls_A1.tsv", sep="\t", index=False, float_format="%.4f")

# 4) composition QC table
comp = df.groupby("lvl").agg(
    n=("label", "size"), pos_rate=("label", "mean"),
    mean_grantham=("neg_grantham", "mean"), mean_blosum=("blosum62", "mean"),
    mean_len=("protein_length", "mean"), mean_termd=("term_dist", "mean"),
    top_wt_aa=("wt_aa", lambda x: x.value_counts().head(3).to_dict()),
)
comp.to_csv("results/qc_composition_A1.tsv", sep="\t", float_format="%.4f")
print(comp.to_string())

# save level assignment for reproducibility
df[["variation_id", "uniprot_id", "protein_variant", "label", "am",
    "blosum62", "neg_grantham", "entropy", "lvl"]].to_csv(
    "data/processed/panel_scored.tsv", sep="\t", index=False, float_format="%.6f")
print("\nwrote results/spc_A1_entropy.tsv results/gate_A1.tsv results/controls_A1.tsv "
      "results/qc_composition_A1.tsv data/processed/panel_scored.tsv")
