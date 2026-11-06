#!/usr/bin/env python3
"""07_analyze_C1C5.py — frozen axes C1 (population observability) and
C5 (splice proximity). Implements docs/AXIS_SPEC_C1C5.md.
"""
import gzip
import numpy as np, pandas as pd
from scipy.stats import rankdata

rng = np.random.default_rng(20260722)
df = pd.read_csv("data/processed/panel_frozen.tsv", sep="\t", low_memory=False)
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

def run_levels(df, lvlcol, order, axis_name, spc_out, gate_out):
    rows, levels = [], {}
    for l in order:
        d = df[df[lvlcol] == l]
        levels[l] = d
        row = {"axis": axis_name, "level": l, "n": len(d),
               "n_pos": int(d["label"].sum()), "pos_rate": d["label"].mean()}
        for sc in SCORES:
            e, lo, hi = boot_auc(d["label"], d[sc])
            row[f"auc_{sc}"], row[f"auc_{sc}_lo"], row[f"auc_{sc}_hi"] = e, lo, hi
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(spc_out, sep="\t", index=False, float_format="%.4f")
    print(res[["level", "n", "n_pos", "pos_rate", "auc_am", "auc_neg_blosum62",
               "auc_grantham"]].to_string(index=False))
    gate_rows = []
    top, bot = levels[order[0]], levels[order[-1]]
    for sc in SCORES:
        e, lo, hi, p = boot_delta(top, bot, sc)
        gate_rows.append({"axis": axis_name, "score": sc, "delta_auc": e,
                          "ci_lo": lo, "ci_hi": hi, "boot_p": p})
        print(f"delta {order[0]}->{order[-1]} [{sc}]: {e:.4f} (95% CI {lo:.4f},{hi:.4f}) p={p:.2g}")
    pd.DataFrame(gate_rows).to_csv(gate_out, sep="\t", index=False, float_format="%.4f")
    return levels

# ---------- C1: population observability ----------
need = set(df["variation_id"].astype(str))
afs = {}
with gzip.open("data/raw/clinvar.vcf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.split("\t", 8)
        if p[2] in need:
            info = dict(kv.split("=", 1) for kv in p[7].split(";") if "=" in kv)
            vals = [float(info[k]) for k in ("AF_ESP", "AF_EXAC", "AF_TGP") if k in info]
            afs[p[2]] = max(vals) if vals else np.nan
df["af_max"] = df["variation_id"].astype(str).map(afs)
print(f"C1: AF extracted for {df['af_max'].notna().sum()}/{len(df)} variants")

c1 = df.copy()
c1["lvl_c1"] = np.where(c1["af_max"].isna(), "L1(unobserved)",
               np.where(c1["af_max"] < 1e-3, "L2(rare<1e-3)", "L3(common>=1e-3)"))
print(c1["lvl_c1"].value_counts())
print("\n=== C1: population observability ===")
run_levels(c1, "lvl_c1", ["L1(unobserved)", "L2(rare<1e-3)", "L3(common>=1e-3)"],
           "C1_popobs", "results/spc_C1_popobs.tsv", "results/gate_C1.tsv")

# ---------- C5: splice proximity ----------
# exons for AM transcripts from GENCODE v50 basic
want_tx = set(t.split(".")[0] for t in df["transcript_id"].unique())
exons = {}  # unversioned ENST -> list of (start, end)
with gzip.open("data/raw/gencode.v50.basic.annotation.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if p[2] != "exon":
            continue
        attrs = p[8]
        tid = None
        for kv in attrs.split(";"):
            kv = kv.strip()
            if kv.startswith("transcript_id"):
                tid = kv.split('"')[1].split(".")[0]
                break
        if tid in want_tx:
            exons.setdefault(tid, []).append((int(p[3]), int(p[4])))
print(f"C5: exons loaded for {len(exons)}/{len(want_tx)} AM transcripts")

def splice_dist(row):
    tid = row["transcript_id"].split(".")[0]
    pos = row["pos"]
    best = None
    inside = False
    for s, e in exons.get(tid, []):
        if s <= pos <= e:
            inside = True
            d = min(pos - s, e - pos)
            best = d if best is None else min(best, d)
    return best if inside else np.nan

df["splice_dist"] = df.apply(splice_dist, axis=1)
n_nan = df["splice_dist"].isna().sum()
print(f"C5: variants not inside an exon of the AM transcript: {n_nan} (excluded)")
c5 = df.dropna(subset=["splice_dist"]).copy()

def c5_lev(d):
    if d >= 30: return "L1(>=30nt)"
    if d >= 11: return "L2(11-29nt)"
    if d >= 3:  return "L3(3-10nt)"
    return "L4(1-2nt)"
c5["lvl_c5"] = c5["splice_dist"].map(c5_lev)
order5 = ["L1(>=30nt)", "L2(11-29nt)", "L3(3-10nt)", "L4(1-2nt)"]
counts = c5["lvl_c5"].value_counts().to_dict()
print("C5 level counts:", counts)
# predeclared merge toward L1 if <1000
groups = [[l] for l in order5]
def gcount(g): return sum(counts.get(l, 0) for l in g)
while len(groups) > 1 and min(gcount(g) for g in groups) < 1000:
    i = min(range(len(groups)), key=lambda k: gcount(groups[k]))
    j = i - 1 if i > 0 else 1
    groups[j] = groups[j] + groups[i] if j < i else groups[i] + groups[j]
    del groups[i]
name_of = {l: "+".join(g) for g in groups for l in g}
order = ["+".join(g) for g in groups]
c5["lvl_c5m"] = c5["lvl_c5"].map(name_of)
print("C5 post-merge:", c5["lvl_c5m"].value_counts().to_dict())
print("\n=== C5: splice proximity ===")
run_levels(c5, "lvl_c5m", order, "C5_splice", "results/spc_C5_splice.tsv", "results/gate_C5.tsv")

c5[["variation_id", "uniprot_id", "protein_variant", "gene", "transcript_id",
    "splice_dist", "lvl_c5m", "label", "am", "neg_blosum62", "grantham"]].to_csv(
    "data/processed/panel_C5.tsv", sep="\t", index=False, float_format="%.6f")
c1[["variation_id", "af_max", "lvl_c1"]].to_csv(
    "data/processed/panel_C1_af.tsv", sep="\t", index=False)
print("wrote results/spc_C1_popobs.tsv results/gate_C1.tsv results/spc_C5_splice.tsv "
      "results/gate_C5.tsv data/processed/panel_C5.tsv data/processed/panel_C1_af.tsv")
