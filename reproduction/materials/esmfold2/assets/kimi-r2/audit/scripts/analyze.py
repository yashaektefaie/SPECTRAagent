#!/usr/bin/env python
"""Confirmatory analysis of the frozen axis (n=160, pre-declared n=40/bin).

Endpoints (pre-declared):
 1. mean TM(I1) - mean TM(I4) > 0, Welch t p<0.05
 2. Spearman trend of TM_model vs bin order, p<0.05
Gate: model degradation vs template-copy reference degradation on units where
both defined (I1-I3) + I4 statement.
Sensitivity: S2 obs_frac>=0.8, S3 single-protein-entity, S4 both.
"""
import csv, json, statistics as st
from scipy import stats
import numpy as np

rows = list(csv.DictReader(open("audit/results/scores.csv")))
conf = json.load(open("audit/data/entry_confounds.json"))
BINS = ["I1_ge80", "I2_40_80", "I3_20_40", "I4_lt20"]

for r in rows:
    e = r["entity"]
    r["entry"] = e.split("_")[0]
    obs = 0
    for line in open(f"audit/results/exp/{e}.pdb"):
        if line.startswith("ATOM"):
            obs += 1
    r["obs_frac"] = obs / int(r["len"])
    r["prot_ent"] = conf[r["entry"]]["prot_ent"]
    r["tm_model"] = float(r["tm_model"]) if r.get("tm_model") else None
    r["tm_ref"] = float(r["tm_ref"]) if r.get("tm_ref") else None

def report(subset, label):
    print(f"\n=== {label} (n={len(subset)}) ===")
    means = {}
    for b in BINS:
        v = [r["tm_model"] for r in subset if r["bin"] == b and r["tm_model"] is not None]
        vr = [r["tm_ref"] for r in subset if r["bin"] == b and r["tm_ref"] is not None]
        means[b] = (st.mean(v) if v else None, st.stdev(v) if len(v) > 1 else None, len(v),
                    st.mean(vr) if vr else None, len(vr))
        print(f"  {b}: model {means[b][0]:.4f} +/- {means[b][1]:.3f} (n={means[b][2]})"
              f" | ref {f'{means[b][3]:.4f}' if means[b][3] else 'n/a'} (n={means[b][4]})")
    x = [BINS.index(r["bin"]) for r in subset if r["tm_model"] is not None]
    y = [r["tm_model"] for r in subset if r["tm_model"] is not None]
    rho, p_s = stats.spearmanr(x, y)
    a = [r["tm_model"] for r in subset if r["bin"] == "I1_ge80" and r["tm_model"] is not None]
    d = [r["tm_model"] for r in subset if r["bin"] == "I4_lt20" and r["tm_model"] is not None]
    t, p_t = stats.ttest_ind(a, d, equal_var=False)
    print(f"  delta I1-I4 = {st.mean(a)-st.mean(d):.4f}, Welch p = {p_t:.2e}")
    print(f"  Spearman rho = {rho:.3f}, p = {p_s:.2e}")
    lens = [[int(r["len"]) for r in subset if r["bin"] == b] for b in BINS]
    kw = stats.kruskal(*lens)
    print(f"  length KW p = {kw.pvalue:.3f}; len means = "
          + ", ".join(f"{st.mean(l):.0f}" for l in lens))
    # per-unit paired model-vs-ref on I1-I3 where both defined
    both = [r for r in subset if r["tm_model"] is not None and r["tm_ref"] is not None]
    if both:
        diffs = [r["tm_model"] - r["tm_ref"] for r in both]
        print(f"  paired model-ref (I1-I3, n={len(both)}): mean diff {st.mean(diffs):+.4f}")
    return means

m_all = report(rows, "S1 all units")
s2 = [r for r in rows if r["obs_frac"] >= 0.8]
report(s2, "S2 obs_frac>=0.8")
s3 = [r for r in rows if r["prot_ent"] == 1]
report(s3, "S3 single-protein-entity entries")
s4 = [r for r in rows if r["obs_frac"] >= 0.8 and r["prot_ent"] == 1]
report(s4, "S4 = S2 & S3")

# gate decomposition: per-bin paired diffs
print("\n=== gate decomposition (model minus ref, per bin) ===")
for b in BINS[:3]:
    both = [(r["tm_model"], r["tm_ref"]) for r in rows
            if r["bin"] == b and r["tm_model"] is not None and r["tm_ref"] is not None]
    if both:
        diffs = [a - c for a, c in both]
        t, p = stats.ttest_1samp(diffs, 0)
        print(f"  {b}: n={len(diffs)} mean model-ref = {st.mean(diffs):+.4f} (p={p:.2e})")

# write enriched per-unit table
keys = list(rows[0].keys()) + ["entry", "obs_frac", "prot_ent"]
with open("audit/results/per_unit_enriched.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print("\nwrote audit/results/per_unit_enriched.csv")
