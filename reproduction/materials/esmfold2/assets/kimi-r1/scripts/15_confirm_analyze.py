#!/usr/bin/env python
"""Confirmation analysis for frozen Axis A' (orphan vs non-orphan).
Pre-declared success: (i) Welch p<0.01 one-sided, orphan lower;
(ii) gap >= 0.08; (iii) direction consistent in all 3 length bins.
Writes results/confirm/confirm_summary.json + results/confirm/merged_per_unit.tsv
"""
import json, math
import numpy as np
from scipy import stats
from collections import defaultdict

rows = {}
for line in open("results/confirm/panel_final.tsv").read().splitlines()[1:]:
    p = line.split("\t")
    rows[p[0]] = {"group": p[1], "len": int(p[2]), "max_pident": float(p[3]),
                  "lcfrac": float(p[4]), "resolved_frac": float(p[5])}
for line in open("results/confirm/target_scores.tsv").read().splitlines()[1:]:
    p = line.split("\t")
    rows[p[0]].update({"tm": float(p[2]), "gdt_ts": float(p[3]),
                       "gdt_ha": float(p[4]), "rmsd": float(p[5])})
for line in open("results/confirm/pred_plddt.csv").read().splitlines()[1:]:
    p = line.split(",")
    if p[0] in rows:
        rows[p[0]]["plddt"] = float(p[2])
try:
    for line in open("results/confirm/baseline_scores.tsv").read().splitlines()[1:]:
        p = line.split("\t")
        if p[0] in rows:
            rows[p[0]]["base_tm"] = None if p[5] == "NA" else float(p[5])
except FileNotFoundError:
    pass

with open("results/confirm/merged_per_unit.tsv", "w") as f:
    f.write("rep\tgroup\tlen\tmax_pident\tlcfrac\tresolved_frac\ttm\tgdt_ts\tgdt_ha\trmsd\tplddt\tbase_tm\n")
    for rep, r in sorted(rows.items()):
        f.write("%s\t%s\t%d\t%.1f\t%.4f\t%.3f\t%.4f\t%.4f\t%.4f\t%.2f\t%.1f\t%s\n" % (
            rep, r["group"], r["len"], r["max_pident"], r["lcfrac"],
            r["resolved_frac"], r["tm"], r["gdt_ts"], r["gdt_ha"], r["rmsd"],
            r["plddt"], "%.4f" % r["base_tm"] if r.get("base_tm") is not None else "NA"))

orph = [r["tm"] for r in rows.values() if r["group"] == "orphan"]
non = [r["tm"] for r in rows.values() if r["group"] == "nonorphan"]
t, p2 = stats.ttest_ind(orph, non, equal_var=False)
p1 = p2 / 2 if t < 0 else 1 - p2 / 2
gap = float(np.mean(non) - np.mean(orph))
d = gap / math.sqrt((np.var(orph) + np.var(non)) / 2)

BINS = [(40, 150), (150, 250), (250, 401)]
bins = []
for lo, hi in BINS:
    o = [r["tm"] for r in rows.values() if r["group"] == "orphan" and lo <= r["len"] < hi]
    n_ = [r["tm"] for r in rows.values() if r["group"] == "nonorphan" and lo <= r["len"] < hi]
    bins.append({"bin": [lo, hi], "orphan_n": len(o), "nonorphan_n": len(n_),
                 "orphan_tm": float(np.mean(o)), "nonorphan_tm": float(np.mean(n_)),
                 "gap": float(np.mean(n_) - np.mean(o))})

bo = [r["base_tm"] for r in rows.values() if r["group"] == "orphan" and r.get("base_tm") is not None]
bn = [r["base_tm"] for r in rows.values() if r["group"] == "nonorphan" and r.get("base_tm") is not None]
orph_lo = [r for r in rows.values() if r["group"] == "orphan"]
frac_low_orph = float(np.mean([r["tm"] < 0.5 for r in orph_lo]))
frac_low_non = float(np.mean([r["tm"] < 0.5 for r in rows.values() if r["group"] == "nonorphan"]))

crit_i = bool(p1 < 0.01 and np.mean(orph) < np.mean(non))
crit_ii = bool(gap >= 0.08)
crit_iii = bool(all(b["gap"] > 0 for b in bins))
summary = {
    "n_orphan": len(orph), "n_nonorphan": len(non),
    "orphan_tm_mean": float(np.mean(orph)), "orphan_tm_sd": float(np.std(orph)),
    "orphan_tm_median": float(np.median(orph)),
    "nonorphan_tm_mean": float(np.mean(non)), "nonorphan_tm_sd": float(np.std(non)),
    "nonorphan_tm_median": float(np.median(non)),
    "gap": gap, "cohens_d": float(d),
    "welch_t": float(t), "welch_p_one_sided": float(p1),
    "frac_tm_lt_05_orphan": frac_low_orph, "frac_tm_lt_05_nonorphan": frac_low_non,
    "orphan_gdt_ts": float(np.mean([r["gdt_ts"] for r in rows.values() if r["group"] == "orphan"])),
    "nonorphan_gdt_ts": float(np.mean([r["gdt_ts"] for r in rows.values() if r["group"] == "nonorphan"])),
    "orphan_plddt": float(np.mean([r["plddt"] for r in rows.values() if r["group"] == "orphan"])),
    "nonorphan_plddt": float(np.mean([r["plddt"] for r in rows.values() if r["group"] == "nonorphan"])),
    "len_bins": bins,
    "baseline_orphan_covered_n": len(bo), "baseline_orphan_tm": float(np.mean(bo)) if bo else None,
    "baseline_nonorphan_covered_n": len(bn), "baseline_nonorphan_tm": float(np.mean(bn)) if bn else None,
    "criteria": {"i_sig": crit_i, "ii_gap": crit_ii, "iii_bins": crit_iii,
                 "CONFIRMED": crit_i and crit_ii and crit_iii},
}
json.dump(summary, open("results/confirm/confirm_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
