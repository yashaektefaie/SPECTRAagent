#!/usr/bin/env python
"""Merge all per-unit artifacts, run the frozen audit statistics, and write
results/spc_summary.json + results/merged_per_unit.tsv + figures.

Statistics (pre-declared in docs/FROZEN_AXIS_SPEC.md):
- per-level mean/median TM (ESMFold) and baseline (covered examples)
- Jonckheere-Terpstra permutation trend test across A1..A4 (10k perms)
- Welch t A1 vs A4 + Cohen's d
- length-standardized (3-bin) level means (pre-declared confound control)
- confound table: length, resolution, resolved fraction, lcfrac per level
- gate: delta_model vs delta_baseline along the frozen axis
"""
import json, math
import numpy as np
from collections import defaultdict
from scipy import stats

def load_panel():
    rows = {}
    for line in open("results/panel_final.tsv").read().splitlines()[1:]:
        p = line.split("\t")
        rows[p[0]] = {"levelA": p[1], "levelB": p[2], "len": int(p[3]),
                      "max_pident": float(p[4]), "lcfrac": float(p[5]),
                      "resolved_frac": float(p[6])}
    return rows

panel = load_panel()

# ESMFold scores
for line in open("results/target_scores.tsv").read().splitlines()[1:]:
    p = line.split("\t")
    panel[p[0]].update({"n_resolved": int(p[1]), "tm": float(p[2]),
                        "gdt_ts": float(p[3]), "gdt_ha": float(p[4]),
                        "rmsd": float(p[5])})
# pLDDT
for line in open("results/pred_plddt.csv").read().splitlines()[1:]:
    p = line.split(",")
    if p[0] in panel:
        panel[p[0]]["plddt"] = float(p[2])
# baseline
for line in open("results/baseline_scores.tsv").read().splitlines()[1:]:
    p = line.split("\t")
    if p[0] not in panel:
        continue
    panel[p[0]]["base_tm"] = None if p[5] == "NA" else float(p[5])
    panel[p[0]]["base_cov"] = float(p[3])
    panel[p[0]]["base_pairs"] = int(p[4])

# resolution from CIFs
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
res_cache = {}
def resolution(rep):
    ent = rep.split("_")[0]
    if ent in res_cache:
        return res_cache[ent]
    try:
        d = MMCIF2Dict("data/cifs/%s.cif" % ent)
        v = d.get("_refine.ls_d_res_high", [None])[0]
        v = float(v) if v not in (None, ".", "?") else None
    except Exception:
        v = None
    res_cache[ent] = v
    return v

for rep in panel:
    panel[rep]["resolution"] = resolution(rep)

# merged per-unit table
with open("results/merged_per_unit.tsv", "w") as f:
    f.write("rep\tlevelA\tlevelB\tlen\tmax_pident\tlcfrac\tresolved_frac\tresolution\ttm\tgdt_ts\tgdt_ha\trmsd\tplddt\tbase_tm\tbase_pairs\n")
    for rep, r in sorted(panel.items()):
        f.write("%s\t%s\t%s\t%d\t%.1f\t%.4f\t%.3f\t%s\t%.4f\t%.4f\t%.4f\t%.2f\t%.1f\t%s\t%s\n" % (
            rep, r["levelA"], r["levelB"], r["len"], r["max_pident"], r["lcfrac"],
            r["resolved_frac"],
            "%.2f" % r["resolution"] if r["resolution"] else "NA",
            r["tm"], r["gdt_ts"], r["gdt_ha"], r["rmsd"], r["plddt"],
            "%.4f" % r["base_tm"] if r["base_tm"] is not None else "NA",
            r.get("base_pairs", 0)))

LEVELS = ["A1", "A2", "A3", "A4"]
def vals(key, lev, cond=None):
    return [r[key] for r in panel.values() if r["levelA"] == lev and key in r
            and r[key] is not None and (cond is None or cond(r))]

def jt_perm(groups, n_perm=10000, seed=1):
    """Jonckheere-Terpstra: H1 = decreasing medians across ordered groups."""
    def stat(gs):
        s = 0
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                for a in gs[i]:
                    for b in gs[j]:
                        s += (a > b) + 0.5 * (a == b)
        return s
    obs = stat(groups)
    pooled = [x for g in groups for x in g]
    sizes = [len(g) for g in groups]
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        gs, k = [], 0
        for s in sizes:
            gs.append(perm[k:k + s]); k += s
        if stat(gs) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)

summary = {"levels": {}}
groups = []
for lev in LEVELS:
    tm = vals("tm", lev)
    groups.append(tm)
    base_pairs = [(r["tm"], r["base_tm"]) for r in panel.values()
                  if r["levelA"] == lev and r.get("base_tm") is not None]
    btm = [b for _, b in base_pairs]
    mtm = [m for m, _ in base_pairs]
    summary["levels"][lev] = {
        "n": len(tm),
        "tm_mean": float(np.mean(tm)), "tm_sd": float(np.std(tm)),
        "tm_median": float(np.median(tm)),
        "frac_tm_ge_05": float(np.mean([x >= 0.5 for x in tm])),
        "gdt_ts_mean": float(np.mean(vals("gdt_ts", lev))),
        "plddt_mean": float(np.mean(vals("plddt", lev))),
        "max_pident_mean": float(np.mean(vals("max_pident", lev))),
        "len_mean": float(np.mean(vals("len", lev))),
        "len_sd": float(np.std(vals("len", lev))),
        "resolution_mean": float(np.mean(vals("resolution", lev))),
        "resolved_frac_mean": float(np.mean(vals("resolved_frac", lev))),
        "lcfrac_mean": float(np.mean(vals("lcfrac", lev))),
        "baseline_n": len(btm),
        "baseline_tm_mean": float(np.mean(btm)) if btm else None,
        "model_tm_mean_on_baseline_covered": float(np.mean(mtm)) if mtm else None,
    }

# trend + A1 vs A4
obs, p_jt = jt_perm(groups)
a1, a4 = vals("tm", "A1"), vals("tm", "A4")
tt = stats.ttest_ind(a1, a4, equal_var=False)
d = (np.mean(a1) - np.mean(a4)) / math.sqrt((np.var(a1) + np.var(a4)) / 2)
summary["jt_obs"] = obs
summary["jt_p_perm"] = p_jt
summary["welch_A1_A4"] = {"t": float(tt.statistic), "p": float(tt.pvalue)}
summary["cohens_d_A1_A4"] = float(d)
summary["gap_A1_A4"] = float(np.mean(a1) - np.mean(a4))
summary["monotonic_means"] = all(
    summary["levels"][LEVELS[i]]["tm_mean"] > summary["levels"][LEVELS[i + 1]]["tm_mean"]
    for i in range(3))

# length-standardized level means (bins 40-150, 150-250, 250-400)
BINS = [(40, 150), (150, 250), (250, 401)]
std = {}
for lev in LEVELS:
    bm = []
    for lo, hi in BINS:
        v = [r["tm"] for r in panel.values()
             if r["levelA"] == lev and lo <= r["len"] < hi]
        bm.append(float(np.mean(v)) if v else None)
    std[lev] = bm
summary["length_std_tm_means"] = std
valid = [b for b in range(3) if all(std[l][b] is not None for l in LEVELS)]
summary["length_std_avg"] = {l: float(np.mean([std[l][b] for b in valid])) for l in LEVELS}

# gate: degradation deltas
lv = summary["levels"]
delta_model = lv["A1"]["tm_mean"] - lv["A4"]["tm_mean"]
# baseline delta only on levels with coverage: A1..A3 (A4 uncovered by def)
cov_levels = [l for l in LEVELS if lv[l]["baseline_n"] >= 20]
summary["gate"] = {
    "delta_model_A1_A4": delta_model,
    "baseline_covered_levels": cov_levels,
}
if len(cov_levels) >= 2:
    first, last = cov_levels[0], cov_levels[-1]
    summary["gate"]["delta_baseline_%s_%s" % (first, last)] = (
        lv[first]["baseline_tm_mean"] - lv[last]["baseline_tm_mean"])
    summary["gate"]["delta_model_on_covered_%s_%s" % (first, last)] = (
        lv[first]["model_tm_mean_on_baseline_covered"] -
        lv[last]["model_tm_mean_on_baseline_covered"])

# Axis B (exploratory if underpowered)
b = defaultdict(list)
for r in panel.values():
    b[r["levelB"]].append(r["tm"])
summary["axisB"] = {k: {"n": len(v), "tm_mean": float(np.mean(v))}
                    for k, v in sorted(b.items())}

json.dump(summary, open("results/spc_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
