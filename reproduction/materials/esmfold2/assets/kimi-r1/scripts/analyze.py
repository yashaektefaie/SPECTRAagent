#!/usr/bin/env python3
"""SPECTRA analysis: SPC construction, trend test, gate call, confound checks."""
import csv, collections
import numpy as np
from scipy import stats

def load(path):
    d = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        if r["status"] == "ok":
            d[r["unit"]] = {k: float(r[k]) for k in ("tm_score", "rmsd", "gdt_ts")}
    return d

conf = {r["unit"]: r for r in csv.DictReader(open("results/confirmation_set.tsv"), delimiter="\t")}
esmf = load("results/per_unit_scores.tsv")
base = load("results/baseline_scores.tsv")
levels = ["L1", "L2", "L3", "L4"]

def summarize(scores, metric="gdt_ts"):
    print(f"  level  n   mean {metric}  sd    median")
    means = {}
    for lev in levels:
        v = [scores[u][metric] for u in conf if conf[u]["level"] == lev and u in scores]
        if v:
            means[lev] = (np.mean(v), np.std(v), np.median(v), len(v))
            print(f"  {lev}   {len(v):3d}  {np.mean(v):7.2f}  {np.std(v):5.1f}  {np.median(v):6.2f}")
    return means

print("== ESMFold (n=160) ==")
em = summarize(esmf)
print("== Template-copy baseline (built models only) ==")
bm = summarize(base)

# abstention per level
print("\n== Baseline abstentions per level ==")
for lev in levels:
    units = [u for u in conf if conf[u]["level"] == lev]
    nab = sum(1 for u in units if u not in base)
    print(f"  {lev}: {nab}/{len(units)} abstained")

# Jonckheere-Terpstra trend (ESMFold gdt_ts across levels L1>L2>L3>L4 expected)
groups = [[esmf[u]["gdt_ts"] for u in conf if conf[u]["level"] == lev] for lev in levels]
# JT statistic (manual, descending alternative)
def jonckheere(groups):
    n_groups = len(groups)
    S = 0.0
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            for a in groups[i]:
                for b in groups[j]:
                    S += (a > b) - (a < b)
    # normal approximation with ties neglected (fine here)
    N = sum(len(g) for g in groups)
    var = (N * N * (2 * N + 3) - sum(len(g) ** 2 * (2 * len(g) + 3) for g in groups)) / 72
    z = S / np.sqrt(var)
    return S, z, 1 - stats.norm.cdf(z)
S, z, p = jonckheere(groups)
print(f"\nJonckheere-Terpstra (descending): S={S:.0f} z={z:.2f} p={p:.2e}")

d14 = em["L1"][0] - em["L4"][0]
print(f"Delta GDT-TS L1-L4 (ESMFold): {d14:.2f}")

# bootstrap CI for delta L1-L4
rng = np.random.default_rng(0)
g1, g4 = np.array(groups[0]), np.array(groups[3])
boots = [rng.choice(g1, len(g1), True).mean() - rng.choice(g4, len(g4), True).mean() for _ in range(10000)]
print(f"bootstrap 95% CI: [{np.percentile(boots,2.5):.1f}, {np.percentile(boots,97.5):.1f}]")

# gate: paired comparison on units where baseline produced a model
print("\n== GATE: paired per-level means (units with baseline model) ==")
for lev in levels:
    units = [u for u in conf if conf[u]["level"] == lev and u in base and u in esmf]
    if not units:
        print(f"  {lev}: no paired units"); continue
    e = np.mean([esmf[u]["gdt_ts"] for u in units])
    b = np.mean([base[u]["gdt_ts"] for u in units])
    print(f"  {lev}: n={len(units):3d}  ESMFold {e:6.2f}  baseline {b:6.2f}  diff {e-b:+6.2f}")

paired = [u for u in conf if u in base and u in esmf]
lev_of = {u: levels.index(conf[u]["level"]) for u in paired}
e_arr = np.array([esmf[u]["gdt_ts"] for u in paired])
b_arr = np.array([base[u]["gdt_ts"] for u in paired])
l_arr = np.array([lev_of[u] for u in paired])
se = stats.linregress(l_arr, e_arr)
sb = stats.linregress(l_arr, b_arr)
print(f"\nslope vs level index (0=L1..3=L4): ESMFold {se.slope:.2f}/level (p={se.pvalue:.1e}), "
      f"baseline {sb.slope:.2f}/level (p={sb.pvalue:.1e})")

# baseline-as-floor treatment: abstained baseline = structural null (report only)
print("\n== Confound checks ==")
for lev in levels:
    v = [int(conf[u]["length"]) for u in conf if conf[u]["level"] == lev]
    print(f"  {lev}: length mean {np.mean(v):.0f}")
# length-adjusted: regression of gdt_ts on level + length
import numpy.linalg as la
X = np.array([[1, levels.index(conf[u]["level"]), int(conf[u]["length"])] for u in esmf])
y = np.array([esmf[u]["gdt_ts"] for u in esmf])
beta, res_, rank, sv = la.lstsq(X, y, rcond=None)
# se of beta
yhat = X @ beta
dof = len(y) - 3
sigma2 = np.sum((y - yhat) ** 2) / dof
covb = sigma2 * la.inv(X.T @ X)
t_lev = beta[1] / np.sqrt(covb[1, 1])
p_lev = 2 * (1 - stats.t.cdf(abs(t_lev), dof))
print(f"  OLS gdt_ts ~ level + length: level coef {beta[1]:.2f}/level (p={p_lev:.1e}), "
      f"length coef {beta[2]:.4f}")
