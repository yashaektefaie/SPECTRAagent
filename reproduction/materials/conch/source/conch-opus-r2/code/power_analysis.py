"""Power / sample-size calculation from the development blur pilot.

Primary endpoint: top-1 accuracy drop of CONCH from sigma=0 to sigma=max on a
class-balanced sample (paired over tiles). We size the confirmation sample so the
two frozen decisions are adequately powered:

  Test A (real degradation): paired McNemar on CONCH correct(sigma0) vs
    correct(sigma_max). Effect = discordant-pair rate.
  Test B (gate / not-robustness): difference-in-differences of the accuracy drop,
    CONCH vs baseline. (Baseline pilot drop is estimated separately; here we size
    mainly on Test A, which is the within-model degradation, then verify Test B
    is feasible at that N.)

Reports n per class and total for 80% and 90% power at alpha=0.05 two-sided.
Deterministic; reads only development pilot outcomes (exploration).
"""
import csv, json
import numpy as np
from collections import defaultdict
from math import ceil
from scipy import stats

rows = list(csv.DictReader(open("results/pilot_blur_conch.csv")))
sigmas = sorted({float(r["sigma"]) for r in rows})
smax = max(sigmas)

# per-tile correctness by sigma
by_tile = defaultdict(dict)
for r in rows:
    by_tile[r["sample_id"]][float(r["sigma"])] = int(r["correct"])

tiles = list(by_tile.keys())
c0 = np.array([by_tile[t][0.0] for t in tiles])
cmax = np.array([by_tile[t][smax] for t in tiles])

acc0 = c0.mean(); accm = cmax.mean()
drop = acc0 - accm
# discordant pairs for McNemar
b = int(np.sum((c0 == 1) & (cmax == 0)))   # correct->wrong
cc = int(np.sum((c0 == 0) & (cmax == 1)))  # wrong->correct
n = len(tiles)
print(f"PILOT n={n}  acc(sigma0)={acc0:.4f}  acc(sigma{smax})={accm:.4f}  drop={drop:.4f}")
print(f"discordant: correct->wrong b={b}  wrong->correct c={cc}  (net={b-cc})")
print("per-level acc:")
for s in sigmas:
    a = np.mean([by_tile[t][s] for t in tiles])
    print(f"  sigma={s}: acc={a:.4f}")

# McNemar power sizing: effect = difference in discordant proportions.
# p_discordant and p_b given pilot
p_b = b / n           # prob correct->wrong
p_c = cc / n          # prob wrong->correct
pdisc = p_b + p_c
# Under McNemar, test statistic ~ depends on (p_b - p_c) and pdisc.
# Use normal approx: effect size for the difference of proportions in discordant.
# Required n for McNemar (Connor 1987 approx):
za = stats.norm.ppf(1 - 0.05 / 2)
def mcnemar_n(power):
    zb = stats.norm.ppf(power)
    if pdisc <= 0 or abs(p_b - p_c) < 1e-9:
        return float("inf")
    psi = p_b / pdisc  # proportion of discordant that are correct->wrong
    # Connor: n_discordant = (z_a*? ) ... use formula for n total:
    num = (za * np.sqrt(pdisc) + zb * np.sqrt(pdisc - (p_b - p_c) ** 2)) ** 2
    den = (p_b - p_c) ** 2
    return num / den

for pw in (0.80, 0.90):
    ntot = mcnemar_n(pw)
    print(f"McNemar required total n (power={pw}): {ntot:.0f}  -> per class {ceil(ntot/9)}")

# Simple paired-difference sizing on accuracy drop (bootstrap SE on drop)
rng = np.random.default_rng(0)
drops = []
for _ in range(3000):
    idx = rng.integers(0, n, n)
    drops.append(c0[idx].mean() - cmax[idx].mean())
se_drop_pilot = np.std(drops)
print(f"pilot drop={drop:.4f}  bootstrap SE(drop) at n={n}: {se_drop_pilot:.4f}")
# SE scales as 1/sqrt(n). To get drop detectable (CI excludes 0) with power:
# required SE so that drop/SE = za+zb
for pw in (0.80, 0.90):
    zb = stats.norm.ppf(pw)
    need_se = drop / (za + zb)
    # se_drop_pilot corresponds to n; need_se -> n_req = n*(se_pilot/need_se)^2
    n_req = n * (se_drop_pilot / need_se) ** 2
    print(f"accuracy-drop test required total n (power={pw}): {ceil(n_req)} -> per class {ceil(n_req/9)}")

out = {"pilot_n": n, "acc_sigma0": acc0, "acc_sigmamax": accm, "drop": drop,
       "sigma_max": smax, "discordant_b_correct_to_wrong": b,
       "discordant_c_wrong_to_correct": cc,
       "per_level_acc": {str(s): float(np.mean([by_tile[t][s] for t in tiles])) for s in sigmas},
       "mcnemar_n_total": {"0.80": mcnemar_n(0.80), "0.90": mcnemar_n(0.90)}}
json.dump(out, open("results/power_pilot.json", "w"), indent=2)
print("WROTE results/power_pilot.json")
