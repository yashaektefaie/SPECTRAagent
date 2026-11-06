"""Development discovery screen (uses dev outcomes - exploration only).

For each input-side feature:
  - point-biserial correlation with CONCH correctness (marginal)
  - class-stratified (within-class) mean correlation, to separate the axis from
    tissue-class confounding (accuracy varies strongly by class)
  - accuracy across quintile bins (marginal) and the monotone trend
Also prints feature-feature correlations (e.g. sharpness vs stain) for confound
screening. Nothing here is frozen; this only informs axis selection.
"""
import csv
import numpy as np
from collections import defaultdict
from scipy import stats

FEATS = ["sharpness_varlap", "sharpness_tenengrad", "highfreq_frac", "brightness",
         "contrast_std", "saturation_mean", "colorfulness", "tissue_frac",
         "hematoxylin_mean", "eosin_mean", "he_ratio", "entropy"]

rows = []
with open("results/dev_features.csv") as f:
    for r in csv.DictReader(f):
        rows.append(r)

y = np.array([int(r["csv_correct"]) for r in rows])
cls = np.array([r["label_name"] for r in rows])
X = {k: np.array([float(r[k]) for r in rows]) for k in FEATS}
N = len(y)
print(f"N={N}  overall acc={y.mean():.4f}\n")

print("=== marginal + class-stratified point-biserial corr with correctness ===")
print(f"{'feature':20s} {'r_marg':>8s} {'p_marg':>10s} {'r_within':>9s}")
results = {}
for k in FEATS:
    x = X[k]
    r_marg, p_marg = stats.pointbiserialr(y, x)
    # within-class: average correlation weighted by class n, using standardized x per class
    rs = []; ws = []
    for c in np.unique(cls):
        m = cls == c
        if m.sum() < 30 or y[m].std() == 0:
            continue
        xc = x[m]
        if xc.std() == 0:
            continue
        rc, _ = stats.pointbiserialr(y[m], xc)
        rs.append(rc); ws.append(m.sum())
    r_within = float(np.average(rs, weights=ws)) if rs else float("nan")
    results[k] = {"r_marg": float(r_marg), "p_marg": float(p_marg), "r_within": r_within}
    print(f"{k:20s} {r_marg:8.4f} {p_marg:10.2e} {r_within:9.4f}")

print("\n=== quintile accuracy (marginal) for top features by |r_within| ===")
order = sorted(FEATS, key=lambda k: -abs(results[k]["r_within"]))
for k in order[:6]:
    x = X[k]
    qs = np.quantile(x, [0, .2, .4, .6, .8, 1.0])
    print(f"\n{k}  (r_marg={results[k]['r_marg']:.3f}, r_within={results[k]['r_within']:.3f})")
    accs = []
    for i in range(5):
        lo, hi = qs[i], qs[i + 1]
        if i < 4:
            m = (x >= lo) & (x < hi)
        else:
            m = (x >= lo) & (x <= hi)
        accs.append(y[m].mean())
        print(f"  Q{i+1} [{lo:9.3f},{hi:9.3f}) n={m.sum():5d} acc={y[m].mean():.4f}")
    mono = all(accs[i] <= accs[i+1] for i in range(4)) or all(accs[i] >= accs[i+1] for i in range(4))
    print(f"  monotone={mono}  range={max(accs)-min(accs):.3f}")

print("\n=== feature-feature |corr| (confound screen, sharpness vs stain etc) ===")
key = ["sharpness_varlap", "saturation_mean", "colorfulness", "hematoxylin_mean", "eosin_mean", "brightness", "tissue_frac", "contrast_std"]
print("        " + " ".join(f"{k[:7]:>7s}" for k in key))
for a in key:
    line = f"{a[:7]:>7s} "
    for b in key:
        rr = np.corrcoef(X[a], X[b])[0, 1]
        line += f"{rr:7.2f} "
    print(line)

# class composition across sharpness quintiles (confound check)
print("\n=== class fraction across sharpness_varlap quintiles ===")
x = X["sharpness_varlap"]
qs = np.quantile(x, [0, .2, .4, .6, .8, 1.0])
classes = sorted(np.unique(cls))
print("quintile " + " ".join(f"{c:>5s}" for c in classes))
for i in range(5):
    lo, hi = qs[i], qs[i + 1]
    m = (x >= lo) & (x <= hi) if i == 4 else (x >= lo) & (x < hi)
    fr = [ (cls[m] == c).mean() for c in classes ]
    print(f"Q{i+1}      " + " ".join(f"{v:5.2f}" for v in fr))
