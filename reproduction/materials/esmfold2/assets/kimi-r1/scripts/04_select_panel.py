#!/usr/bin/env python
"""Compute frozen Axis-A value per cluster representative and the stratified
panel selection. Axis A = max pident among train-pool hits with qcov>=0.5
(implementation detail of the frozen spec, recorded here pre-scoring).
Also computes frozen Axis-B feature (low-complexity fraction) for later use.
Writes:
  data/rep_features.tsv     rep, level A, max_pident, qcov, lcfrac, level B, len
  data/panel.tsv            selected panel (80/level) + reserves (25/level)
"""
import math, random
from collections import defaultdict

def lc_fraction(seq, win=12, thr=2.2):
    n = len(seq)
    if n < win:
        return 0.0
    mask = [0] * n
    for i in range(n - win + 1):
        w = seq[i:i + win]
        from collections import Counter
        c = Counter(w)
        H = -sum((v / win) * math.log2(v / win) for v in c.values())
        if H < thr:
            for j in range(i, i + win):
                mask[j] = 1
    return sum(mask) / n

def level_a(p):
    if p >= 80: return "A1"
    if p >= 50: return "A2"
    if p >= 30: return "A3"
    return "A4"

def level_b(f):
    if f < 0.05: return "B1"
    if f < 0.10: return "B2"
    if f < 0.20: return "B3"
    return "B4"

# sequences of reps
seqs = {}
name = None
for line in open("results/testclu_rep_seq.fasta"):
    if line.startswith(">"):
        name = line[1:].strip()
        seqs[name] = ""
    else:
        seqs[name] += line.strip()

# best hit with qcov>=0.5 (primary), and best unrestricted (audit)
best = defaultdict(lambda: (0.0, 0.0))   # rep -> (pident, qcov)
best_any = defaultdict(float)
with open("results/rep_vs_train.m8") as f:
    for line in f:
        q, t, pident, alnlen, qlen, qs, qe, ts, te, cigar, bits = line.split("\t")
        pident = float(pident); qlen = float(qlen)
        qcov = (int(qe) - int(qs) + 1) / qlen
        if pident > best_any[q]:
            best_any[q] = pident
        if qcov >= 0.5 and pident > best[q][0]:
            best[q] = (pident, qcov)

rows = []
for rep, seq in seqs.items():
    p, qc = best.get(rep, (0.0, 0.0))
    lcf = lc_fraction(seq)
    rows.append({"rep": rep, "len": len(seq), "max_pident": p, "qcov": qc,
                 "best_any": best_any.get(rep, 0.0), "lcfrac": lcf,
                 "levelA": level_a(p), "levelB": level_b(lcf)})

with open("data/rep_features.tsv", "w") as f:
    f.write("rep\tlen\tmax_pident\tqcov\tbest_any_pident\tlcfrac\tlevelA\tlevelB\n")
    for r in rows:
        f.write("%s\t%d\t%.1f\t%.3f\t%.1f\t%.4f\t%s\t%s\n" % (
            r["rep"], r["len"], r["max_pident"], r["qcov"], r["best_any"],
            r["lcfrac"], r["levelA"], r["levelB"]))

cnt = defaultdict(int)
for r in rows:
    cnt[r["levelA"]] += 1
    cnt[r["levelB"]] += 1
print("level counts:", dict(sorted(cnt.items())))

# stratified panel: per level A, length-matched across bins
BINS = [(40, 150), (150, 250), (250, 401)]
random.seed(20260722)
panel = []
for lev in ["A1", "A2", "A3", "A4"]:
    pool = [r for r in rows if r["levelA"] == lev]
    random.shuffle(pool)
    # length-matched: equal target per bin (approx), within bin random order
    need, res = 80, 25
    sel, reserve = [], []
    per_bin = need // len(BINS) + 1
    for lo, hi in BINS:
        binpool = [r for r in pool if lo <= r["len"] < hi]
        sel.extend(binpool[:per_bin])
        reserve.extend(binpool[per_bin:per_bin + res // len(BINS) + 1])
    panel += [(r["rep"], lev, "panel") for r in sel[:need]]
    panel += [(r["rep"], lev, "reserve") for r in reserve[:res]]

with open("data/panel.tsv", "w") as f:
    f.write("rep\tlevelA\trole\n")
    for rep, lev, role in panel:
        f.write("%s\t%s\t%s\n" % (rep, lev, role))
print("panel+reserve:", len(panel))
