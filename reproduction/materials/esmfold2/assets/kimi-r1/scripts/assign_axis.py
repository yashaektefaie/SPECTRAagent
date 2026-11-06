#!/usr/bin/env python3
"""Assign frozen axis (max identity to pre-cutoff corpus) + spectral levels,
then stratified-sample the powered confirmation set.

Frozen rule: hit filter = max(qcov,tcov) >= 0.5; identity = max pident among
passing hits; no passing hit => 0. Template (for baseline) = argmax pident hit
(ties: coverage, then evalue).
"""
import csv, random, collections

cov_min = 0.5
best = {}   # query -> (pident, cov, evalue, target)
for row in csv.reader(open("results/test_vs_train.m8"), delimiter="\t"):
    q, t, pident, qcov, tcov, evalue = row[0], row[1], float(row[2]), float(row[3]), float(row[4]), float(row[5])
    cov = max(qcov, tcov)
    if cov < cov_min:
        continue
    cur = best.get(q)
    key = (pident, cov, -evalue)
    if cur is None or key > cur[0]:
        best[q] = (key, t, pident, cov, evalue)

# all cluster reps (from rep_seq fasta)
reps = [l[1:].strip() for l in open("results/testclu_rep_seq.fasta") if l.startswith(">")]
lengths = {r[0]: int(r[2]) for r in csv.reader(open("data/test_pool_meta.tsv"), delimiter="\t") if r[0] != "entity"}

rows = []
for q in reps:
    if q in best:
        _, t, pid, cov, ev = best[q]
    else:
        t, pid, cov, ev = "NONE", 0.0, 0.0, -1.0
    level = "L1" if pid >= 80 else "L2" if pid >= 50 else "L3" if pid >= 25 else "L4"
    rows.append({"unit": q, "identity": pid, "cov": cov, "template": t,
                 "evalue": ev, "level": level, "length": lengths.get(q, -1)})

with open("results/prospective_similarity.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader(); w.writerows(rows)

counts = collections.Counter(r["level"] for r in rows)
print("level counts (all reps):", dict(counts))

# powered stratified sample: 40 per level (frozen target)
random.seed(20260722)
sample = []
for lev in ["L1", "L2", "L3", "L4"]:
    pool = [r for r in rows if r["level"] == lev]
    random.shuffle(pool)
    take = pool[:40]
    sample.extend(take)
    print(f"{lev}: available {len(pool)}, sampled {len(take)}")

with open("results/confirmation_set.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader(); w.writerows(sample)
print(f"confirmation set: {len(sample)}")
