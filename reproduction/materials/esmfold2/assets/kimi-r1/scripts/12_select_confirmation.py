#!/usr/bin/env python
"""Select the FROZEN confirmation set for Axis A' (orphan vs non-orphan),
from cluster reps disjoint from the discovery panel AND its reserves.
n=170/group, equal per length bin. Writes results/confirm/panel.tsv +
results/confirm/input_candidates.fasta
"""
import random, os

os.makedirs("results/confirm", exist_ok=True)
BINS = [(40, 150), (150, 250), (250, 401)]
N_GROUP = 170

used = set()
for line in open("data/panel.tsv").read().splitlines()[1:]:
    used.add(line.split("\t")[0])

rows = []
for line in open("data/rep_features.tsv").read().splitlines()[1:]:
    p = line.split("\t")
    rep = p[0]
    if rep in used:
        continue
    rows.append({"rep": rep, "len": int(p[1]), "max_pident": float(p[2])})

seqs = {}
name = None
for line in open("results/testclu_rep_seq.fasta"):
    if line.startswith(">"):
        name = line[1:].strip(); seqs[name] = ""
    else:
        seqs[name] += line.strip()

random.seed(20260723)
sel = []
N_RESERVE = 30
per_bin = N_GROUP // 3  # 56,56,58
for grp, cond in [("orphan", lambda p: p < 30), ("nonorphan", lambda p: p >= 30)]:
    pool = [r for r in rows if cond(r["max_pident"])]
    random.shuffle(pool)
    got = 0
    for i, (lo, hi) in enumerate(BINS):
        target = per_bin + (1 if i < N_GROUP % 3 else 0)
        binpool = [r for r in pool if lo <= r["len"] < hi]
        for r in binpool[:target]:
            sel.append((r["rep"], grp, "primary"))
            got += 1
        rsv = binpool[target:target + N_RESERVE // 3 + 1]
        for r in rsv:
            sel.append((r["rep"], grp, "reserve"))
    print(grp, "selected", got, "from pool", len(pool))

with open("results/confirm/panel.tsv", "w") as f:
    f.write("rep\tgroup\trole\n")
    for rep, grp, role in sel:
        f.write("%s\t%s\t%s\n" % (rep, grp, role))
with open("results/confirm/input_candidates.fasta", "w") as f:
    for rep, grp, role in sel:
        f.write(">%s\n%s\n" % (rep, seqs[rep]))
print("total", len(sel))
