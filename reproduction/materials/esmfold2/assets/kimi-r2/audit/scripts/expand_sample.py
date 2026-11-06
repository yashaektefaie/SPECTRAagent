#!/usr/bin/env python
"""Expand the confirmation sample to n per bin under the frozen rule
(cluster reps, round-robin length strata, seed 20260722), excluding the pilot."""
import csv, random, collections, sys

n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 40
random.seed(20260723)  # expansion seed (pilot seed was 20260722)

rows = list(csv.DictReader(open("audit/data/axis_table.csv")))
pilot = set()
for r in csv.DictReader(open("audit/data/sample_plan.csv")):
    pilot.add(r["entity"])

bycell = collections.defaultdict(list)
for r in rows:
    if r["rep"] == "True" and r["entity"] not in pilot:
        bycell[(r["bin"], r["strata"])].append(r["entity"])

plan = []
for b in ["I1_ge80", "I2_40_80", "I3_20_40", "I4_lt20"]:
    cells = []
    for s in "SML":
        c = sorted(bycell[(b, s)])
        random.shuffle(c)
        cells.append(c)
    pool, idx = [], [0, 0, 0]
    need = n_target - sum(1 for e in pilot if any(
        r2["entity"] == e and r2["bin"] == b for r2 in rows))
    while len(pool) < need:
        progressed = False
        for s_i, c in enumerate(cells):
            if idx[s_i] < len(c) and len(pool) < need:
                pool.append(c[idx[s_i]]); idx[s_i] += 1; progressed = True
        if not progressed:
            break
    plan.extend((b, e) for e in pool)

with open("audit/data/sample_plan_expansion.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["bin", "entity"]); w.writerows(plan)
print("expansion units:", len(plan))
cnt = collections.Counter(b for b, _ in plan)
print(cnt)
