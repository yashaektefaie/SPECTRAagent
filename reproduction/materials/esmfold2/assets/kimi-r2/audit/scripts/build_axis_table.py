#!/usr/bin/env python
"""Assign axis values, verify split, and draw the stratified confirmation sample.

Axis value = max fident (fraction identical positions) over MMseqs2 hits of the
test entity vs the pre-2021-09-30 PDB pool; 0.0 if no hit (search run with
-s 7.5 --cov-mode 0 -c 0.8).
Only cluster representatives (40% internal clustering) are eligible.
Length strata: S=40-120, M=121-250, L=251-400.
Outputs audit/data/axis_table.csv and audit/data/sample_plan.csv.
"""
import csv, json, sys, random, collections

W = "audit/work_mmseqs"
random.seed(20260722)

# lengths / chains
meta = {}
for line in open("audit/data/test_candidates.jsonl"):
    d = json.loads(line)
    meta[d["entity"]] = d

# cluster representatives
reps = set()
hdr = None
for line in open(f"{W}/clus_rep_seq.fasta"):
    if line.startswith(">"):
        reps.add(line[1:].split()[0].strip())

# axis values
best = {}
for row in csv.reader(open(f"{W}/res.tsv"), delimiter="\t"):
    if len(row) < 3:
        continue
    q, t, fid = row[0], row[1], float(row[2])
    if q not in meta:
        continue
    if fid > best.get(q, (0, None))[0]:
        best[q] = (fid, t)

def bin_of(x):
    if x >= 0.80: return "I1_ge80"
    if x >= 0.40: return "I2_40_80"
    if x >= 0.20: return "I3_20_40"
    return "I4_lt20"

rows = []
for eid, m in meta.items():
    fid, tgt = best.get(eid, (0.0, None))
    L = m["length"]
    strata = "S" if L <= 120 else ("M" if L <= 250 else "L")
    rows.append({"entity": eid, "entry": m["entry"], "length": L,
                 "strata": strata, "rep": eid in reps,
                 "ident": round(fid, 4), "hit": tgt or "",
                 "bin": bin_of(fid)})

with open("audit/data/axis_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

cnt = collections.Counter((r["bin"], r["rep"]) for r in rows)
print("bin counts (all / reps):")
for b in ["I1_ge80", "I2_40_80", "I3_20_40", "I4_lt20"]:
    print(f"  {b}: all={cnt[(b,False)]+cnt[(b,True)]} reps={cnt[(b,True)]}")

# stratified sampling plan: per bin x strata counts among reps
bycell = collections.defaultdict(list)
for r in rows:
    if r["rep"]:
        bycell[(r["bin"], r["strata"])].append(r["entity"])
print("reps per bin x strata:")
for b in ["I1_ge80", "I2_40_80", "I3_20_40", "I4_lt20"]:
    print(" ", b, {s: len(bycell[(b, s)]) for s in "SML"})

if "--sample" in sys.argv:
    n_pilot = int(sys.argv[sys.argv.index("--sample") + 1])
    plan = []
    for b in ["I1_ge80", "I2_40_80", "I3_20_40", "I4_lt20"]:
        cells = [sorted(bycell[(b, s)]) for s in "SML"]
        # round-robin across strata for balance
        pool = []
        for c in cells:
            random.shuffle(c)
        idx = [0, 0, 0]
        while len(pool) < n_pilot:
            progressed = False
            for s_i, c in enumerate(cells):
                if idx[s_i] < len(c) and len(pool) < n_pilot:
                    pool.append(c[idx[s_i]]); idx[s_i] += 1; progressed = True
            if not progressed:
                break
        plan.extend((b, e) for e in pool)
    with open("audit/data/sample_plan.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bin", "entity"]); w.writerows(plan)
    print("sample plan:", len(plan))
