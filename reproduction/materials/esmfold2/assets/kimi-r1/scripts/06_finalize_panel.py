#!/usr/bin/env python
"""Finalize panel: resolved-fraction >= 70% filter, backfill from reserves
(same level, same length-bin order as selected). Writes
results/panel_final.tsv and results/panel_input.fasta (ESMFold inputs).
Also sanity-checks CIF-derived chain sequence vs pdb_seqres sequence."""
import numpy as np

# pdb_seqres sequences (used for clustering/axis)
ps = {}
name = None
for line in open("data/test_one_per_entry.fasta"):
    if line.startswith(">"):
        name = line[1:].strip(); ps[name] = ""
    else:
        ps[name] += line.strip()

feats = {}
for line in open("data/rep_features.tsv").read().splitlines()[1:]:
    p = line.split("\t")
    feats[p[0]] = {"len": int(p[1]), "max_pident": float(p[2]),
                   "lcfrac": float(p[5]), "levelA": p[6], "levelB": p[7]}

rows = []
for line in open("data/panel.tsv").read().splitlines()[1:]:
    rep, lev, role = line.split("\t")
    z = np.load("results/native/%s.npz" % rep, allow_pickle=False)
    seq = str(z["seq"]); res_idx = z["res_idx"]
    rf = len(res_idx) / len(seq)
    match = (seq == ps.get(rep))
    rows.append({"rep": rep, "lev": lev, "role": role, "rf": rf,
                 "seq": seq, "seqmatch": match})

n_mismatch = sum(1 for r in rows if not r["seqmatch"])
print("sequence mismatches (CIF vs pdb_seqres):", n_mismatch)
for r in rows:
    if not r["seqmatch"]:
        print("  MISMATCH", r["rep"], len(r["seq"]), len(ps.get(r["rep"], "")))

final, used = [], set()
for lev in ["A1", "A2", "A3", "A4"]:
    panel = [r for r in rows if r["lev"] == lev and r["role"] == "panel"]
    reserve = [r for r in rows if r["lev"] == lev and r["role"] == "reserve"]
    good = [r for r in panel if r["rf"] >= 0.70 and r["seqmatch"]]
    need = 80 - len(good)
    if need > 0:
        backfill = [r for r in reserve if r["rf"] >= 0.70 and r["seqmatch"]][:need]
        good += backfill
        print("level", lev, ": panel survivors", len(panel) - need + len(backfill) - len(backfill),
          "| used", len(backfill), "backfills")
    else:
        print("level", lev, ": no backfill needed")
    for r in good[:80]:
        final.append(r)

with open("results/panel_final.tsv", "w") as f:
    f.write("rep\tlevelA\tlevelB\tlen\tmax_pident\tlcfrac\tresolved_frac\n")
    for r in final:
        ft = feats[r["rep"]]
        f.write("%s\t%s\t%s\t%d\t%.1f\t%.4f\t%.3f\n" % (
            r["rep"], ft["levelA"], ft["levelB"], len(r["seq"]),
            ft["max_pident"], ft["lcfrac"], r["rf"]))
with open("results/panel_input.fasta", "w") as f:
    for r in final:
        f.write(">%s\n%s\n" % (r["rep"], r["seq"]))
print("final panel size:", len(final))
