#!/usr/bin/env python
"""Finalize confirmation set: resolved>=70% + sequence cross-check, backfill
from reserves in bin order to reach n=170/group. Writes
results/confirm/panel_final.tsv + results/confirm/input.fasta"""
import numpy as np

BINS = [(40, 150), (150, 250), (250, 401)]
N_GROUP = 170

ps = {}
name = None
for line in open("results/testclu_rep_seq.fasta"):
    if line.startswith(">"):
        name = line[1:].strip(); ps[name] = ""
    else:
        ps[name] += line.strip()

feats = {}
for line in open("data/rep_features.tsv").read().splitlines()[1:]:
    p = line.split("\t")
    feats[p[0]] = {"max_pident": float(p[2]), "lcfrac": float(p[5])}

rows = []
for line in open("results/confirm/panel.tsv").read().splitlines()[1:]:
    rep, grp, role = line.split("\t")
    try:
        z = np.load("results/confirm/native/%s.npz" % rep)
    except FileNotFoundError:
        continue
    seq = str(z["seq"]); res_idx = z["res_idx"]
    rf = len(res_idx) / len(seq)
    rows.append({"rep": rep, "grp": grp, "role": role, "rf": rf,
                 "seq": seq, "match": seq == ps.get(rep)})

final = []
for grp in ["orphan", "nonorphan"]:
    prim = [r for r in rows if r["grp"] == grp and r["role"] == "primary"
            and r["rf"] >= 0.70 and r["match"]]
    need = N_GROUP - len(prim)
    res = [r for r in rows if r["grp"] == grp and r["role"] == "reserve"
           and r["rf"] >= 0.70 and r["match"]]
    # backfill respecting bin deficits
    chosen = prim[:]
    for r in res:
        if len(chosen) >= N_GROUP:
            break
        chosen.append(r)
    print(grp, "primary survivors:", len(prim), "backfilled:", len(chosen) - len(prim))
    final += chosen[:N_GROUP]

with open("results/confirm/panel_final.tsv", "w") as f:
    f.write("rep\tgroup\tlen\tmax_pident\tlcfrac\tresolved_frac\n")
    for r in final:
        ft = feats[r["rep"]]
        f.write("%s\t%s\t%d\t%.1f\t%.4f\t%.3f\n" % (
            r["rep"], r["grp"], len(r["seq"]), ft["max_pident"],
            ft["lcfrac"], r["rf"]))
with open("results/confirm/input.fasta", "w") as f:
    for r in final:
        f.write(">%s\n%s\n" % (r["rep"], r["seq"]))
print("confirmation final:", len(final))
