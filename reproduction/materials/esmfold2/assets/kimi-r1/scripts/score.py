#!/usr/bin/env python3
"""Score models vs native chains with TM-align (tmtools bindings).

For each unit: TM-score (normalized by native length), RMSD, and GDT-TS
(mean fraction of aligned residue pairs within 1/2/4/8 A after the TM-align
superposition). Identical pipeline for ESMFold predictions and baseline.
"""
import csv, os, sys
import numpy as np
from tmtools import tm_align

THREE2ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}

def read_ca(path):
    coords, seq = [], []
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        resn = line[17:20].strip()
        seq.append(THREE2ONE.get(resn, "X"))
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords), "".join(seq)

def gdt_ts(dists, cutoffs=(1.0, 2.0, 4.0, 8.0)):
    if len(dists) == 0:
        return 0.0
    return float(np.mean([np.mean(dists <= c) for c in cutoffs]) * 100.0)

def score(model_path, native_path):
    nc, ns = read_ca(native_path)
    mc, ms = read_ca(model_path)
    res = tm_align(nc, mc, ns, ms)
    # tmtools u,t superpose chain1 (native) onto chain2 (model)
    nc_sup = (np.array(res.u) @ nc.T).T + np.array(res.t)
    # aligned pairs from the alignment strings
    xi, yi = -1, -1
    dists = []
    for a, b in zip(res.seqxA, res.seqyA):
        if a != "-":
            xi += 1
        if b != "-":
            yi += 1
        if a != "-" and b != "-":
            dists.append(np.linalg.norm(nc_sup[xi] - mc[yi]))
    dists = np.array(dists)
    return {
        "tm_score": float(res.tm_norm_chain1),
        "rmsd": float(res.rmsd),
        "gdt_ts": gdt_ts(dists),
        "n_aligned": int(len(dists)),
        "n_native_ca": int(len(nc)),
        "n_model_ca": int(len(mc)),
    }

which = sys.argv[1]           # pred | baseline
outpath = sys.argv[2]         # results/per_unit_scores.tsv | results/baseline_scores.tsv
rows = list(csv.DictReader(open(sys.argv[3] if len(sys.argv)>3 else "results/confirmation_set.tsv"), delimiter="\t"))
out = open(outpath, "w")
out.write("unit\ttm_score\trmsd\tgdt_ts\tn_aligned\tn_native_ca\tn_model_ca\tstatus\n")
n_ok = 0
for r in rows:
    unit = r["unit"]
    mp = f"results/{which}/{unit}.pdb"
    np_ = f"structures/native/{unit}.pdb"
    if not os.path.exists(mp):
        out.write(f"{unit}\t\t\t\t\t\t\tmissing\n")
        continue
    try:
        s = score(mp, np_)
        out.write(f"{unit}\t{s['tm_score']:.4f}\t{s['rmsd']:.3f}\t{s['gdt_ts']:.2f}"
                  f"\t{s['n_aligned']}\t{s['n_native_ca']}\t{s['n_model_ca']}\tok\n")
        n_ok += 1
    except Exception as e:
        out.write(f"{unit}\t\t\t\t\t\t\terror:{str(e)[:40]}\n")
out.close()
print(f"{which}: scored {n_ok}/{len(rows)}")
