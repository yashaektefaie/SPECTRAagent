#!/usr/bin/env python
"""Frozen model-independent baseline: naive template coordinate transfer.
For each panel rep, template = train-pool hit with max pident & qcov>=0.5
(same search table as the frozen axis). Transfer template CA coords to
aligned test positions (MMseqs cigar); score TM vs experimental reference
with the SAME normalization (resolved reference length) as the target metric.
Writes results/confirm/baseline_scores.tsv
"""
import os, re, time, sys
import numpy as np
import requests
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
sc = importlib.import_module("08_score")

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
       "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
       "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
       "MSE":"M","SEC":"U","PYL":"O"}

os.makedirs("data/cifs", exist_ok=True)

def dl_cif(e):
    p = "data/cifs/%s.cif" % e
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p
    for att in range(4):
        try:
            r = requests.get("https://files.rcsb.org/download/%s.cif" % e, timeout=120)
            if r.status_code == 200 and len(r.content) > 1000:
                open(p, "wb").write(r.content)
                return p
        except Exception:
            pass
        time.sleep(2 * (att + 1))
    return None

def col(d, k):
    return d.get(k, [])

def chain_ca(e, auth_chain):
    """Return dict label_seq_id(1-based)->(x,y,z) and chain length for a chain."""
    p = dl_cif(e)
    if p is None:
        return None, None
    d = MMCIF2Dict(p)
    sch_asym = col(d, "_pdbx_poly_seq_scheme.asym_id")
    sch_seq = col(d, "_pdbx_poly_seq_scheme.seq_id")
    sch_mon = col(d, "_pdbx_poly_seq_scheme.mon_id")
    sch_pdbc = col(d, "_pdbx_poly_seq_scheme.pdb_strand_id")
    canon = {}
    auth2label = {}
    for a, s, m, pc in zip(sch_asym, sch_seq, sch_mon, sch_pdbc):
        canon.setdefault(a, []).append((int(s), m))
        auth2label.setdefault(pc, a)
    la = auth2label.get(auth_chain)
    if la is None or la not in canon:
        return None, None
    L = len(canon[la])
    atom_asym = col(d, "_atom_site.label_asym_id")
    atom_seq = col(d, "_atom_site.label_seq_id")
    atom_name = col(d, "_atom_site.label_atom_id")
    atom_alt = col(d, "_atom_site.label_alt_id")
    ax = col(d, "_atom_site.Cartn_x"); ay = col(d, "_atom_site.Cartn_y"); az = col(d, "_atom_site.Cartn_z")
    ca = {}
    for a, s, n, al, x, y, z in zip(atom_asym, atom_seq, atom_name, atom_alt, ax, ay, az):
        if a != la or n != "CA" or s == "." or al not in (".", "A", "1"):
            continue
        k = int(s)
        if k not in ca:
            ca[k] = (float(x), float(y), float(z))
    return ca, L

def parse_cigar(cigar):
    return [(int(n), op) for n, op in re.findall(r"(\d+)([MID])", cigar)]

def alignment_pairs(qstart, tstart, cigar):
    """0-based (qpos, tpos) aligned pairs from MMseqs (1-based starts)."""
    pairs = []
    q, t = qstart - 1, tstart - 1
    for n, op in parse_cigar(cigar):
        if op == "M":
            for i in range(n):
                pairs.append((q + i, t + i))
            q += n; t += n
        elif op == "I":
            q += n
        elif op == "D":
            t += n
    return pairs

# best template per rep: max pident with qcov>=0.5 (frozen definition)
best = {}
with open("results/rep_vs_train.m8") as f:
    for line in f:
        p = line.split("\t")
        q, t, pident, qlen = p[0], p[1], float(p[2]), float(p[4])
        qs, qe, ts, te = int(p[5]), int(p[6]), int(p[7]), int(p[8])
        qcov = (qe - qs + 1) / qlen
        if qcov >= 0.5 and (q not in best or pident > best[q][0]):
            best[q] = (pident, t, qs, qe, ts, te, p[9], qcov)

panel = [l.split("\t")[0] for l in open("results/confirm/panel_final.tsv").read().splitlines()[1:]]

# template fasta lengths, to verify positional mapping against CIF scheme
tlen_map = {}
need = set(best[r][1] for r in panel if r in best)
name = None
for line in open("data/train_pool.fasta"):
    if line.startswith(">"):
        name = line[1:].strip()
        if name in need:
            tlen_map[name] = 0
    elif name in tlen_map and tlen_map.get(name) is not None:
        tlen_map[name] += len(line.strip())

out = open("results/confirm/baseline_scores.tsv", "w")
out.write("rep\ttemplate\tpident\tqcov\tn_pairs\tbaseline_tm\tbaseline_gdt_ts\n")
for rep in panel:
    z = np.load("results/confirm/native/%s.npz" % rep)
    res_idx = z["res_idx"]; nat = z["ca"]; L = len(res_idx)
    resolved = set(res_idx.tolist())
    if rep not in best:
        out.write("%s\tNONE\t0\t0\t0\tNA\tNA\n" % rep)
        continue
    pident, target, qs, qe, ts, te, cigar, qcov = best[rep]
    tent, tchain = target.rsplit("_", 1)
    tca, tlen = chain_ca(tent, tchain)
    if tca is None or tlen_map.get(target) != tlen:
        out.write("%s\t%s\t%.1f\t%.2f\t0\tNA\tNA\n" % (rep, target, pident, qcov))
        continue
    pairs = alignment_pairs(qs, ts, cigar)
    P, Q = [], []
    for qp, tp in pairs:
        if qp in resolved and (tp + 1) in tca:
            P.append(tca[tp + 1])
            Q.append(nat[res_idx.tolist().index(qp)])
    if len(P) < 15:
        out.write("%s\t%s\t%.1f\t%.2f\t%d\tNA\tNA\n" % (rep, target, pident, qcov, len(P)))
        continue
    P = np.array(P, dtype=np.float32); Q = np.array(Q, dtype=np.float32)
    tm, gts, gha, rmsd = sc.tm_gdt(P, Q, L)
    out.write("%s\t%s\t%.1f\t%.2f\t%d\t%.4f\t%.4f\n" % (rep, target, pident, qcov, len(P), tm, gts))
    out.flush()
out.close()
print("baseline done")
