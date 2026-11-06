#!/usr/bin/env python
"""Download mmCIFs for panel entries; extract per-chain canonical sequence
(from _pdbx_poly_seq_scheme), resolved CA coordinates, resolved fraction.
Writes results/confirm/native/{rep}.npz with keys: seq, res_idx (0-based canonical
indices of resolved residues), ca (N,3)."""
import os, time, json
import numpy as np
import requests
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
       "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
       "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
       "MSE":"M","SEC":"U","PYL":"O"}

os.makedirs("data/cifs", exist_ok=True)
os.makedirs("results/confirm/native", exist_ok=True)

reps = [l.split("\t")[0] for l in open("results/confirm/panel.tsv").read().splitlines()[1:]]
entries = sorted(set(r.split("_")[0] for r in reps))

def dl(e):
    p = "data/cifs/%s.cif" % e
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p
    for att in range(4):
        try:
            r = requests.get("https://files.rcsb.org/download/%s.cif" % e, timeout=120)
            if r.status_code == 200 and len(r.content) > 1000:
                open(p, "wb").write(r.content)
                return p
        except Exception as ex:
            print("dl retry", e, att, ex)
        time.sleep(3 * (att + 1))
    return None

def col(d, k):
    return d.get(k, [])

n_ok = n_fail = 0
for e in entries:
    p = dl(e)
    if p is None:
        print("DL FAIL", e); n_fail += 1; continue
    try:
        d = MMCIF2Dict(p)
        # poly seq scheme: map (auth chain, auth seq) -> (label_asym, label_seq, mon)
        sch_asym = col(d, "_pdbx_poly_seq_scheme.asym_id")          # label asym
        sch_ent  = col(d, "_pdbx_poly_seq_scheme.entity_id")
        sch_seq  = col(d, "_pdbx_poly_seq_scheme.seq_id")           # label seq id
        sch_mon  = col(d, "_pdbx_poly_seq_scheme.mon_id")
        sch_pdbn = col(d, "_pdbx_poly_seq_scheme.pdb_seq_num")      # auth num
        sch_pdbc = col(d, "_pdbx_poly_seq_scheme.pdb_strand_id")    # auth chain
        # canonical entity sequence per label asym
        canon = {}   # label_asym -> list of (label_seq_id int, mon)
        for a, s, m in zip(sch_asym, sch_seq, sch_mon):
            canon.setdefault(a, []).append((int(s), m))
        asym_seq = {a: "".join(AA3.get(m, "X") for _, m in sorted(v))
                    for a, v in canon.items()}
        # auth chain -> label asym (chains with poly scheme)
        auth2label = {}
        for a, pc in zip(sch_asym, sch_pdbc):
            auth2label.setdefault(pc, a)
        # resolved CA from atom_site (label numbering)
        atom_asym = col(d, "_atom_site.label_asym_id")
        atom_seq  = col(d, "_atom_site.label_seq_id")
        atom_name = col(d, "_atom_site.label_atom_id")
        atom_alt  = col(d, "_atom_site.label_alt_id")
        atom_x = col(d, "_atom_site.Cartn_x")
        atom_y = col(d, "_atom_site.Cartn_y")
        atom_z = col(d, "_atom_site.Cartn_z")
        ca = {}
        for a, s, n, al, x, y, z in zip(atom_asym, atom_seq, atom_name,
                                        atom_alt, atom_x, atom_y, atom_z):
            if n != "CA" or s == "." or a not in asym_seq:
                continue
            if al not in (".", "A", "1"):
                continue
            key = (a, int(s))
            if key not in ca:
                ca[key] = (float(x), float(y), float(z))
        for rep in reps:
            if not rep.startswith(e + "_"):
                continue
            auth = rep.split("_", 1)[1]
            la = auth2label.get(auth)
            if la is None or la not in asym_seq:
                print("NO CHAIN", rep); n_fail += 1; continue
            seq = asym_seq[la]
            idx = sorted(s for (a, s) in ca if a == la)
            coords = np.array([ca[(la, s)] for s in idx], dtype=np.float32)
            res_idx = np.array([s - 1 for s in idx], dtype=np.int32)  # 0-based
            np.savez_compressed("results/confirm/native/%s.npz" % rep, seq=seq,
                                res_idx=res_idx, ca=coords)
            n_ok += 1
    except Exception as ex:
        print("PARSE FAIL", e, repr(ex)[:200]); n_fail += 1

print("ok:", n_ok, "fail:", n_fail)
