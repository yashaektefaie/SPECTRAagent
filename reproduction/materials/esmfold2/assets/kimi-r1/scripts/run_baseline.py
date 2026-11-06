#!/usr/bin/env python3
"""Model-independent template-copy baseline.

For each confirmation-set unit: take the frozen top pre-cutoff mmseqs hit,
globally align test vs template sequence (BLOSUM62, -10/-0.5), copy template
backbone (N,CA,C,O) coordinates for aligned residue pairs, write a model PDB.
Abstains (NA) when pident < 20 or qcov < 0.3 (frozen rule).
"""
import csv, os, warnings
from Bio.PDB import MMCIFParser
from Bio.Align import PairwiseAligner, substitution_matrices
warnings.filterwarnings("ignore")

os.makedirs("results/baseline", exist_ok=True)

import sys
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv)>1 else "results/confirmation_set.tsv"), delimiter="\t"))
seqs = {}
for l in open(sys.argv[2] if len(sys.argv)>2 else "results/confirmation.fasta"):
    l = l.strip()
    if l.startswith(">"):
        k = l[1:]; seqs[k] = ""
    else:
        seqs[k] += l

# qcov per unit (abstention uses coverage of the TEST chain = qcov)
qcov = {}
with open("results/test_vs_train.m8") as f:
    for r in csv.reader(f, delimiter="\t"):
        q, t, p, qc, tc = r[0], r[1], float(r[2]), float(r[3]), float(r[4])
        if max(qc, tc) < 0.5:
            continue
        if q not in qcov or (p, qc) > qcov[q][0]:
            qcov[q] = ((p, qc), qc)

parser = MMCIFParser(QUIET=True)
aligner = PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5

THREE2ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}
ONE2THREE = {v: k for k, v in THREE2ONE.items() if k != 'MSE'}

out = open(sys.argv[3] if len(sys.argv)>3 else "results/baseline_report.tsv", "w")
out.write("unit\ttemplate\tpident\tqcov\tstatus\tn_grafted\n")

for r in rows:
    unit = r["unit"]
    tseq_full = seqs[unit]
    if r["template"] == "NONE":
        out.write(f"{unit}\tNONE\t0\t0\tabstain_no_hit\t0\n"); continue
    pident = float(r["identity"])
    qc = qcov.get(unit, (None, 0.0))[1]
    if pident < 20 or qc < 0.3:
        out.write(f"{unit}\t{r['template']}\t{pident}\t{qc:.3f}\tabstain_low_conf\t0\n")
        continue
    t_entry, t_chain = r["template"].split("_")
    cif = f"structures/template/{t_entry}.cif"
    if not os.path.exists(cif):
        out.write(f"{unit}\t{r['template']}\t{pident}\t{qc:.3f}\tfail_no_cif\t0\n")
        continue
    try:
        st = parser.get_structure(t_entry, cif)
        ch = None
        for model in st:
            for c in model:
                if c.id == t_chain:
                    ch = c; break
            break
        if ch is None:
            out.write(f"{unit}\t{r['template']}\t{pident}\t{qc:.3f}\tfail_chain\t0\n")
            continue
        # template sequence + backbone coords per residue
        tres, tcoords, tseq = [], {}, []
        for res in ch:
            if res.id[0] != " ":
                continue
            rn = res.resname.strip()
            if rn not in THREE2ONE or "CA" not in res:
                continue
            tres.append(res)
            tseq.append(THREE2ONE[rn])
            tcoords[len(tseq) - 1] = {a.id: a.coord.copy() for a in res
                                      if a.id in ("N", "CA", "C", "O")}
        tseq = "".join(tseq)
        aln = aligner.align(tseq_full, tseq)[0]
        # aligned blocks: indices into tseq_full (target) and tseq (query)
        (t_blocks, q_blocks) = aln.aligned
        lines = []
        serial = 1
        n_graft = 0
        for (tstart, tend), (qstart, qend) in zip(*[t_blocks, q_blocks]):
            for off in range(tend - tstart):
                qi = qstart + off  # template seq index
                ti = tstart + off  # test seq index
                if qi not in tcoords:
                    continue
                aa = tseq_full[ti]
                resn = ONE2THREE.get(aa, "GLY")
                for atom in ("N", "CA", "C", "O"):
                    if atom not in tcoords[qi]:
                        continue
                    x, y, z = tcoords[qi][atom]
                    el = atom[0]
                    lines.append(
                        f"ATOM  {serial:5d} {atom:^4s} {resn:>3s} A{ti+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00          {el:>2s}\n")
                    serial += 1
                n_graft += 1
        if n_graft < 10:
            out.write(f"{unit}\t{r['template']}\t{pident}\t{qc:.3f}\tfail_few_grafted\t0\n")
            continue
        with open(f"results/baseline/{unit}.pdb", "w") as f:
            f.writelines(lines)
            f.write("TER\nEND\n")
        out.write(f"{unit}\t{r['template']}\t{pident}\t{qc:.3f}\tok\t{n_graft}\n")
    except Exception as e:
        out.write(f"{unit}\t{r['template']}\t{pident}\t{qc:.3f}\tfail:{str(e)[:40]}\t0\n")
out.close()
print("baseline done")
