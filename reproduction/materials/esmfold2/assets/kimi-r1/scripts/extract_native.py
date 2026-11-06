#!/usr/bin/env python3
"""Extract native reference chains (CA + backbone) from test mmCIFs.

Writes structures/native/{unit}.pdb (ATOM records, chain A) and
results/native_report.tsv (unit, n_res_seqres, n_res_resolved, frac_resolved).
"""
import csv, os, warnings
from Bio.PDB import MMCIFParser, PDBIO, Select
warnings.filterwarnings("ignore")

os.makedirs("structures/native", exist_ok=True)
import sys
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv)>1 else "results/confirmation_set.tsv"), delimiter="\t"))
seqs = {}
for l in open(sys.argv[2] if len(sys.argv)>2 else "results/testclu_rep_seq.fasta"):
    l = l.strip()
    if l.startswith(">"):
        k = l[1:]; seqs[k] = ""
    else:
        seqs[k] += l

parser = MMCIFParser(QUIET=True)
rep = open(sys.argv[3] if len(sys.argv)>3 else "results/native_report.tsv", "w")
rep.write("unit\tn_seqres\tn_resolved_ca\tfrac_resolved\n")
fails = []
for r in rows:
    unit = r["unit"]
    out_p = f"structures/native/{unit}.pdb"
    if os.path.exists(out_p) and os.path.getsize(out_p) > 50:
        continue
    entry, chain = unit.split("_")
    cif = f"structures/test/{entry}.cif"
    try:
        st = parser.get_structure(entry, cif)
        ch = None
        for model in st:
            for c in model:
                if c.id == chain:
                    ch = c
                    break
            break
        if ch is None:
            fails.append((unit, "chain not found")); continue
        ch.id = "A"  # normalize (auth ids like 'AAA' break PDB format)
        ca = [a for a in ch.get_atoms() if a.id == "CA"]
        n_res = len(seqs.get(unit, ""))
        class ChainSelect(Select):
            def accept_chain(self, c): return c.id == "A"
        io = PDBIO()
        io.set_structure(st)
        out = f"structures/native/{unit}.pdb"
        io.save(out, ChainSelect())
        rep.write(f"{unit}\t{n_res}\t{len(ca)}\t{len(ca)/max(n_res,1):.3f}\n")
    except Exception as e:
        fails.append((unit, str(e)[:80]))
rep.close()
print(f"extracted {len(rows)-len(fails)}/{len(rows)}; failures: {len(fails)}")
for f_ in fails:
    print("FAIL", f_)
