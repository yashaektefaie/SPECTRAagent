#!/usr/bin/env python
"""Score predictions and the template-copy reference with TMalign.

For each plan entity:
  exp chain CA  <- audit/data/cif/{entry}.cif (auth chain = first in meta.chains)
  model score   <- TMalign(pred.pdb, exp_ca.pdb)   (TM-score normalized by exp)
  reference     <- template-copy: transfer template CA coords (best MMseqs2 hit
                   in pre-cutoff pool) onto test sequence over the pairwise
                   alignment from the mmseqs2 search output (cigar-free local
                   alignment recomputed here with parasail-free NW via Biopython),
                   then TMalign(template_copy.pdb, exp_ca.pdb)
Writes audit/results/scores.csv with per-unit rows.
"""
import csv, json, os, subprocess, sys, re
sys.path.insert(0, "audit/scripts")
from struct_utils import extract_ca, ca_to_pdb, selector_from_seqres_id
from Bio import Align

PLAN = sys.argv[1] if len(sys.argv) > 1 else "audit/data/sample_plan.csv"
TM = "work/tmalign/TMalign"
os.makedirs("audit/results/exp", exist_ok=True)
os.makedirs("audit/results/ref", exist_ok=True)

meta = {}
for line in open("audit/data/test_candidates.jsonl"):
    d = json.loads(line)
    meta[d["entity"]] = d
axis = {r["entity"]: r for r in csv.DictReader(open("audit/data/axis_table.csv"))}
seqs = {}
for line in open("audit/data/test_candidates.fasta"):
    line = line.strip()
    if line.startswith(">"):
        k = line[1:]; seqs[k] = ""
    else:
        seqs[k] += line

def tmalign(pdb1, pdb2):
    """Return (TM-score normalized by pdb2 length, RMSD). pdb2 = reference (exp)."""
    out = subprocess.run([TM, pdb1, pdb2], capture_output=True, text=True).stdout
    m = re.findall(r"TM-score=\s*([0-9.]+)", out)
    r = re.search(r"RMSD=\s*([0-9.]+)", out)
    # second TM-score in output is normalized by pdb2 length
    tm = float(m[1]) if len(m) > 1 else (float(m[0]) if m else None)
    return tm, (float(r.group(1)) if r else None)

aligner = Align.PairwiseAligner()
aligner.mode = "local"
aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5

poolseq = {}
hdr = None
for line in open("audit/data/pool_seqs.fasta"):
    line = line.strip()
    if line.startswith(">"):
        hdr = line[1:].split()[0]; poolseq[hdr] = ""
    elif hdr:
        poolseq[hdr] += line

rows = []
for row in csv.DictReader(open(PLAN)):
    e = row["entity"]
    m = meta[e]
    entry = m["entry"]
    chain = m["chains"][0]
    exp_pdb = f"audit/results/exp/{e}.pdb"
    if not os.path.exists(exp_pdb):
        seq_exp, coords, _ = extract_ca(f"audit/data/cif/{entry}.cif", auth_chain=chain)
        if not coords:
            print("NOEXP", e); continue
        open(exp_pdb, "w").write(ca_to_pdb(seq_exp, coords))
        obs = len(coords)
    else:
        obs = None
    rec = {"entity": e, "bin": row["bin"], "ident": axis[e]["ident"],
           "len": m["length"], "obs_ca": obs}
    # model score
    pred = f"audit/results/pred/{e}.pdb"
    if os.path.exists(pred):
        tm, rmsd = tmalign(pred, exp_pdb)
        rec.update(tm_model=tm, rmsd_model=rmsd)
    # reference: template copy
    hit = axis[e]["hit"]
    if hit:
        t_entry = hit.split("_")[0]
        auth, ent = selector_from_seqres_id(hit)
        t_cif = f"audit/data/cif/{t_entry}.cif"
        ref_pdb = f"audit/results/ref/{e}.pdb"
        if os.path.exists(t_cif):
            t_seq, t_coords, _ = extract_ca(t_cif, auth_chain=auth, entity_id=ent)
            if t_coords:
                q_seq = seqs[e]
                aln = aligner.align(q_seq, t_seq)[0]
                q_blocks, t_blocks = aln.aligned
                coords_out, seq_out = [], []
                for (qs, qe), (ts, te) in zip(q_blocks, t_blocks):
                    for k in range(qe - qs):
                        seq_out.append(q_seq[qs + k])
                        coords_out.append(t_coords[ts + k])
                open(ref_pdb, "w").write(ca_to_pdb("".join(seq_out), coords_out))
                tm_r, rmsd_r = tmalign(ref_pdb, exp_pdb)
                rec.update(tm_ref=tm_r, rmsd_ref=rmsd_r, template=hit)
    rows.append(rec)
    print(rec, flush=True)

keys = sorted({k for r in rows for k in r})
with open("audit/results/scores.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader(); w.writerows(rows)
print("wrote audit/results/scores.csv", len(rows))
