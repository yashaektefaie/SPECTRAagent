#!/usr/bin/env python
"""Run ESMFold (esmfold_v1) on the frozen panel. Resume-safe.
Writes results/pred/{rep}.pdb and appends results/pred_plddt.csv."""
import os, sys, time, json
import torch
import esm

os.makedirs("results/pred", exist_ok=True)

seqs = []
name = None
for line in open("results/panel_input.fasta"):
    if line.startswith(">"):
        name = line[1:].strip()
        seqs.append([name, ""])
    else:
        seqs[-1][1] += line.strip()

done = set()
pl_path = "results/pred_plddt.csv"
if os.path.exists(pl_path):
    for line in open(pl_path).read().splitlines()[1:]:
        done.add(line.split(",")[0])

model = esm.pretrained.esmfold_v1().eval().cuda()
plf = open(pl_path, "a")
if not done:
    plf.write("rep,length,mean_plddt,seconds\n")

for name, seq in seqs:
    if name in done:
        continue
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.infer(seq)
            pdb = model.output_to_pdb(out)[0]
        open("results/pred/%s.pdb" % name, "w").write(pdb)
        pl = float(out["plddt"].mean().item())
        dt = time.time() - t0
        plf.write("%s,%d,%.2f,%.1f\n" % (name, len(seq), pl, dt))
        plf.flush()
        print("OK", name, len(seq), round(pl, 1), round(dt, 1), flush=True)
    except Exception as e:
        print("FAIL", name, repr(e)[:200], flush=True)
        torch.cuda.empty_cache()
plf.close()
print("ALL DONE")
