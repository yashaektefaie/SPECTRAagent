#!/usr/bin/env python3
"""Run ESMFold (esmfold_v1, ESM-2 3B) on the frozen confirmation set.

Single-sequence mode, default inference settings, fp32, chunk_size=128.
Writes results/pred/{unit}.pdb and results/pred_stats.tsv.
"""
import os, time, csv, sys
FASTA = sys.argv[1] if len(sys.argv) > 1 else "results/confirmation.fasta"
os.environ.setdefault("TORCH_HOME", os.path.abspath(".cache/torch"))
import torch
import esm

os.makedirs("results/pred", exist_ok=True)

seqs = {}
for l in open(FASTA):
    l = l.strip()
    if l.startswith(">"):
        k = l[1:]; seqs[k] = ""
    else:
        seqs[k] += l

todo = [(k, s) for k, s in seqs.items()
        if not os.path.exists(f"results/pred/{k}.pdb")]
todo.sort(key=lambda x: len(x[1]))
print(f"{len(todo)} units to predict", flush=True)

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()
model.set_chunk_size(128)

stats = open("results/pred_stats.tsv", "a")
if os.path.getsize("results/pred_stats.tsv") == 0:
    stats.write("unit\tlength\tmean_plddt\truntime_s\n")

for unit, seq in todo:
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.infer(seq)
        pdb = model.output_to_pdb(out)[0] if isinstance(model.output_to_pdb(out), list) else model.output_to_pdb(out)
        with open(f"results/pred/{unit}.pdb", "w") as f:
            f.write(pdb)
        plddt = float(out["plddt"].mean()) * 100 if out["plddt"].max() <= 1.0 else float(out["plddt"].mean())
        stats.write(f"{unit}\t{len(seq)}\t{plddt:.2f}\t{time.time()-t0:.1f}\n")
        stats.flush()
        print(f"OK {unit} len={len(seq)} plddt={plddt:.1f} {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"FAIL {unit}: {e}", flush=True)
    torch.cuda.empty_cache()
stats.close()
print("done", flush=True)
