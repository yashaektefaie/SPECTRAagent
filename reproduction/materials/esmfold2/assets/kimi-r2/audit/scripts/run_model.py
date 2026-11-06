#!/usr/bin/env python
"""Run ESMFold2 (biohub/ESMFold2, single-sequence mode) on plan entities.

Fixed inference config for ALL units (frozen):
  num_loops=3, num_sampling_steps=50, num_diffusion_samples=1, seed=0
Backbone: local ESMC-6B fp8-storage (dequantized to bf16 at load).
Writes audit/results/pred/{entity}.pdb and appends audit/results/model_log.jsonl.
"""
import csv, json, os, sys, time
import torch

PLAN = sys.argv[1] if len(sys.argv) > 1 else "audit/data/sample_plan.csv"
os.makedirs("audit/results/pred", exist_ok=True)

seqs = {}
for line in open("audit/data/test_candidates.fasta"):
    line = line.strip()
    if line.startswith(">"):
        k = line[1:]
        seqs[k] = ""
    else:
        seqs[k] += line

todo = []
for row in csv.DictReader(open(PLAN)):
    e = row["entity"]
    if not os.path.exists(f"audit/results/pred/{e}.pdb"):
        todo.append((row["bin"], e))
print(f"todo: {len(todo)}", flush=True)
if not todo:
    sys.exit(0)

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config

HF_REPO = "audit/weights/hf/hub/models--biohub--ESMFold2/snapshots/1ebf0e3481a5184eb6171d40615c79e384b48796"
cfg = ESMFold2Config.from_pretrained(HF_REPO)
cfg.esmc_id = os.path.abspath("audit/weights/esmc6b-fp8")
model = ESMFold2Model.from_pretrained(HF_REPO, config=cfg, esmc_precision="bf16")
model = model.cuda().eval()

log_f = open("audit/results/model_log.jsonl", "a")
for b, e in todo:
    seq = seqs[e]
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    t0 = time.time()
    try:
        pdb = model.infer_protein_as_pdb(
            seq, num_loops=3, num_sampling_steps=50, num_diffusion_samples=1)
        dt = time.time() - t0
        open(f"audit/results/pred/{e}.pdb", "w").write(pdb)
        rec = {"entity": e, "bin": b, "len": len(seq), "sec": round(dt, 2),
               "ok": True}
    except Exception as ex:
        rec = {"entity": e, "bin": b, "len": len(seq), "ok": False,
               "err": str(ex)[:300]}
    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
    print(rec, flush=True)
log_f.close()
