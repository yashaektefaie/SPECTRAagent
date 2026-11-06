#!/usr/bin/env python
"""Smoke test: fold ubiquitin (1UBQ, 76 aa) with ESMFold2 + local fp8 ESMC-6B.
Checks: load works, pLDDT sane, TM-score vs experimental 1UBQ >= 0.9 expected."""
import os, sys, time, subprocess, re, requests
import torch
sys.path.insert(0, "audit/scripts")
from struct_utils import extract_ca, ca_to_pdb

seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config

HF_REPO = "audit/weights/hf/hub/models--biohub--ESMFold2/snapshots/1ebf0e3481a5184eb6171d40615c79e384b48796"
cfg = ESMFold2Config.from_pretrained(HF_REPO)
cfg.esmc_id = os.path.abspath("audit/weights/esmc6b-fp8")
t0 = time.time()
model = ESMFold2Model.from_pretrained(HF_REPO, config=cfg, esmc_precision="bf16")
model = model.cuda().eval()
print(f"load ok in {time.time()-t0:.0f}s", flush=True)

torch.manual_seed(0); torch.cuda.manual_seed_all(0)
t0 = time.time()
out = model.infer_protein(seq, num_loops=3, num_sampling_steps=50, num_diffusion_samples=1)
dt = time.time() - t0
pdb = model.output_to_pdb(out)
open("audit/results/ubq_smoke.pdb", "w").write(pdb)
print(f"folded 76aa in {dt:.1f}s, pLDDT mean {float(out['plddt'].mean()):.3f}, "
      f"pTM {float(out['ptm'].mean()):.3f}", flush=True)

if not os.path.exists("audit/data/cif/1UBQ.cif"):
    r = requests.get("https://files.rcsb.org/download/1UBQ.cif", timeout=60)
    open("audit/data/cif/1UBQ.cif", "wb").write(r.content)
s, c, _ = extract_ca("audit/data/cif/1UBQ.cif", auth_chain="A")
open("audit/results/exp/1UBQ.pdb", "w").write(ca_to_pdb(s, c))
o = subprocess.run(["work/tmalign/TMalign", "audit/results/ubq_smoke.pdb",
                    "audit/results/exp/1UBQ.pdb"], capture_output=True, text=True).stdout
tms = re.findall(r"TM-score=\s*([0-9.]+)", o)
print("TM vs 1UBQ (norm by exp):", tms[1] if len(tms) > 1 else tms, flush=True)
print("GPU mem GB:", torch.cuda.max_memory_allocated() / 1e9, flush=True)
