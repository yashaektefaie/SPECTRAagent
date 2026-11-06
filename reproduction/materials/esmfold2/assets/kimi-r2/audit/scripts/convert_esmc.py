#!/usr/bin/env python
"""Stream-convert biohub/ESMC-6B fp32 shards to storage-fp8 (weights) / bf16 (small
tensors), downloading via /dev/shm to avoid double disk usage.

Provenance: https://huggingface.co/biohub/ESMC-6B revision
45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a (fetched 2026-07-22).
Justification: the ESMFold2 paper states ESMC weights are cast to FP8 during
training; we store 2D weight matrices in float8_e4m3fn and small tensors
(embeddings, norms, biases) in bfloat16, then dequantize to bf16 at load time.
"""
import json, os, subprocess, sys, hashlib
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "https://huggingface.co/biohub/ESMC-6B/resolve/main"
REVA = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
OUT = "audit/weights/esmc6b-fp8"
SHM = "/dev/shm/esmc6b"
os.makedirs(OUT, exist_ok=True)
os.makedirs(SHM, exist_ok=True)

index = json.load(open(f"{OUT}/model.safetensors.index.json")) \
    if os.path.exists(f"{OUT}/model.safetensors.index.json") else None
if index is None:
    subprocess.run(["curl", "-sL", f"{BASE}/model.safetensors.index.json",
                    "-o", f"{OUT}/model.safetensors.index.json"], check=True)
    index = json.load(open(f"{OUT}/model.safetensors.index.json"))
    for extra in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "README.md"]:
        subprocess.run(["curl", "-sL", f"{BASE}/{extra}", "-o", f"{OUT}/{extra}"],
                       check=False)

shards = sorted(set(index["weight_map"].values()))
log = open("audit/logs/esmc_convert.log", "a")
for shard in shards:
    dst = os.path.join(OUT, shard)
    if os.path.exists(dst):
        print("skip", shard, flush=True)
        continue
    src = os.path.join(SHM, shard)
    subprocess.run(["curl", "-sL", "--retry", "3", f"{BASE}/{shard}", "-o", src],
                   check=True)
    tensors = {}
    n_fp8 = n_bf16 = 0
    with safe_open(src, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if t.dtype == torch.float32 and t.dim() == 2 and t.numel() >= 1_000_000:
                t = t.to(torch.float8_e4m3fn)
                n_fp8 += 1
            elif t.dtype == torch.float32:
                t = t.to(torch.bfloat16)
                n_bf16 += 1
            tensors[k] = t
    os.remove(src)  # delete fp32 before writing fp8 to minimize peak disk
    save_file(tensors, dst)
    log.write(f"{shard}: fp8={n_fp8} bf16={n_bf16}\n"); log.flush()
    print("done", shard, n_fp8, n_bf16, flush=True)
log.close()
print("ALL DONE", flush=True)
