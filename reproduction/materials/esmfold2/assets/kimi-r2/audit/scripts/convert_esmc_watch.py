#!/usr/bin/env python
"""Convert ESMC-6B fp32 shards present in /dev/shm/esmc6b to fp8-storage shards.
Runs as a watcher: converts any complete shard (verified via safe_open) once."""
import os, time, torch, sys
from safetensors import safe_open
from safetensors.torch import save_file

OUT = "audit/weights/esmc6b-fp8"
SHM = "/dev/shm/esmc6b"
os.makedirs(OUT, exist_ok=True)
shards = [f"model-0000{i}-of-00006.safetensors" for i in range(1, 7)]
done = set(s for s in shards if os.path.exists(os.path.join(OUT, s)))
while len(done) < 6:
    for shard in shards:
        if shard in done:
            continue
        src = os.path.join(SHM, shard)
        if not os.path.exists(src):
            continue
        try:
            tensors = {}
            n_fp8 = n_bf16 = 0
            with safe_open(src, framework="pt") as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    if t.dtype == torch.float32 and t.dim() == 2 and t.numel() >= 1_000_000:
                        t = t.to(torch.float8_e4m3fn); n_fp8 += 1
                    elif t.dtype == torch.float32:
                        t = t.to(torch.bfloat16); n_bf16 += 1
                    tensors[k] = t
        except Exception as e:
            time.sleep(10)
            continue
        os.remove(src)
        save_file(tensors, os.path.join(OUT, shard))
        done.add(shard)
        print("converted", shard, n_fp8, n_bf16, flush=True)
    time.sleep(5)
print("ALL DONE", flush=True)
