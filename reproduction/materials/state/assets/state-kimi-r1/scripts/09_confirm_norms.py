#!/usr/bin/env python
"""Pert-embedding norms for a confirmation run (axis A1c, label-free).

Same construction as 06_proximity_check.py: L2 norm of the run's pert_encoder
row per gene, mapped via the run's own pert_onehot_map.pt.
Writes: results/proximity_embedding_norms_<ctx>.csv
"""
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu

CTX = sys.argv[1]


def main():
    ck = torch.load(f"weights/fewshot_{CTX}/checkpoints/final.ckpt",
                    map_location="cpu", weights_only=False)
    W = ck["state_dict"]["pert_encoder.0.weight"].numpy()
    norms = np.linalg.norm(W, axis=0)
    omap = torch.load(f"weights/fewshot_{CTX}/pert_onehot_map.pt", weights_only=False)
    name2idx = {str(k): int(v.argmax()) for k, v in omap.items()}
    man = pd.read_csv(f"results/axis_manifest_{CTX}.csv")
    man["emb_norm"] = [norms[name2idx[g]] if g in name2idx else np.nan for g in man["gene"]]
    print("genes missing from onehot map:", man["emb_norm"].isna().sum())
    man = man.dropna(subset=["emb_norm"])
    man.to_csv(f"results/proximity_embedding_norms_{CTX}.csv", index=False)

    seen = man[man.seen_pair == 1]["emb_norm"]
    un = man[man.seen_pair == 0]
    print(f"seen: n={len(seen)} norm mean={seen.mean():.4f} sd={seen.std():.4f}")
    print(f"unseen: n={len(un)} norm mean={un['emb_norm'].mean():.4f} sd={un['emb_norm'].std():.4f}")
    u = mannwhitneyu(seen, un["emb_norm"], alternative="greater")
    print("MWU seen > unseen: p=", u.pvalue)
    rho, p = spearmanr(man["exposure_E"], man["emb_norm"])
    print(f"spearman(E, norm) all: rho={rho:.4f} p={p:.3e}")


if __name__ == "__main__":
    main()
