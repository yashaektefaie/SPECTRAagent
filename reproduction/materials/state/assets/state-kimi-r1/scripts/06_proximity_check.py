#!/usr/bin/env python
"""Label-free proximity verification for axis A1.

The ST-HVG model encodes perturbations as one-hot -> pert_encoder embedding
rows. A (gene, context) pair that was never trained receives no gradient for
that context; but the *gene* embedding row is shared across contexts, so its
training exposure is measurable directly from the checkpoint as the embedding
row norm. We verify:
  V1: seen k562 pairs and higher-E genes have systematically larger trained
      embedding norms (measured train proximity decreases along the axis).
No outcome data used.
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu

def main():
    ck = torch.load("weights/sthvg_fewshot_k562_final.ckpt", map_location="cpu", weights_only=False)
    W = ck["state_dict"]["pert_encoder.0.weight"].numpy()  # Linear(2024->328): (328, 2024)
    norms = np.linalg.norm(W, axis=0)  # per-perturbation embedding norm
    omap = torch.load("weights/fewshot_k562/pert_onehot_map.pt", weights_only=False)
    print(type(omap))
    if isinstance(omap, dict):
        name2idx = {str(k): int(v.argmax()) for k, v in omap.items()}
    else:
        print(omap if not hasattr(omap, 'keys') else list(omap.keys())[:5])
        raise SystemExit
    man = pd.read_csv("results/axis_manifest.csv")
    man["emb_norm"] = [norms[name2idx[g]] if g in name2idx else np.nan for g in man["gene"]]
    print("genes missing from onehot map:", man["emb_norm"].isna().sum())
    man = man.dropna(subset=["emb_norm"])
    man.to_csv("results/proximity_embedding_norms.csv", index=False)

    seen = man[man.seen_pair == 1]["emb_norm"]
    un = man[man.seen_pair == 0]
    print(f"seen pairs: n={len(seen)} norm mean={seen.mean():.4f} sd={seen.std():.4f}")
    for E in [3, 2, 1]:
        v = un[un.exposure_E == E]["emb_norm"]
        print(f"unseen E={E}: n={len(v)} norm mean={v.mean():.4f} sd={v.std():.4f}")
    u = mannwhitneyu(seen, un["emb_norm"], alternative="greater")
    print("MWU seen > unseen:", u.statistic, "p=", u.pvalue)
    rho, p = spearmanr(man["exposure_E"], man["emb_norm"])
    print(f"spearman(E, norm) all: rho={rho:.4f} p={p:.3e}")
    rho2, p2 = spearmanr(un["exposure_E"], un["emb_norm"])
    print(f"spearman(E, norm) unseen only: rho={rho2:.4f} p={p2:.3e}")
    # how many unseen genes have norm exactly at init scale?
    print("norm percentiles overall:", np.percentile(norms, [1, 5, 25, 50, 75, 95, 99]).round(3))

if __name__ == "__main__":
    main()
