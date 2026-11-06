#!/usr/bin/env python
"""Score ESMFold predictions vs experimental natives: TM-score (iterative,
normalized by resolved reference length per frozen spec) and GDT-TS/GDT-HA
from the TM-optimal superposition. Cross-checked against tmtools on a
fully-resolved example (see sanity output).

Writes results/target_scores.tsv
"""
import numpy as np

def kabsch(P, Q, w=None):
    """R,t aligning P->Q (N,3), optional weights."""
    if w is None:
        w = np.ones(len(P))
    w = w / w.sum()
    cp = (P * w[:, None]).sum(0)
    cq = (Q * w[:, None]).sum(0)
    Pc = P - cp; Qc = Q - cq
    H = (Pc * w[:, None]).T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T   # column convention: R @ p ~ q
    return R, cq - R @ cp

def tm_gdt(P, Q, Lnorm):
    """P mobile, Q reference (N,3 aligned pairs). TM normalized by Lnorm."""
    d0 = 1.24 * (max(Lnorm, 17) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    w = np.ones(len(P))
    for _ in range(10):
        R, t = kabsch(P, Q, w)
        d = np.linalg.norm(P @ R.T + t - Q, axis=1)
        w_new = 1.0 / (1.0 + (d / d0) ** 2)
        if np.allclose(w_new, w, atol=1e-6):
            w = w_new
            break
        w = w_new
    tm = float(np.mean(1.0 / (1.0 + (d / d0) ** 2))) * len(P) / Lnorm
    gdt_ts = float(np.mean([np.mean(d <= c) for c in (1, 2, 4, 8)])) * len(P) / Lnorm
    gdt_ha = float(np.mean([np.mean(d <= c) for c in (0.5, 1, 2, 4)])) * len(P) / Lnorm
    rmsd = float(np.sqrt(np.mean(d ** 2)))
    return tm, gdt_ts, gdt_ha, rmsd

def pred_ca(pdbpath):
    coords = []
    for line in open(pdbpath):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords, dtype=np.float32)

if __name__ == "__main__":
    panel = [l.split("\t")[0] for l in open("results/panel_final.tsv").read().splitlines()[1:]]
    with open("results/target_scores.tsv", "w") as f:
        f.write("rep\tn_resolved\ttm\tgdt_ts\tgdt_ha\trmsd\n")
        for rep in panel:
            z = np.load("results/native/%s.npz" % rep)
            res_idx = z["res_idx"]; nat = z["ca"]
            P = pred_ca("results/pred/%s.pdb" % rep)
            L = len(res_idx)
            assert len(P) >= res_idx.max() + 1, rep
            Q = P[res_idx]
            tm, gts, gha, rmsd = tm_gdt(Q, nat, L)
            f.write("%s\t%d\t%.4f\t%.4f\t%.4f\t%.2f\n" % (rep, L, tm, gts, gha, rmsd))
    print("scored", len(panel))
