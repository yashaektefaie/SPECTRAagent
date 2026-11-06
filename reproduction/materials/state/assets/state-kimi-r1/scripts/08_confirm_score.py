#!/usr/bin/env python
"""Score every confirmation-context unit for STATE and fixed references B1/B2.

Identical metric definitions as 04_score.py (frozen): pearson_delta vs the
context's real non-targeting control mean (from that run's train aggregates);
B1 = mean training delta of the context's own train perts; B2 = equal-weighted
mean of the gene's training deltas in the other three contexts (fallback B1).
Writes: results/per_unit_scores_<ctx>.csv
"""
import sys
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr

CTX = sys.argv[1]
CTRL = "non-targeting"
ALL_LINES = ["rpe1", "jurkat", "hepg2", "k562"]
OTHERS = [l for l in ALL_LINES if l != CTX]


def safe_pearson(a, b):
    if a is None or b is None:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return pearsonr(a, b)[0]


def main():
    man = pd.read_csv(f"results/axis_manifest_{CTX}.csv")
    gene_names = yaml.safe_load(open("configs/sthvg_fewshot_k562_hparams.yaml"))["gene_names"]
    gidx = {g: i for i, g in enumerate(gene_names)}

    ag = np.load(f"results/agg_train_{CTX}.npz", allow_pickle=True)
    keys = list(zip(ag["celltype"].astype(str), ag["pert"].astype(str)))
    rmean = dict(zip(keys, ag["real_means"]))
    pmean = dict(zip(keys, ag["pred_means"]))
    cnt = dict(zip(keys, ag["counts"]))

    ctrl_real = rmean[(CTX, CTRL)]
    print(f"{CTX} control cells (train):", cnt[(CTX, CTRL)])

    delta = {}
    for l in ALL_LINES:
        cl = rmean[(l, CTRL)]
        for (ct, g), v in rmean.items():
            if ct == l and g != CTRL:
                delta[(l, g)] = v - cl

    ctx_train_perts = [g for g in man.loc[man.seen_pair == 1, "gene"]]
    B1 = np.mean([delta[(CTX, g)] for g in ctx_train_perts if (CTX, g) in delta], axis=0)

    def B2(g):
        ds = [delta[(l, g)] for l in OTHERS if (l, g) in delta]
        return np.mean(ds, axis=0) if ds else B1

    z = np.load(f"results/preds_test_{CTX}.npz")
    test_reals, test_preds = z["reals"], z["preds"]
    pert = z["pert"].astype(str)
    idx_by_pert = pd.DataFrame({"p": pert}).groupby("p").indices
    test_rmean = {g: test_reals[ii].mean(axis=0) for g, ii in idx_by_pert.items()}
    test_pmean = {g: test_preds[ii].mean(axis=0) for g, ii in idx_by_pert.items()}
    test_ncell = {g: len(ii) for g, ii in idx_by_pert.items()}

    rows = []
    for _, r in man.iterrows():
        g = r["gene"]
        if r["seen_pair"] == 1:
            key = (CTX, g)
            if key not in rmean:
                continue
            rm, pmv, n = rmean[key], pmean[key], int(cnt[key])
        else:
            if g not in test_rmean:
                continue
            rm, pmv, n = test_rmean[g], test_pmean[g], int(test_ncell[g])
        d_real = rm - ctrl_real
        d_pred = pmv - ctrl_real
        b2 = B2(g)
        rows.append(dict(
            gene=g, level=r["level"], seen_pair=r["seen_pair"], exposure_E=r["exposure_E"],
            n_cells=n,
            pearson_delta_state=safe_pearson(d_pred, d_real),
            mse_delta_state=float(np.mean((d_pred - d_real) ** 2)),
            pearson_delta_B1=safe_pearson(B1, d_real),
            mse_delta_B1=float(np.mean((B1 - d_real) ** 2)),
            pearson_delta_B2=safe_pearson(b2, d_real),
            mse_delta_B2=float(np.mean((b2 - d_real) ** 2)),
            mse_delta_B0=float(np.mean(d_real ** 2)),
            effect_mag=float(np.linalg.norm(d_real)),
            basal_expr_target=float(ctrl_real[gidx[g]]) if g in gidx else np.nan,
        ))
    out = pd.DataFrame(rows)
    out.to_csv(f"results/per_unit_scores_{CTX}.csv", index=False)
    print(out.groupby("level").size())
    print("missing scores:", out["pearson_delta_state"].isna().sum())
    print(f"wrote results/per_unit_scores_{CTX}.csv", len(out))


if __name__ == "__main__":
    main()
