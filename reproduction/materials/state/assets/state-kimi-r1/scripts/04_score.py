#!/usr/bin/env python
"""Score every manifest unit for STATE and the fixed model-independent references.

Primary metric (pre-declared): pearson_delta with deltas vs the real k562
non-targeting control mean (train aggregates, largest sample).
Outputs results/per_unit_scores.csv
"""
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr

CTRL = "non-targeting"
LINES = ["rpe1", "jurkat", "hepg2"]

def safe_pearson(a, b):
    if a is None or b is None:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return pearsonr(a, b)[0]

def main():
    man = pd.read_csv("results/axis_manifest.csv")
    gene_names = yaml.safe_load(open("configs/sthvg_fewshot_k562_hparams.yaml"))["gene_names"]
    gidx = {g: i for i, g in enumerate(gene_names)}

    # ---- train aggregates: per (celltype, pert) mean pred/real (model gene space)
    ag = np.load("results/agg_train_k562.npz", allow_pickle=True)
    keys = list(zip(ag["celltype"].astype(str), ag["pert"].astype(str)))
    rmean = {k: v for k, v in zip(keys, ag["real_means"])}
    pmean = {k: v for k, v in zip(keys, ag["pred_means"])}
    cnt = {k: c for k, c in zip(keys, ag["counts"])}

    ctrl_real = rmean[("k562", CTRL)]
    print("k562 control cells (train):", cnt[("k562", CTRL)])

    # per-line training deltas (real)
    delta = {}
    for l in LINES + ["k562"]:
        cl = rmean[(l, CTRL)]
        for (ct, g), v in rmean.items():
            if ct == l and g != CTRL:
                delta[(l, g)] = v - cl

    k562_train_perts = [g for g in man.loc[man.seen_pair == 1, "gene"]]
    B1 = np.mean([delta[("k562", g)] for g in k562_train_perts if ("k562", g) in delta], axis=0)

    def B2(g):
        ds = [delta[(l, g)] for l in LINES if (l, g) in delta]
        return np.mean(ds, axis=0) if ds else B1

    # ---- unseen units from per-cell test predictions
    z = np.load("results/preds_test_k562.npz")
    test_reals = z["reals"]
    test_preds = z["preds"]
    pert = z["pert"].astype(str)
    idx_by_pert = pd.DataFrame({"p": pert}).groupby("p").indices
    test_rmean = {g: test_reals[ii].mean(axis=0) for g, ii in idx_by_pert.items()}
    test_pmean = {g: test_preds[ii].mean(axis=0) for g, ii in idx_by_pert.items()}
    test_ncell = {g: len(ii) for g, ii in idx_by_pert.items()}

    rows = []
    for _, r in man.iterrows():
        g = r["gene"]
        if r["seen_pair"] == 1:
            key = ("k562", g)
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
    out.to_csv("results/per_unit_scores.csv", index=False)
    print(out.groupby("level").size())
    print("missing scores:", out["pearson_delta_state"].isna().sum())
    print("wrote results/per_unit_scores.csv", len(out))

if __name__ == "__main__":
    main()
