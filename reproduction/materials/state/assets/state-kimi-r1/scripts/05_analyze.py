#!/usr/bin/env python
"""SPECTRA analysis for frozen axis A1 (cross-context perturbation exposure).

Computes the spectral performance curve (SPC), pre-declared statistics, gate
call, and confound-adjusted models. Writes:
  results/spc_table.csv, results/analysis_summary.json, results/spc_plot.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

LEVEL_ORDER = {"L4_seen_pair": 4, "L3_unseen_E3": 3, "L2_unseen_E2": 2, "L1_unseen_E1": 1}
rng = np.random.default_rng(20260722)

def boot_ci_diff(deep_vals, top_vals, n_boot=10000):
    """CI for mean(top) - mean(deep) via independent bootstrap."""
    ds = np.empty(n_boot)
    for b in range(n_boot):
        t = rng.choice(top_vals, size=len(top_vals), replace=True)
        d = rng.choice(deep_vals, size=len(deep_vals), replace=True)
        ds[b] = t.mean() - d.mean()
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(ds.mean())

def main():
    df = pd.read_csv("results/per_unit_scores.csv")
    df["level_num"] = df["level"].map(LEVEL_ORDER)
    df = df.sort_values("level_num")

    levels = sorted(df["level"].unique(), key=lambda l: -LEVEL_ORDER[l])
    summ = df.groupby("level").agg(
        n=("gene", "size"),
        pd_state=("pearson_delta_state", "mean"),
        pd_state_sd=("pearson_delta_state", "std"),
        pd_B1=("pearson_delta_B1", "mean"),
        pd_B2=("pearson_delta_B2", "mean"),
        mse_state=("mse_delta_state", "mean"),
        mse_B1=("mse_delta_B1", "mean"),
        mse_B2=("mse_delta_B2", "mean"),
        mse_B0=("mse_delta_B0", "mean"),
        effect_mag=("effect_mag", "mean"),
        n_cells=("n_cells", "mean"),
    ).reindex(levels)
    summ.to_csv("results/spc_table.csv")
    print(summ.round(4))

    # primary test: spearman over all units
    rho, p_two = spearmanr(df["level_num"], df["pearson_delta_state"])
    p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
    # spearman among unseen only
    un = df[df.seen_pair == 0]
    rho_u, p_two_u = spearmanr(un["level_num"], un["pearson_delta_state"])
    p_one_u = p_two_u / 2 if rho_u > 0 else 1 - p_two_u / 2

    top = df[df.level == "L4_seen_pair"]
    deep = df[df.level == "L1_unseen_E1"]
    res = {}
    for col in ["pearson_delta_state", "pearson_delta_B1", "pearson_delta_B2"]:
        lo, hi, est = boot_ci_diff(deep[col].values, top[col].values)
        res[f"delta_L4_minus_L1_{col}"] = dict(est=est, ci95=[lo, hi])

    # adjusted model
    df["log_effect"] = np.log(df["effect_mag"])
    df["log_n"] = np.log(df["n_cells"])
    m1 = smf.ols("pearson_delta_state ~ C(level)", data=df).fit()
    m2 = smf.ols("pearson_delta_state ~ C(level) + log_effect + log_n", data=df).fit()
    adj = {k: dict(coef=float(v), p=float(m2.pvalues[k]))
           for k, v in m2.params.items() if k.startswith("C(level)")}

    summary = dict(
        n_units=int(len(df)),
        level_counts={l: int((df.level == l).sum()) for l in levels},
        spearman_all=dict(rho=float(rho), p_one_sided=float(p_one)),
        spearman_unseen_only=dict(rho=float(rho_u), p_one_sided=float(p_one_u)),
        level_means=summ.round(5).to_dict(),
        degradation_deltas=res,
        ancova_adjusted=adj,
        ancova_R2_unadjusted=float(m1.rsquared), ancova_R2_adjusted=float(m2.rsquared),
    )
    with open("results/analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [LEVEL_ORDER[l] for l in levels]
    for col, lab, mk in [("pearson_delta_state", "STATE", "o-"),
                         ("pearson_delta_B1", "B1 context-mean", "s--"),
                         ("pearson_delta_B2", "B2 cross-context", "^--")]:
        ax[0].plot(xs, [summ.loc[l, col.replace("pearson_delta_", "pd_")] for l in levels], mk, label=lab)
    ax[0].set_xticks(xs, [l.replace("_", "\n") for l in levels], fontsize=8)
    ax[0].set_ylabel("pearson_delta"); ax[0].legend(); ax[0].set_title("SPC: axis A1 (exposure)")
    for col, lab, mk in [("mse_delta_state", "STATE", "o-"),
                         ("mse_delta_B1", "B1 context-mean", "s--"),
                         ("mse_delta_B2", "B2 cross-context", "^--")]:
        ax[1].plot(xs, [summ.loc[l, col.replace("mse_delta_", "mse_")] for l in levels], mk, label=lab)
    ax[1].set_xticks(xs, [l.replace("_", "\n") for l in levels], fontsize=8)
    ax[1].set_ylabel("mse_delta"); ax[1].legend(); ax[1].set_title("MSE delta by level")
    fig.tight_layout(); fig.savefig("results/spc_plot.png", dpi=150)
    print(json.dumps({k: v for k, v in summary.items() if 'means' not in k}, indent=1)[:1500])

if __name__ == "__main__":
    main()
