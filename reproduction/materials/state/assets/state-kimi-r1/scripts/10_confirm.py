#!/usr/bin/env python
"""Pooled confirmation analysis for frozen axes A1b (primary) and A1c (secondary).

Confirmation set: ST-HVG-Replogle fewshot/{rpe1,jurkat,hepg2} runs (new held-out
contexts, new checkpoints) — frozen in FROZEN_AXIS.md addendum before any of
these runs were scored.

A1b success: pooled stratified (per-context z, Stouffer) test of
  Delta(seen-unseen) > 0 at one-sided p < 0.04 (group-sequential look 2),
  pooled Delta >= 0.05, directionally consistent in >= 2/3 contexts, robust to
  adjustment (log effect_mag + log n_cells).
A1c success: pooled (per-context standardized) spearman(emb_norm, pd_state) > 0
  within unseen units, one-sided p < 0.05, surviving effect-magnitude adjustment.
Gate call: Delta_state vs Delta_B1, Delta_B2 on identical units; label
  model-specific vs shared by reference comparison.

Writes: results/confirm_summary.json, results/confirm_spc_plot.png,
        results/spc_table_confirm.csv
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr
import statsmodels.formula.api as smf

CTXS = ["rpe1", "jurkat", "hepg2"]
rng = np.random.default_rng(20260722)


def boot_ci(x, n=10000, fn=np.mean):
    bs = np.empty(n)
    for b in range(n):
        bs[b] = fn(rng.choice(x, size=len(x), replace=True))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    per = {}
    for c in CTXS:
        df = pd.read_csv(f"results/per_unit_scores_{c}.csv")
        nm = pd.read_csv(f"results/proximity_embedding_norms_{c}.csv")[["gene", "emb_norm"]]
        df = df.merge(nm, on="gene", how="left")
        per[c] = df

    out = {"contexts": {}}
    zs, ws, deltas = [], [], []
    for c in CTXS:
        df = per[c]
        s = df[df.seen_pair == 1]
        u = df[df.seen_pair == 0]
        d_state = s.pearson_delta_state.mean() - u.pearson_delta_state.mean()
        d_B1 = s.pearson_delta_B1.mean() - u.pearson_delta_B1.mean()
        d_B2 = s.pearson_delta_B2.mean() - u.pearson_delta_B2.mean()
        # Welch t -> z for seen-vs-unseen on pd_state
        v = s.pearson_delta_state.var() / len(s) + u.pearson_delta_state.var() / len(u)
        z = d_state / np.sqrt(v)
        # bootstrap CI for delta
        bs = np.empty(5000)
        sv, uv = s.pearson_delta_state.values, u.pearson_delta_state.values
        for b in range(5000):
            bs[b] = rng.choice(sv, len(sv), True).mean() - rng.choice(uv, len(uv), True).mean()
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        # adjusted seen coefficient
        d = df.copy()
        d["log_effect"] = np.log(d.effect_mag)
        d["log_n"] = np.log(d.n_cells)
        m = smf.ols("pearson_delta_state ~ seen_pair + log_effect + log_n", data=d).fit()
        adj = dict(coef=float(m.params["seen_pair"]), p=float(m.pvalues["seen_pair"]))
        # A1c: within unseen, spearman(norm, pd) + adjusted
        un = df[df.seen_pair == 0].dropna(subset=["emb_norm"])
        rho, p_two = spearmanr(un.emb_norm, un.pearson_delta_state)
        p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
        un2 = un.copy()
        un2["log_effect"] = np.log(un2.effect_mag)
        un2["log_n"] = np.log(un2.n_cells)
        m2 = smf.ols("pearson_delta_state ~ emb_norm + log_effect + log_n", data=un2).fit()
        a1c_adj = dict(coef=float(m2.params["emb_norm"]), p=float(m2.pvalues["emb_norm"]))
        # reference deltas on identical units
        out["contexts"][c] = dict(
            n_seen=int(len(s)), n_unseen=int(len(u)),
            mean_pd_seen=float(s.pearson_delta_state.mean()),
            mean_pd_unseen=float(u.pearson_delta_state.mean()),
            delta_state=float(d_state), delta_boot_ci95=ci, z=float(z),
            delta_B1=float(d_B1), delta_B2=float(d_B2),
            adjusted_seen_coef=adj,
            a1c_spearman_rho=float(rho), a1c_p_one_sided=float(p_one),
            a1c_adjusted=a1c_adj,
            a1c_n_unseen=int(len(un)),
        )
        zs.append(z)
        ws.append(np.sqrt(len(s) + len(u)))
        deltas.append(d_state)

    # Stouffer pooled z (sqrt-n weights)
    Z = float(np.sum(np.array(zs) * ws) / np.sqrt(np.sum(np.array(ws) ** 2)))
    p_pool = float(1 - norm.cdf(Z))
    pooled_delta = float(np.mean(deltas))
    n_consistent = int(sum(d > 0 for d in deltas))

    # pooled A1c: fixed-effect meta of per-context spearman via Fisher z
    fz, fw = [], []
    for c in CTXS:
        r = out["contexts"][c]["a1c_spearman_rho"]
        n = out["contexts"][c]["a1c_n_unseen"]
        fz.append(np.arctanh(min(max(r, -0.999), 0.999)))
        fw.append(n - 3)
    r_pool = float(np.tanh(np.sum(np.array(fz) * fw) / np.sum(fw)))
    Z_r = float(np.sum(np.array(fz) * np.sqrt(fw)))  # Stouffer form
    p_r = float(1 - norm.cdf(Z_r))

    out["pooled"] = dict(
        A1b_stouffer_Z=Z, A1b_p_one_sided=p_pool,
        A1b_pooled_delta=pooled_delta, A1b_contexts_consistent=n_consistent,
        A1b_pass=bool(p_pool < 0.04 and pooled_delta >= 0.05 and n_consistent >= 2),
        A1c_pooled_spearman=r_pool, A1c_pooled_Z=Z_r, A1c_p_one_sided=p_r,
    )
    with open("results/confirm_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    # SPC table + plot: E-graded levels per context (ledger/exploratory)
    rows = []
    for c in CTXS:
        df = per[c]
        for lev in ["seen_pair", "unseen_E3", "unseen_E2", "unseen_E1"]:
            sub = df[df.level == lev]
            rows.append(dict(context=c, level=lev, n=len(sub),
                             pd_state=sub.pearson_delta_state.mean(),
                             pd_B1=sub.pearson_delta_B1.mean(),
                             pd_B2=sub.pearson_delta_B2.mean(),
                             mse_state=sub.mse_delta_state.mean(),
                             effect_mag=sub.effect_mag.mean()))
    spc = pd.DataFrame(rows)
    spc.to_csv("results/spc_table_confirm.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    order = ["seen_pair", "unseen_E3", "unseen_E2", "unseen_E1"]
    for ax, c in zip(axes, CTXS + ["k562"]):
        if c == "k562":
            df = pd.read_csv("results/per_unit_scores.csv")
            df["level"] = df.level.map(lambda l: "seen_pair" if l == "L4_seen_pair" else l.replace("L3_", "").replace("L2_", "").replace("L1_", ""))
        else:
            df = per[c]
        for col, lab, mk in [("pearson_delta_state", "STATE", "o-"),
                             ("pearson_delta_B1", "B1", "s--"),
                             ("pearson_delta_B2", "B2", "^--")]:
            ys = [df.loc[df.level == l, col].mean() for l in order]
            ax.plot(range(4), ys, mk, label=lab, ms=4)
        ax.set_xticks(range(4), ["seen", "E3", "E2", "E1"], fontsize=8)
        ax.set_title(c)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pearson_delta")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle("SPC: pair-novelty axis (A1b) across contexts — confirmation")
    fig.tight_layout()
    fig.savefig("results/confirm_spc_plot.png", dpi=150)


if __name__ == "__main__":
    main()
