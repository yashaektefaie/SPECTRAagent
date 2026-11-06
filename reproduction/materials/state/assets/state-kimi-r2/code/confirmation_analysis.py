"""Per-context metrics + frozen confirmation analysis for all 4 zeroshot contexts."""
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

CTX = ["k562", "jurkat", "hepg2", "rpe1"]

print("loading X_hvg ...")
a = ad.read_h5ad("data/replogle_concat.h5ad", backed="r")
obs = pd.read_parquet("data/obs_meta.parquet")
obs["gene"] = obs["gene"].astype(str)
obs["cell_line"] = obs["cell_line"].astype(str)
Xh = np.asarray(a.obsm["X_hvg"][:], dtype=np.float32)
genes_all = obs["gene"].values
lines_all = obs["cell_line"].values
is_ctrl = genes_all == "non-targeting"

rng = np.random.default_rng(0)
def boot_ci(a_, b_, n=10000):
    a_, b_ = np.asarray(a_), np.asarray(b_)
    d = np.empty(n)
    for i in range(n):
        d[i] = a_[rng.integers(0, len(a_), len(a_))].mean() - b_[rng.integers(0, len(b_), len(b_))].mean()
    return np.percentile(d, [2.5, 50, 97.5])

all_rows = []
summaries = []
for ctx in CTX:
    train_lines = [l for l in CTX if l != ctx]
    supp = {}
    tr_mask = lines_all != ctx
    tr_genes = genes_all[tr_mask]
    import collections
    cnt = collections.Counter(tr_genes)
    te_idx = np.where(lines_all == ctx)[0]
    te_genes = genes_all[te_idx]
    perts = sorted(set(te_genes) - {"non-targeting"})
    ncells = collections.Counter(te_genes)
    perts = [p for p in perts if ncells[p] >= 30]

    # true deltas
    te_ctrl = te_idx[te_genes == "non-targeting"]
    true_ctrl_mean = Xh[te_ctrl].mean(axis=0)
    true_delta = {p: Xh[te_idx[te_genes == p]].mean(axis=0) - true_ctrl_mean for p in perts}

    # STATE predicted deltas
    pa = ad.read_h5ad(f"results/{ctx}_pred_best.h5ad", backed="r")
    pg = pa.obs["gene"].astype(str).values
    Ph = np.asarray(pa.obsm["X_hvg"][:], dtype=np.float32)
    pred_ctrl_mean = Ph[pg == "non-targeting"].mean(axis=0)
    state_delta = {p: Ph[pg == p].mean(axis=0) - pred_ctrl_mean for p in perts}
    del Ph

    # references from this run's training lines
    line_delta = {}
    line_has = {}
    for line in train_lines:
        l_idx = np.where(lines_all == line)[0]
        l_genes = genes_all[l_idx]
        l_ctrl_mean = Xh[l_idx[l_genes == "non-targeting"]].mean(axis=0)
        for p in set(l_genes) - {"non-targeting"}:
            line_delta[(line, p)] = Xh[l_idx[l_genes == p]].mean(axis=0) - l_ctrl_mean
            line_has.setdefault(p, []).append(line)
    R2 = np.stack(list(line_delta.values())).mean(axis=0)

    def pear(x, y):
        sx, sy = x.std(), y.std()
        return float(np.corrcoef(x, y)[0, 1]) if sx > 0 and sy > 0 else np.nan

    for p in perts:
        S = int(cnt.get(p, 0))
        td = true_delta[p]
        sd = state_delta[p]
        r1d = np.stack([line_delta[(l, p)] for l in line_has[p]]).mean(axis=0) if p in line_has else R2
        all_rows.append(dict(
            context=ctx, pert=p, support=S, n_lines=len(line_has.get(p, [])),
            n_test_cells=ncells[p], true_l2=float(np.linalg.norm(td)),
            state_pearson_delta=pear(sd, td), state_mse_delta=float(np.mean((sd - td) ** 2)),
            r1_pearson_delta=pear(r1d, td), r1_mse_delta=float(np.mean((r1d - td) ** 2)),
            r2_pearson_delta=pear(R2, td), r2_mse_delta=float(np.mean((R2 - td) ** 2)),
        ))

    dfc = pd.DataFrame([r for r in all_rows if r["context"] == ctx])
    n0 = dfc[dfc.n_lines == 0]
    nP = dfc[dfc.n_lines > 0]
    ci = boot_ci(nP.state_pearson_delta.values, n0.state_pearson_delta.values)
    ci_r1 = boot_ci(nP.r1_pearson_delta.values, n0.r1_pearson_delta.values)
    by_n = dfc.groupby("n_lines")[["state_pearson_delta", "r1_pearson_delta", "r2_pearson_delta",
                                    "state_mse_delta", "r1_mse_delta"]].mean()
    summaries.append(dict(
        context=ctx, n0=len(n0), n_seen=len(nP),
        state_n0=n0.state_pearson_delta.mean(), state_seen=nP.state_pearson_delta.mean(),
        boot_lo=ci[0], boot_mid=ci[1], boot_hi=ci[2],
        r1_n0=n0.r1_pearson_delta.mean(), r1_seen=nP.r1_pearson_delta.mean(),
        r2_n0=n0.r2_pearson_delta.mean(),
        state_below_r2_at_n0=float(n0.state_pearson_delta.mean() < n0.r2_pearson_delta.mean()),
        state_drop=nP.state_pearson_delta.mean() - n0.state_pearson_delta.mean(),
        r1_drop=nP.r1_pearson_delta.mean() - n0.r1_pearson_delta.mean(),
        mse_n0=n0.state_mse_delta.mean(), mse_seen=nP.state_mse_delta.mean(),
        r1_mse_n0=n0.r1_mse_delta.mean(), r1_mse_seen=nP.r1_mse_delta.mean(),
    ))
    print(f"\n=== {ctx} ===")
    print(by_n.round(3))
    print(f"N=0 vs N>0 STATE boot CI of (seen - n0): [{ci[0]:.3f}, {ci[2]:.3f}]; "
          f"STATE(N=0)={n0.state_pearson_delta.mean():.3f} vs R2(N=0)={n0.r2_pearson_delta.mean():.3f}")

df = pd.DataFrame(all_rows)
df.to_csv("results/per_unit_scores_all_contexts.csv", index=False)
sdf = pd.DataFrame(summaries)
sdf.to_csv("results/confirmation_summary.csv", index=False)
print("\n=== confirmation summary ===")
print(sdf.round(3).to_string())
