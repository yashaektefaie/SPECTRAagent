"""Compute per-perturbation deltas and metrics for STATE predictions and fixed references.

All in the 2000-HVG space of State-Replogle-Filtered (obsm X_hvg).
Outputs:
  results/per_unit_scores.csv  -- per-perturbation pearson_delta / mse_delta for STATE, R1, R2
  results/deltas.npz           -- stored delta vectors (true, STATE, R1, R2)
"""
import anndata as ad
import numpy as np
import pandas as pd

H5AD = "data/replogle_concat.h5ad"
PRED = "results/k562_pred_best.h5ad"
SUPP = "data/training_support_k562_perts.csv"

print("loading metadata ...")
obs = pd.read_parquet("data/obs_meta.parquet")
obs["gene"] = obs["gene"].astype(str)
obs["cell_line"] = obs["cell_line"].astype(str)

print("loading X_hvg (all cells) ...")
a = ad.read_h5ad(H5AD, backed="r")
Xh = np.asarray(a.obsm["X_hvg"][:], dtype=np.float32)
print("X_hvg:", Xh.shape)

is_k562 = (obs["cell_line"] == "k562").values
is_ctrl = (obs["gene"] == "non-targeting").values

# ---- true k562 deltas ----
k_idx = np.where(is_k562)[0]
k_ctrl_idx = np.where(is_k562 & is_ctrl)[0]
true_ctrl_mean = Xh[k_ctrl_idx].mean(axis=0)

k_perts = sorted(obs["gene"].values[k_idx][obs["gene"].values[k_idx] != "non-targeting"].tolist().__class__(set(obs["gene"].values[k_idx]) - {"non-targeting"}))
true_delta = {}
n_cells = {}
for p in k_perts:
    ii = k_idx[(obs["gene"].values[k_idx] == p)]
    true_delta[p] = Xh[ii].mean(axis=0) - true_ctrl_mean
    n_cells[p] = len(ii)

# ---- STATE predicted deltas ----
print("loading predictions ...")
pa = ad.read_h5ad(PRED, backed="r")
pobs = pa.obs[["gene"]].copy()
pobs["gene"] = pobs["gene"].astype(str)
Ph = np.asarray(pa.obsm["X_hvg"][:], dtype=np.float32)
pgenes = pobs["gene"].values
pred_ctrl_mean = Ph[pgenes == "non-targeting"].mean(axis=0)
state_delta = {}
for p in k_perts:
    state_delta[p] = Ph[pgenes == p].mean(axis=0) - pred_ctrl_mean

# ---- references from training lines ----
print("computing references ...")
train_lines = ["jurkat", "hepg2", "rpe1"]
line_delta = {}   # (line, pert) -> delta
line_has = {}
for line in train_lines:
    l_idx = np.where((obs["cell_line"] == line).values)[0]
    l_ctrl = l_idx[obs["gene"].values[l_idx] == "non-targeting"]
    l_mean_ctrl = Xh[l_ctrl].mean(axis=0)
    genes_l = obs["gene"].values[l_idx]
    for p in set(genes_l) - {"non-targeting"}:
        line_delta[(line, p)] = Xh[l_idx[genes_l == p]].mean(axis=0) - l_mean_ctrl
        line_has.setdefault(p, []).append(line)

all_train_deltas = np.stack(list(line_delta.values()))
R2 = all_train_deltas.mean(axis=0)  # global training mean delta

r1_delta, r2_delta = {}, {}
for p in k_perts:
    if p in line_has:
        r1_delta[p] = np.stack([line_delta[(l, p)] for l in line_has[p]]).mean(axis=0)
    else:
        r1_delta[p] = R2.copy()  # frozen fallback
    r2_delta[p] = R2.copy()

# ---- metrics ----
supp = pd.read_csv(SUPP, index_col=0).iloc[:, 0]
rows = []
for p in k_perts:
    td, sd, r1d, r2d = true_delta[p], state_delta[p], r1_delta[p], r2_delta[p]
    def pear(x, y):
        sx, sy = x.std(), y.std()
        return float(np.corrcoef(x, y)[0, 1]) if sx > 0 and sy > 0 else np.nan
    rows.append(dict(
        pert=p, support=int(supp.get(p, 0)), n_k562_cells=n_cells[p],
        n_train_lines=len(line_has.get(p, [])),
        true_l2=float(np.linalg.norm(td)),
        state_pearson_delta=pear(sd, td), state_mse_delta=float(np.mean((sd - td) ** 2)),
        r1_pearson_delta=pear(r1d, td), r1_mse_delta=float(np.mean((r1d - td) ** 2)),
        r2_pearson_delta=pear(r2d, td), r2_mse_delta=float(np.mean((r2d - td) ** 2)),
    ))
df = pd.DataFrame(rows)
df["level"] = pd.cut(df.support, bins=[-1, 0, 100, 250, 10**9], labels=["L0", "L1", "L2", "L3"])
df.to_csv("results/per_unit_scores.csv", index=False)

np.savez_compressed(
    "results/deltas.npz",
    perts=np.array(k_perts),
    true=np.stack([true_delta[p] for p in k_perts]),
    state=np.stack([state_delta[p] for p in k_perts]),
    r1=np.stack([r1_delta[p] for p in k_perts]),
    r2=np.stack([r2_delta[p] for p in k_perts]),
)
print(df.groupby("level", observed=True)[["state_pearson_delta", "r1_pearson_delta", "r2_pearson_delta"]].mean())
print("saved results/per_unit_scores.csv and results/deltas.npz")
