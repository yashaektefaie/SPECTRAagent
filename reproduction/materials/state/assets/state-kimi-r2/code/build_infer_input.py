"""Build k562-only inference input h5ad from State-Replogle-Filtered.

Input for `state tx infer`: obs[gene, cell_line, gem_group], obsm[X_state, X_hvg].
X is a (n,1) placeholder (untouched when --embed-key is used).
"""
import anndata as ad
import numpy as np
import pandas as pd

SRC = "data/replogle_concat.h5ad"
OUT = "data/k562_infer_input.h5ad"

a = ad.read_h5ad(SRC, backed="r")
mask = (a.obs["cell_line"].astype(str) == "k562").values
idx = np.where(mask)[0]
print("k562 cells:", len(idx))

obs = a.obs.iloc[idx][["gem_group", "gene", "cell_line", "UMI_count", "mitopercent"]].copy()
obs["gem_group"] = obs["gem_group"].astype(str)
obs["gene"] = obs["gene"].astype(str)
obs["cell_line"] = obs["cell_line"].astype(str)

Xs = a.obsm["X_state"][idx, :]
Xh = a.obsm["X_hvg"][idx, :]
Xh = np.asarray(Xh)
Xs = np.asarray(Xs)

out = ad.AnnData(
    X=np.zeros((len(idx), 1), dtype=np.float32),
    obs=obs,
    obsm={"X_state": Xs.astype(np.float32), "X_hvg": Xh.astype(np.float32)},
)
out.obs_names = [f"k562_{i}" for i in idx]
out.write_h5ad(OUT)
print("wrote", OUT, out.shape)
