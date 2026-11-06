"""Build per-context inference inputs and run all three confirmations end-to-end."""
import subprocess, sys
import anndata as ad
import numpy as np
import pandas as pd

SRC = "data/replogle_concat.h5ad"
a = ad.read_h5ad(SRC, backed="r")
cell_lines = a.obs["cell_line"].astype(str).values

for ct in ["jurkat", "hepg2", "rpe1"]:
    out_path = f"data/{ct}_infer_input.h5ad"
    idx = np.where(cell_lines == ct)[0]
    obs = a.obs.iloc[idx][["gem_group", "gene", "cell_line", "UMI_count", "mitopercent"]].copy()
    for c in ["gem_group", "gene", "cell_line"]:
        obs[c] = obs[c].astype(str)
    Xs = np.asarray(a.obsm["X_state"][idx, :], dtype=np.float32)
    Xh = np.asarray(a.obsm["X_hvg"][idx, :], dtype=np.float32)
    out = ad.AnnData(X=np.zeros((len(idx), 1), dtype=np.float32), obs=obs,
                     obsm={"X_state": Xs, "X_hvg": Xh})
    out.obs_names = [f"{ct}_{i}" for i in idx]
    out.write_h5ad(out_path)
    print("wrote", out_path, out.shape, flush=True)
