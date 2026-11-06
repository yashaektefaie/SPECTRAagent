#!/usr/bin/env python
"""Build the frozen-axis manifest for a confirmation context of ST-HVG-Replogle.

Identical axis definition as A1 (FROZEN_AXIS.md addendum, confirmation axes
A1b/A1c): for run fewshot/<ctx>,
  S(g): 1 if g is a <ctx> training perturbation, 0 if in the frozen <ctx>
        fewshot test list
  E(g): number of the other 3 cell lines where g is present as a perturbation
Uses training-config membership and data obs ONLY (no outcome data).
Writes: results/axis_manifest_<ctx>.csv
"""
import sys
import tomllib
import pandas as pd
import anndata as ad

CTX = sys.argv[1]  # rpe1 | jurkat | hepg2
DATA = "data/replogle/replogle_concat.h5ad"
CONTROL = "non-targeting"
ALL_LINES = ["rpe1", "jurkat", "hepg2", "k562"]


def main():
    with open(f"configs/local_{CTX}.toml", "rb") as f:
        toml = tomllib.load(f)
    test = set(toml["fewshot"][f"replogle.{CTX}"]["test"]) - {CONTROL}
    print(f"{CTX} frozen test list: {len(test)} perts (excl control)")

    a = ad.read_h5ad(DATA, backed="r")
    obs = a.obs[["gene", "cell_line"]]
    obs = obs.to_memory() if hasattr(obs, "to_memory") else obs
    obs["gene"] = obs["gene"].astype(str)
    obs["cell_line"] = obs["cell_line"].astype(str).str.lower()

    others = [l for l in ALL_LINES if l != CTX]
    perts_by_line = {l: set(obs.loc[obs.cell_line == l, "gene"].unique()) for l in others}
    ctx_perts = set(obs.loc[obs.cell_line == CTX, "gene"].unique()) - {CONTROL}
    print("ctx perts:", len(ctx_perts))

    rows = []
    for g in sorted(ctx_perts):
        S = 0 if g in test else 1
        E = sum(g in perts_by_line[l] for l in others)
        level = "seen_pair" if S == 1 else f"unseen_E{E}"
        rows.append(dict(gene=g, seen_pair=S, exposure_E=E, level=level))
    man = pd.DataFrame(rows)
    print(man["level"].value_counts())
    # test-list genes missing from data (report only)
    missing = test - ctx_perts
    print("test-list genes absent from data (excluded):", len(missing), sorted(missing)[:5])
    man.to_csv(f"results/axis_manifest_{CTX}.csv", index=False)
    print(f"wrote results/axis_manifest_{CTX}.csv", len(man), "units")


if __name__ == "__main__":
    main()
