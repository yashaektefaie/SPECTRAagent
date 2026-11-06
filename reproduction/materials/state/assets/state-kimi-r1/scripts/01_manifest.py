#!/usr/bin/env python
"""Build the frozen-axis manifest for the SPECTRA audit of ST-HVG-Replogle fewshot/k562.

Axis A1 levels from training-config membership ONLY (no outcome data):
  S(g): 1 if g is a k562 training perturbation, 0 if in the frozen k562 test list
  E(g): number of other lines (rpe1, jurkat, hepg2) where g is present as a
        perturbation in the training data (all of their perts are train in this run)
Writes: results/axis_manifest.csv  (one row per k562 unit)
"""
import sys
import tomllib
import pandas as pd
import anndata as ad

REPO = "code/state/state_preprint_toml_files/replogle_tomls"
DATA = "data/replogle/replogle_concat.h5ad"
CONTROL = "non-targeting"

def parse_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)

def test_list(toml, line):
    fs = toml.get("fewshot", {})
    return set(fs.get(f"replogle.{line}", {}).get("test", []))

def main():
    k562_test = test_list(parse_toml(f"{REPO}/k562.toml"), "k562")
    print(f"k562 frozen test list: {len(k562_test)} perts (incl control: {CONTROL in k562_test})")

    # observed perturbation membership per line from the data itself (obs only)
    a = ad.read_h5ad(DATA, backed="r")
    obs = a.obs[["gene", "cell_line"]].to_memory() if hasattr(a.obs, 'to_memory') else a.obs[["gene", "cell_line"]]
    obs["gene"] = obs["gene"].astype(str)
    obs["cell_line"] = obs["cell_line"].astype(str).str.lower()
    print(obs["cell_line"].value_counts())

    lines = ["rpe1", "jurkat", "hepg2"]
    perts_by_line = {l: set(obs.loc[obs.cell_line == l, "gene"].unique()) for l in lines}
    k562_perts = set(obs.loc[obs.cell_line == "k562", "gene"].unique())
    print({l: len(s) for l, s in perts_by_line.items()}, "k562:", len(k562_perts))

    rows = []
    for g in sorted(k562_perts - {CONTROL}):
        S = 0 if g in k562_test else 1
        E = sum(g in perts_by_line[l] for l in lines)
        if S == 1:
            level = "L4_seen_pair"
        else:
            level = {3: "L3_unseen_E3", 2: "L2_unseen_E2", 1: "L1_unseen_E1", 0: "L0_unseen_E0"}[E]
        rows.append(dict(gene=g, seen_pair=S, exposure_E=E, level=level))
    man = pd.DataFrame(rows)
    print(man["level"].value_counts())
    man.to_csv("results/axis_manifest.csv", index=False)
    print("wrote results/axis_manifest.csv", len(man), "units")

if __name__ == "__main__":
    main()
