#!/usr/bin/env python
"""r2 step 07: add per-variant AlphaFold pLDDT (structural reliability feature).
For each eval variant, parse protein position from AM protein_variant, look up pLDDT in the
cached AlphaFold DB confidence JSON (v6), average over the variant's AM rows.
Output: tables/r2_eval_table2.tsv (eval table + plddt column)
"""
import json, os, re
import pandas as pd
import numpy as np

am = pd.read_csv("tables/r2_am_hits2.tsv", sep="\t",
                 names=["chrom","pos","ref","alt","uniprot","protein_variant","am"],
                 dtype={"chrom":str})
am["key"] = am.chrom + ":" + am.pos.astype(str) + ":" + am.ref + ":" + am.alt
m = am.protein_variant.str.extract(r"^[A-Z][a-z]{0,2}(\d+)[A-Z][a-z]{0,2}$")
am["res"] = pd.to_numeric(m[0], errors="coerce")
am = am.dropna(subset=["res"])
am["res"] = am["res"].astype(int)

# load AF cache index
cache = {}
missing_up = set()
def load_plddt(up):
    if up in cache: return cache[up]
    if up in missing_up: return None
    p = f"data/af/{up}.json"
    if os.path.exists(p):
        d = json.load(open(p))
        arr = dict(zip(d["residueNumber"], d["confidenceScore"]))
        cache[up] = arr
        return arr
    missing_up.add(up)
    return None

pls, oob = [], 0
for up, res in zip(am.uniprot, am.res):
    arr = load_plddt(up)
    if arr is None:
        pls.append(np.nan)
    else:
        v = arr.get(res)
        if v is None:
            oob += 1
            pls.append(np.nan)
        else:
            pls.append(v)
am["plddt"] = pls
print(f"rows={len(am)} missing_uniprot={len(missing_up)} out_of_range={oob} nan_plddt={am.plddt.isna().sum()}")

agg = am.groupby("key").plddt.mean().rename("plddt")
df = pd.read_csv("tables/r2_eval_table.tsv", sep="\t")
df = df.merge(agg, on="key", how="left")
df.to_csv("tables/r2_eval_table2.tsv", sep="\t", index=False)
print(f"eval variants={len(df)} with plddt={df.plddt.notna().sum()}")
print(df.plddt.describe())
# feature-space bin counts (pre-scoring)
b = pd.cut(df.plddt, [0,50,70,90,100], right=False, labels=["S1<50","S2[50,70)","S3[70,90)","S4>=90"])
print(df.groupby(b, observed=True).agg(n=("label","size"), n_path=("label","sum")))
