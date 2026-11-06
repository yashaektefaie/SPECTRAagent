"""Extract a prospective, seed-fixed train/eval sample from NCT-CRC-HE-100K.zip.
Selection uses ONLY filenames + seed (no labels/outcomes beyond stratification by class dir).
Splits: dinov2-probe train = 500/class; eval = 400/class (disjoint); buffer = 100/class."""
import os, sys, zipfile, hashlib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(ROOT, "data", "NCT-CRC-HE-100K.zip")
OUTDIR = os.path.join(ROOT, "data", "NCT-sample")
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
SEED = 20260722
N_TRAIN, N_EVAL, N_BUF = 500, 400, 100

zf = zipfile.ZipFile(ZIP)
names = zf.namelist()
by_class = {c: sorted(n for n in names if n.startswith(f"NCT-CRC-HE-100K/{c}/") and n.lower().endswith((".tif", ".png")))
            for c in CLASSES}
for c in CLASSES:
    print(c, len(by_class[c]))

rng = np.random.RandomState(SEED)
rows = []
for ci, c in enumerate(CLASSES):
    pool = by_class[c]
    perm = rng.permutation(len(pool))
    for split, sl in [("train_probe", perm[:N_TRAIN]),
                      ("eval", perm[N_TRAIN:N_TRAIN + N_EVAL]),
                      ("buffer", perm[N_TRAIN + N_EVAL:N_TRAIN + N_EVAL + N_BUF])]:
        for j in sl:
            rows.append({"zip_entry": pool[j], "class": c, "label": ci, "split": split})
df = pd.DataFrame(rows)
os.makedirs(OUTDIR, exist_ok=True)

paths = []
for _, r in df.iterrows():
    rel = os.path.join(r["split"], r["class"], os.path.basename(r["zip_entry"]))
    out = os.path.join(OUTDIR, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.exists(out):
        with zf.open(r["zip_entry"]) as src, open(out, "wb") as dst:
            dst.write(src.read())
    paths.append(out)
df["path"] = paths

def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
df["md5"] = df["path"].map(md5)
n_dup = int(df["md5"].duplicated().sum())
print("exact md5 duplicates across extracted sample:", n_dup)
df.to_csv(os.path.join(OUTDIR, "nct_sample_manifest.csv"), index=False)
print(df.groupby(["split", "class"]).size().unstack())
print("total extracted:", len(df))
