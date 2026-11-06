"""Compute prospective pixel features for a set of tiles in parallel.
Usage: python 03_extract_features.py <manifest.csv> <out.csv>
manifest.csv columns: id,path,cls
"""
import sys, os, csv, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from feats import features_for_path, NOVELTY_FEATS, BLUR_FEATS

ALLF = NOVELTY_FEATS + BLUR_FEATS

def work(row):
    _id, path, cls = row
    f = features_for_path(path)
    f["id"] = _id; f["cls"] = cls; f["path"] = path
    return f

def main():
    man = pd.read_csv(sys.argv[1])
    rows = list(zip(man["id"], man["path"], man["cls"]))
    print(f"{len(rows)} tiles -> {sys.argv[2]}", flush=True)
    out = []
    with Pool(10) as p:
        for i, f in enumerate(p.imap_unordered(work, rows, chunksize=32)):
            out.append(f)
            if (i+1) % 1000 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
    df = pd.DataFrame(out)
    nerr = df["_error"].notna().sum() if "_error" in df else 0
    print("errors:", int(nerr))
    cols = ["id","cls"] + ALLF + ["path"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(sys.argv[2], index=False)
    print("wrote", sys.argv[2], df.shape)

if __name__ == "__main__":
    main()
