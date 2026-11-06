"""Index CRC-VAL-HE-7K test set + NCT train set; dedup by pixel hash; overlap check."""
import hashlib
import os
import numpy as np
import pandas as pd
from PIL import Image
from common import ART, CLASSES, DATA, TEST_DIR, TRAIN_DIR, SEED

Image.MAX_IMAGE_PIXELS = None


def index_dir(root):
    rows = []
    for cls in CLASSES:
        d = os.path.join(root, cls)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".tif", ".tiff", ".png")):
                rows.append({"path": os.path.join(d, fn), "label": cls})
    return pd.DataFrame(rows)


def pixel_hash(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def main():
    test = index_dir(TEST_DIR)
    assert len(test) == 7180, len(test)
    test["pxhash"] = [pixel_hash(p) for p in test["path"]]
    n_dup = test["pxhash"].duplicated().sum()
    print("test images:", len(test), "exact pixel dups:", n_dup)
    test = test.drop_duplicates("pxhash").reset_index(drop=True)
    test.to_csv(os.path.join(ART, "test_index.csv"), index=False)
    print(test["label"].value_counts().to_dict())

    if os.path.isdir(TRAIN_DIR):
        train = index_dir(TRAIN_DIR)
        print("train images:", len(train))
        # deterministic stratified subsample of 25k, identical across all levels
        sub = (train.groupby("label", group_keys=False)
               .apply(lambda g: g.sample(n=min(2778, len(g)), random_state=SEED))
               .reset_index(drop=True))
        print("train subset:", len(sub), sub["label"].value_counts().to_dict())
        sub["pxhash"] = [pixel_hash(p) for p in sub["path"]]
        sub = sub.drop_duplicates("pxhash").reset_index(drop=True)
        overlap = set(sub["pxhash"]) & set(test["pxhash"])
        print("train/test exact pixel overlap in subset:", len(overlap))
        sub.to_csv(os.path.join(ART, "train_subset_index.csv"), index=False)
    else:
        print("TRAIN_DIR not ready yet")


if __name__ == "__main__":
    main()
