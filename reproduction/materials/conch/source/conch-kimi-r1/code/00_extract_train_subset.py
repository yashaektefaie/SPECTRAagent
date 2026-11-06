"""Extract a bounded stratified subset of NCT-CRC-HE-100K-NONORM from the zip
(disk-constrained environment); enough for the 25k reference train subset."""
import os
import zipfile
from common import CLASSES, DATA, TRAIN_DIR

ZIP = os.path.join(DATA, "NCT-CRC-HE-100K-NONORM.zip")
PER_CLASS = 3200  # > 2778 needed per class

def main():
    zf = zipfile.ZipFile(ZIP)
    names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
    by_class = {c: sorted([n for n in names if f"/{c}/" in n]) for c in CLASSES}
    for c in CLASSES:
        print(c, len(by_class[c]))
    for c in CLASSES:
        d = os.path.join(TRAIN_DIR, c)
        os.makedirs(d, exist_ok=True)
        for n in by_class[c][:PER_CLASS]:
            zf.extract(n, DATA)
    print("extracted")

if __name__ == "__main__":
    main()
