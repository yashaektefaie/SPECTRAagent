"""Freeze-time preparation of the fixed comparator (runs BEFORE freeze).

Uses ONLY the NCT-CRC-HE-100K reference pool (sharp tiles; disjoint from the
confirmation pool and from the development set). Produces, for the frozen run:
  - confirmation/baseline_head.joblib : logistic-regression head trained on
    ResNet50 features of sharp reference tiles (the model-independent baseline).
  - confirmation/reference_centroids.npz : per-class mean embedding on sharp
    reference tiles in BOTH the CONCH space and the ResNet space, used at
    confirmation time to measure proximity-to-reference (the SPECTRA
    similarity-decreases check). CONCH is never scored on confirmation tiles here.

Nothing about confirmation outcomes is touched. Deterministic.
"""
import sys, csv, argparse
from pathlib import Path
import numpy as np
import cv2
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conch_infer import load_model, build_classifier, CLASS_TO_IDX, CLASSES
import baseline_model as bm
import torch
import torch.nn.functional as F
from PIL import Image


def load_rgb(paths):
    out = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return out


@torch.no_grad()
def conch_embed(model, preprocess, imgs_rgb, batch=32):
    feats = []
    buf = []
    def flush(buf):
        x = torch.stack(buf, 0)
        return model.encode_image(x, proj_contrast=True, normalize=True).cpu().numpy()
    for im in imgs_rgb:
        buf.append(preprocess(Image.fromarray(im)))
        if len(buf) == batch:
            feats.append(flush(buf)); buf = []
    if buf:
        feats.append(flush(buf))
    return np.concatenate(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--out_head", required=True)
    ap.add_argument("--out_centroids", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.reference_manifest)))
    paths = [r["image_abspath"] for r in rows]
    labels = np.array([CLASS_TO_IDX[r["label_name"]] for r in rows])
    imgs = load_rgb(paths)

    # baseline: ResNet features + logreg
    net = bm.load_backbone()
    rfeat = bm.extract_features(net, imgs)
    clf = bm.train_head(rfeat, labels, seed=0)
    train_acc = (clf.predict(rfeat) == labels).mean()
    Path(args.out_head).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out_head)
    print(f"baseline head trained on {len(labels)} ref tiles; ref train_acc={train_acc:.4f}")

    # CONCH embeddings on reference tiles (sharp) -> per-class centroids
    model, preprocess = load_model()
    cemb = conch_embed(model, preprocess, imgs)

    def centroids(feat, lab):
        cs = []
        for c in range(len(CLASSES)):
            m = lab == c
            v = feat[m].mean(0)
            cs.append(v)
        return np.stack(cs, 0)

    conch_cent = centroids(cemb, labels)          # [9, 512] (not renormalized)
    resnet_cent = centroids(rfeat, labels)        # [9, 2048]
    np.savez_compressed(args.out_centroids,
                        conch_centroids=conch_cent, resnet_centroids=resnet_cent,
                        classes=np.array(CLASSES))
    print("WROTE", args.out_head, "and", args.out_centroids)


if __name__ == "__main__":
    main()
