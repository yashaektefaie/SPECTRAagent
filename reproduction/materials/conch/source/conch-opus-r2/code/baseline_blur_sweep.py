"""Development-only estimate of the fixed baseline's blur sensitivity (for Test B
power). Runs ResNet50+frozen-head across blur sigmas on a manifest of dev tiles.
Exploration only; does not touch confirmation outcomes.
"""
import sys, csv, argparse, json
from pathlib import Path
import numpy as np
import cv2
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import baseline_model as bm
from conch_infer import CLASS_TO_IDX


def blur_rgb(img, sigma):
    if sigma <= 0:
        return img
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--sigmas", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    rows = list(csv.DictReader(open(args.manifest)))
    labels = np.array([CLASS_TO_IDX[r["label_name"]] for r in rows])
    raw = [cv2.cvtColor(cv2.imread(r["image_abspath"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for r in rows]
    net = bm.load_backbone()
    clf = joblib.load(args.head)
    out = {"sigmas": sigmas, "per_level_acc": {}, "per_tile": []}
    corr_by_sigma = {}
    for sg in sigmas:
        imgs = [blur_rgb(im, sg) for im in raw]
        feats = bm.extract_features(net, imgs)
        pred = clf.predict(feats)
        corr = (pred == labels).astype(int)
        corr_by_sigma[sg] = corr
        out["per_level_acc"][str(sg)] = float(corr.mean())
        print(f"  baseline sigma={sg}: acc={corr.mean():.4f}", flush=True)
    # paired drop sigma0->max
    s0, sm = min(sigmas), max(sigmas)
    drop = float((corr_by_sigma[s0] - corr_by_sigma[sm]).mean())
    out["baseline_drop_sigma0_to_max"] = drop
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"baseline drop {s0}->{sm} = {drop:.4f}; WROTE {args.out}")


if __name__ == "__main__":
    main()
