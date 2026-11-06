"""Score a tile manifest with CONCH zero-shot across a set of Gaussian-blur levels.

Model and zero-shot text classifier are loaded once. For each sigma level, every
tile is blurred in-memory (isotropic Gaussian, kernel = 2*round(3*sigma)+1) and
scored. The SAME tiles are used at every level, so tissue-class composition is
held constant across levels by construction (removes the class confound). Also
records measured sharpness (variance of Laplacian) and mean stain/color per tile
per level, so the axis order and the stain-vs-blur control can be checked.

Outputs long-format CSV: one row per (tile, sigma) with correctness, measured
sharpness, and color summaries. No input image file is modified on disk.
"""
import sys, csv, argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conch_infer import load_model, build_classifier, CLASSES, CLASS_TO_IDX


def blur_rgb(img_rgb, sigma):
    if sigma <= 0:
        return img_rgb
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img_rgb, (k, k), sigma)


@torch.no_grad()
def score_level(model, preprocess, classifier, imgs_rgb, device, batch=32):
    temp = model.logit_scale.exp().item()
    logits = []
    buf = []
    def flush(buf):
        x = torch.stack(buf, 0).to(device)
        feats = model.encode_image(x, proj_contrast=True, normalize=True)
        return (feats @ classifier).cpu().numpy(), feats.cpu().numpy()
    feat_all = []
    for im in imgs_rgb:
        buf.append(preprocess(Image.fromarray(im)))
        if len(buf) == batch:
            lg, ft = flush(buf); logits.append(lg); feat_all.append(ft); buf = []
    if buf:
        lg, ft = flush(buf); logits.append(lg); feat_all.append(ft)
    logits = np.concatenate(logits, 0)
    feats = np.concatenate(feat_all, 0)
    preds = logits.argmax(1)
    return preds, logits, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigmas", required=True, help="comma list, e.g. 0,1,2,3")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--save_feats", default="", help="optional .npz to save CONCH embeddings per level")
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]

    rows = list(csv.DictReader(open(args.manifest)))
    paths = [r["image_abspath"] for r in rows]
    labels = [CLASS_TO_IDX[r["label_name"]] for r in rows]

    # preload raw RGB once
    raw = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        raw.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    model, preprocess = load_model(device="cpu")
    classifier = build_classifier(model, device="cpu")

    out = []
    feats_store = {}
    for sg in sigmas:
        imgs = [blur_rgb(im, sg) for im in raw]
        preds, logits, feats = score_level(model, preprocess, classifier, imgs, "cpu", args.batch)
        feats_store[f"sigma_{sg}"] = feats
        for i, r in enumerate(rows):
            g = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY).astype(np.float64)
            varlap = float(cv2.Laplacian(g, cv2.CV_64F).var())
            hsv = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2HSV)
            out.append({
                "sample_id": r.get("sample_id", str(i)),
                "image_abspath": r["image_abspath"],
                "label_name": r["label_name"],
                "label": labels[i],
                "sigma": sg,
                "pred": int(preds[i]),
                "pred_label_name": CLASSES[int(preds[i])],
                "correct": int(preds[i] == labels[i]),
                "varlap": varlap,
                "brightness": float(g.mean()),
                "saturation_mean": float(hsv[:, :, 1].mean()),
            })
        acc = np.mean([preds[i] == labels[i] for i in range(len(rows))])
        print(f"  sigma={sg}  n={len(rows)}  acc={acc:.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    if args.save_feats:
        np.savez_compressed(args.save_feats, labels=np.array(labels), **feats_store)
    print("WROTE", args.out, "rows", len(out))


if __name__ == "__main__":
    main()
