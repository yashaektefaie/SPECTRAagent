"""FROZEN confirmation command (invoked once by setup/gate.py confirm).

Runs BOTH the target (CONCH zero-shot) and the fixed model-independent baseline
(ResNet50 + frozen logreg head) on the IDENTICAL confirmation tiles across ALL
frozen blur levels. No outcome-adaptive filtering, no level change, no baseline
swap, no retries. For each (tile, sigma) it records:
  - CONCH correctness and baseline correctness
  - measured sharpness (variance of Laplacian) and mean color (axis-order +
    stain-vs-blur control)
  - proximity to the sharp reference manifold in CONCH space and ResNet space
    (max cosine similarity to the per-class reference centroids) -> the SPECTRA
    "measured similarity decreases" check.

Writes per-tile long CSV and a per-level SPC summary with bootstrap CIs. Input
images are never modified on disk.
"""
import sys, csv, json, argparse, datetime
from pathlib import Path
import numpy as np
import cv2
import joblib
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conch_infer import load_model, build_classifier, CLASSES, CLASS_TO_IDX
import baseline_model as bm


def blur_rgb(img, sigma):
    if sigma <= 0:
        return img
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigma)


@torch.no_grad()
def conch_score(model, preprocess, classifier, imgs, batch=32):
    logits, emb = [], []
    buf = []
    def flush(buf):
        x = torch.stack(buf, 0)
        f = model.encode_image(x, proj_contrast=True, normalize=True)
        return (f @ classifier).cpu().numpy(), f.cpu().numpy()
    for im in imgs:
        buf.append(preprocess(Image.fromarray(im)))
        if len(buf) == batch:
            lg, ff = flush(buf); logits.append(lg); emb.append(ff); buf = []
    if buf:
        lg, ff = flush(buf); logits.append(lg); emb.append(ff)
    logits = np.concatenate(logits, 0)
    emb = np.concatenate(emb, 0)
    return logits.argmax(1), emb


def max_cos_to_centroids(feat, centroids):
    fn = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)
    cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    sims = fn @ cn.T          # [N, 9]
    return sims.max(1)        # proximity to nearest reference class centroid


def boot_ci(mask_correct, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(mask_correct, float)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm_manifest", required=True)
    ap.add_argument("--baseline_head", required=True)
    ap.add_argument("--centroids", required=True)
    ap.add_argument("--sigmas", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.confirm_manifest)))
    paths = [r["image_abspath"] for r in rows]
    labels = np.array([CLASS_TO_IDX[r["label_name"]] for r in rows])
    raw = [cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in paths]

    model, preprocess = load_model()
    classifier = build_classifier(model)
    net = bm.load_backbone()
    clf = joblib.load(args.baseline_head)
    cz = np.load(args.centroids, allow_pickle=True)
    conch_cent = cz["conch_centroids"]
    resnet_cent = cz["resnet_centroids"]

    per_tile = []
    summary = {"sigmas": sigmas, "n_per_level": len(rows), "levels": [],
               "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    for sg in sigmas:
        imgs = [blur_rgb(im, sg) for im in raw]
        cpred, cemb = conch_score(model, preprocess, classifier, imgs, args.batch)
        rfeat = bm.extract_features(net, imgs, batch=args.batch)
        bpred = clf.predict(rfeat)
        ccorr = (cpred == labels).astype(int)
        bcorr = (bpred == labels).astype(int)
        conch_prox = max_cos_to_centroids(cemb, conch_cent)
        resnet_prox = max_cos_to_centroids(rfeat, resnet_cent)
        varlaps = []
        for i, r in enumerate(rows):
            g = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY).astype(np.float64)
            vl = float(cv2.Laplacian(g, cv2.CV_64F).var())
            varlaps.append(vl)
            hsv = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2HSV)
            per_tile.append({
                "sample_id": r.get("sample_id", str(i)),
                "image_abspath": r["image_abspath"], "label_name": r["label_name"],
                "label": int(labels[i]), "sigma": sg,
                "conch_pred": int(cpred[i]), "conch_correct": int(ccorr[i]),
                "baseline_pred": int(bpred[i]), "baseline_correct": int(bcorr[i]),
                "varlap": vl,
                "conch_prox": float(conch_prox[i]), "resnet_prox": float(resnet_prox[i]),
                "brightness": float(g.mean()), "saturation_mean": float(hsv[:, :, 1].mean()),
            })
        lvl = {
            "sigma": sg,
            "conch_acc": float(ccorr.mean()), "conch_ci": boot_ci(ccorr),
            "baseline_acc": float(bcorr.mean()), "baseline_ci": boot_ci(bcorr),
            "mean_varlap": float(np.mean(varlaps)),
            "mean_conch_prox": float(conch_prox.mean()),
            "mean_resnet_prox": float(resnet_prox.mean()),
            "mean_brightness": float(np.mean([p["brightness"] for p in per_tile if p["sigma"] == sg])),
            "mean_saturation": float(np.mean([p["saturation_mean"] for p in per_tile if p["sigma"] == sg])),
        }
        summary["levels"].append(lvl)
        print(f"sigma={sg}: CONCH acc={lvl['conch_acc']:.4f} baseline acc={lvl['baseline_acc']:.4f} "
              f"varlap={lvl['mean_varlap']:.1f} conch_prox={lvl['mean_conch_prox']:.4f}", flush=True)

    with open(outd / "confirmation_per_tile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tile[0].keys()))
        w.writeheader(); w.writerows(per_tile)
    summary["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(outd / "spc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("WROTE", str(outd / "confirmation_per_tile.csv"), "and spc_summary.json")


if __name__ == "__main__":
    main()
