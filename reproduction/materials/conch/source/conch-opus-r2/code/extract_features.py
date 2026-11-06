"""Input-side (prospective) image-property feature extraction for CRC H&E tiles.

Features are computed from pixels only; they never see CONCH outputs or labels.
Used for (a) development discovery screening and (b) applying frozen axis levels
to confirmation tiles. Deterministic.

Features:
  sharpness_varlap   : variance of the Laplacian (focus / blur; higher = sharper)
  sharpness_tenengrad: mean squared Sobel gradient magnitude (focus)
  highfreq_frac      : fraction of 2D FFT power above a mid radius (focus)
  brightness         : mean luminance (0-255)
  contrast_std       : std of luminance
  saturation_mean    : mean HSV saturation (0-255)
  colorfulness       : Hasler-Susstrunk colorfulness
  tissue_frac        : fraction of pixels that are tissue (not white background)
  hematoxylin_mean   : mean hematoxylin OD after color deconvolution
  eosin_mean         : mean eosin OD after color deconvolution
  he_ratio           : hematoxylin_mean / (eosin_mean+eps)
  entropy            : Shannon entropy of the grayscale histogram
"""
import sys, csv, argparse
from pathlib import Path
import numpy as np
import cv2
from skimage.color import rgb2hed


def features_for(img_rgb):
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gf = g.astype(np.float64)
    # focus / sharpness
    lap = cv2.Laplacian(gf, cv2.CV_64F)
    varlap = float(lap.var())
    gx = cv2.Sobel(gf, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    F = np.fft.fftshift(np.fft.fft2(gf))
    mag = np.abs(F)
    h, w = gf.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rad = min(h, w) / 8.0  # mid-frequency cutoff
    total = mag.sum() + 1e-9
    highfreq_frac = float(mag[r > rad].sum() / total)
    # intensity
    brightness = float(gf.mean())
    contrast_std = float(gf.std())
    # color
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation_mean = float(hsv[:, :, 1].mean())
    R = img_rgb[:, :, 0].astype(np.float64)
    G = img_rgb[:, :, 1].astype(np.float64)
    B = img_rgb[:, :, 2].astype(np.float64)
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    # tissue fraction: not near-white and some saturation
    white = (gf > 220) & (hsv[:, :, 1] < 25)
    tissue_frac = float(1.0 - white.mean())
    # entropy
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    # stain color deconvolution (H&E)
    hed = rgb2hed(img_rgb.astype(np.float64) / 255.0)
    hematoxylin_mean = float(hed[:, :, 0].mean())
    eosin_mean = float(hed[:, :, 1].mean())
    he_ratio = float(hematoxylin_mean / (abs(eosin_mean) + 1e-6))
    return {
        "sharpness_varlap": varlap,
        "sharpness_tenengrad": tenengrad,
        "highfreq_frac": highfreq_frac,
        "brightness": brightness,
        "contrast_std": contrast_std,
        "saturation_mean": saturation_mean,
        "colorfulness": colorfulness,
        "tissue_frac": tissue_frac,
        "hematoxylin_mean": hematoxylin_mean,
        "eosin_mean": eosin_mean,
        "he_ratio": he_ratio,
        "entropy": entropy,
    }


FEATS = ["sharpness_varlap", "sharpness_tenengrad", "highfreq_frac", "brightness",
         "contrast_std", "saturation_mean", "colorfulness", "tissue_frac",
         "hematoxylin_mean", "eosin_mean", "he_ratio", "entropy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with image_abspath[,label_name,sample_id]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blur_sigma", type=float, default=0.0,
                    help="optional Gaussian blur sigma applied before feature extraction")
    args = ap.parse_args()
    rows = []
    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        img = cv2.imread(r["image_abspath"], cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit("cannot read " + r["image_abspath"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if args.blur_sigma > 0:
            k = max(3, int(2 * round(3 * args.blur_sigma) + 1))
            img = cv2.GaussianBlur(img, (k, k), args.blur_sigma)
        fe = features_for(img)
        rec = {k: r[k] for k in r}
        rec.update(fe)
        out.append(rec)
        if i % 500 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print("WROTE", args.out, "N=", len(out))


if __name__ == "__main__":
    main()
