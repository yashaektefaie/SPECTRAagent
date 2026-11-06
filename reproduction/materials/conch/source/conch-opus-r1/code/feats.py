"""Shared prospective pixel-feature extraction for SPECTRA CONCH audit.
ALL features are computed from PIXELS ONLY. No logits/probs/labels are used.

Two distinct feature groups:
  - COLOR/STAIN novelty features  -> used to build the train-novelty (covariate-shift) axis
  - SHARPNESS/BLUR features        -> used ONLY as the confound axis (kept separate)
"""
import numpy as np
from PIL import Image
from skimage.color import rgb2hed, rgb2hsv
from scipy.ndimage import laplace, sobel

# canonical novelty feature order (color/stain only; NO sharpness)
NOVELTY_FEATS = [
    "mean_R","mean_G","mean_B","std_R","std_G","std_B",
    "mean_H","mean_E","std_H","std_E",
    "mean_sat","std_sat","mean_val","std_val",
    "colorfulness","bg_frac",
]
BLUR_FEATS = ["lap_var","tenengrad","hf_ratio"]


def load_rgb(path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im, dtype=np.uint8)
    return a


def compute_features(rgb):
    """rgb: HxWx3 uint8. Returns dict of all features."""
    a = rgb.astype(np.float32) / 255.0
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    f = {}
    f["mean_R"], f["mean_G"], f["mean_B"] = float(R.mean()), float(G.mean()), float(B.mean())
    f["std_R"], f["std_G"], f["std_B"] = float(R.std()), float(G.std()), float(B.std())
    # H&E color deconvolution (hematoxylin / eosin)
    hed = rgb2hed(a)
    H, E = hed[..., 0], hed[..., 1]
    f["mean_H"], f["mean_E"] = float(H.mean()), float(E.mean())
    f["std_H"], f["std_E"] = float(H.std()), float(E.std())
    # HSV saturation / value
    hsv = rgb2hsv(a)
    S, V = hsv[..., 1], hsv[..., 2]
    f["mean_sat"], f["std_sat"] = float(S.mean()), float(S.std())
    f["mean_val"], f["std_val"] = float(V.mean()), float(V.std())
    # Hasler-Susstrunk colorfulness
    rg = R - G
    yb = 0.5 * (R + G) - B
    f["colorfulness"] = float(np.sqrt(rg.std()**2 + yb.std()**2) + 0.3*np.sqrt(rg.mean()**2 + yb.mean()**2))
    # background (near-white) fraction
    f["bg_frac"] = float(np.mean((R > 0.86) & (G > 0.86) & (B > 0.86)))
    # ---- sharpness / blur (separate group) ----
    gray = (0.299*R + 0.587*G + 0.114*B)
    lap = laplace(gray)
    f["lap_var"] = float(lap.var())
    gx = sobel(gray, axis=0); gy = sobel(gray, axis=1)
    f["tenengrad"] = float(np.mean(gx*gx + gy*gy))
    # high-frequency energy ratio via FFT
    Ft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(Ft)
    h, w = gray.shape
    cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    hf = mag[r > min(h, w)*0.25].sum()
    tot = mag.sum() + 1e-9
    f["hf_ratio"] = float(hf / tot)
    return f


def features_for_path(path):
    try:
        rgb = load_rgb(path)
        return compute_features(rgb)
    except Exception as e:
        return {"_error": str(e)}
