"""Frozen input transforms for the SPECTRA hue-rotation audit of CONCH.

All transforms act on uint8 RGB numpy arrays (H, W, 3) and are pure functions of
the axis parameter -- no label, model output, or dataset statistics is used.
Axis definitions are frozen in AXIS_SPEC.md (2026-07-22).
"""
import cv2
import numpy as np


def hue_rotate(img: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate hue by theta_deg color-wheel degrees (OpenCV H in [0,179])."""
    if theta_deg == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    shift = int(round(theta_deg / 2.0))  # 180 OpenCV units = 360 degrees
    hsv[..., 0] = (hsv[..., 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def sat_scale(img: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 1:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * alpha, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def val_scale(img: np.ndarray, beta: float) -> np.ndarray:
    if beta == 1:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * beta, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ---- prospective, model-free dissimilarity proxies -------------------------

def mean_delta_e(clean: np.ndarray, transformed: np.ndarray) -> float:
    """Mean CIELAB DeltaE76 between two uint8 RGB images."""
    lab_a = cv2.cvtColor(clean, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(transformed, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV scales L to [0,255]; rescale to conventional CIELAB ranges
    scale = np.array([100.0 / 255.0, 1.0, 1.0], dtype=np.float32)
    d = (lab_a - lab_b) * scale
    return float(np.sqrt((d ** 2).sum(-1)).mean())


def hue_hist_distance(clean: np.ndarray, transformed: np.ndarray, bins: int = 36) -> float:
    """1 - histogram intersection of 1-D hue histograms (S>20 px only)."""
    def hist(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = hsv[..., 1] > 20
        h, _ = np.histogram(hsv[..., 0][mask], bins=bins, range=(0, 180), density=True)
        return h
    a, b = hist(clean), hist(transformed)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    a, b = a / a.sum(), b / b.sum()
    return float(1.0 - np.minimum(a, b).sum())


TRANSFORMS = {"hue": hue_rotate, "sat": sat_scale, "val": val_scale}
