#!/usr/bin/env python3
"""Compute model/label-independent color, texture, focus, and tissue features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]


def entropy(x: np.ndarray, bins: int = 64) -> float:
    h = np.histogram(x, bins=bins, range=(0, 1))[0].astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def one(path_text: str) -> dict[str, float | str]:
    path = ROOT / path_text
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    tissue = (sat > 0.08) & (val < 0.92)
    padded = np.pad(gray, 1, mode="reflect")
    gx = (
        -padded[:-2, :-2] + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:]
        - padded[2:, :-2] + padded[2:, 2:]
    ) / 8.0
    gy = (
        -padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
    ) / 8.0
    mag = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + np.pi) % np.pi
    lap = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2]
        + padded[1:-1, 2:] - 4 * gray
    )
    od = -np.log(np.clip(rgb, 1 / 255.0, 1.0))

    center = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:],
        gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2],
    ]
    code = np.zeros_like(center, dtype=np.uint8)
    for bit, n in enumerate(neighbors):
        code |= ((n >= center).astype(np.uint8) << bit)

    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    power = np.abs(f) ** 2
    yy, xx = np.indices(gray.shape)
    rr = np.hypot(yy - (gray.shape[0] - 1) / 2, xx - (gray.shape[1] - 1) / 2)
    rr /= max(gray.shape) / 2

    dh = np.asarray(Image.fromarray(np.uint8(gray * 255)).resize((9, 8), Image.Resampling.BILINEAR))
    dhash = sum(int(bit) << i for i, bit in enumerate((dh[:, 1:] >= dh[:, :-1]).ravel()))
    row: dict[str, float | str] = {"path": path_text, "tile_id": path.stem, "dhash64": f"{dhash:016x}"}
    for c, name in enumerate(("r", "g", "b")):
        v = rgb[..., c]
        row[f"{name}_mean"] = float(v.mean())
        row[f"{name}_std"] = float(v.std())
        row[f"{name}_q10"] = float(np.quantile(v, 0.10))
        row[f"{name}_q50"] = float(np.quantile(v, 0.50))
        row[f"{name}_q90"] = float(np.quantile(v, 0.90))
        hist = np.histogram(v, bins=16, range=(0, 1), density=False)[0] / v.size
        for i, z in enumerate(hist):
            row[f"{name}_hist_{i:02d}"] = float(z)
    row.update({
        "gray_mean": float(gray.mean()), "gray_std": float(gray.std()),
        "gray_entropy": entropy(gray), "sat_mean": float(sat.mean()),
        "sat_std": float(sat.std()), "val_mean": float(val.mean()),
        "tissue_fraction": float(tissue.mean()), "white_fraction": float((val > 0.95).mean()),
        "focus_laplacian_var": float(lap.var()), "tenengrad_mean": float((mag * mag).mean()),
        "edge_fraction": float((mag > 0.08).mean()), "od_mean": float(od.mean()),
        "od_std": float(od.std()), "od_r_mean": float(od[..., 0].mean()),
        "od_g_mean": float(od[..., 1].mean()), "od_b_mean": float(od[..., 2].mean()),
    })
    lbp = np.histogram(code, bins=16, range=(0, 256))[0] / code.size
    for i, z in enumerate(lbp):
        row[f"lbp_{i:02d}"] = float(z)
    orient = np.histogram(angle, bins=8, range=(0, np.pi), weights=mag)[0]
    orient /= orient.sum() + 1e-12
    for i, z in enumerate(orient):
        row[f"orient_{i:02d}"] = float(z)
    for i, (lo, hi) in enumerate(((0.02, 0.08), (0.08, 0.20), (0.20, 0.45), (0.45, 1.0))):
        mask = (rr >= lo) & (rr < hi)
        row[f"fft_band_{i}"] = float(np.log1p(power[mask].mean()))
    gray_image = Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255)))
    for sigma in (1, 2, 4, 8):
        smooth = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32) / 255.0
        row[f"contrast_s{sigma}"] = float((gray - smooth).std())
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = p.parse_args()
    manifest = ROOT / "data" / "manifests" / f"{args.dataset}.blind.csv"
    with manifest.open() as f:
        paths = [r["path"] for r in csv.DictReader(f)]
    output = ROOT / "features" / f"{args.dataset}.prospective.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, paths, chunksize=32))
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{args.dataset}: {len(rows)} rows, {len(rows[0]) - 2} prospective features -> {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
