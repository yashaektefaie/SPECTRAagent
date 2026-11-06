#!/usr/bin/env python3
"""Fit a label/model-blind NCT morphology-texture novelty axis and levels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(__file__).resolve().parents[1]


def deterministic_fit_mask(ids: pd.Series) -> np.ndarray:
    return np.array([int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) % 5 != 0 for x in ids], dtype=bool)


def robust_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(x, axis=0)
    q25, q75 = np.nanquantile(x, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale[scale < 1e-8] = 1.0
    return median, scale


def transform(x: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - median) / scale, nan=0.0, posinf=20.0, neginf=-20.0).clip(-20, 20)


def nearest_distance(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), 4096):
        block = x[start:start + 4096]
        d2 = ((block[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
        out[start:start + len(block)] = np.sqrt(d2.min(axis=1))
    return out


def ecdf_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def assign_levels(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    thresholds = np.quantile(values, [0.2, 0.4, 0.6, 0.8]).tolist()
    return np.digitize(values, thresholds, right=True) + 1, thresholds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="nct_crc_100k")
    p.add_argument("--targets", nargs="+", default=["crc_val_7k", "stanford_crc_val_small", "kather2016_5k"])
    args = p.parse_args()

    ref = pd.read_csv(ROOT / "features" / f"{args.reference}.prospective.csv")
    morph_cols = [c for c in ref if c.startswith(("lbp_", "orient_", "fft_band_", "contrast_s"))]
    morph_cols += ["gray_std", "gray_entropy"]
    color_cols = [c for c in ref if c.startswith(("r_hist_", "g_hist_", "b_hist_"))]
    color_cols += ["r_mean", "g_mean", "b_mean", "sat_mean", "sat_std", "od_mean", "od_std"]
    quality_cols = ["focus_laplacian_var", "tenengrad_mean", "edge_fraction", "tissue_fraction", "white_fraction"]

    fit_mask = deterministic_fit_mask(ref.tile_id)
    model = {"reference": args.reference, "fit_rule": "sha256(tile_id) mod 5 != 0", "random_state": 20260714, "families": {}}
    ref_distances: dict[str, np.ndarray] = {}
    fitted = {}
    for family, cols, k in (("morphology_texture", morph_cols, 64), ("color_stain", color_cols, 32), ("quality", quality_cols, 16)):
        x = ref[cols].to_numpy(float)
        median, scale = robust_fit(x[fit_mask])
        z = transform(x, median, scale)
        km = MiniBatchKMeans(n_clusters=k, random_state=20260714, batch_size=4096, n_init=3, max_iter=200)
        km.fit(z[fit_mask])
        distance = nearest_distance(z, km.cluster_centers_)
        ref_distances[family] = distance
        fitted[family] = (cols, median, scale, km.cluster_centers_)
        model["families"][family] = {
            "columns": cols,
            "median": median.tolist(),
            "iqr_scale": scale.tolist(),
            "centers": km.cluster_centers_.tolist(),
        }

    holdout = ~fit_mask
    diagnostics = {
        "axis": "NCT morphology-texture novelty distance",
        "direction": "larger is less similar",
        "reference_fit_n": int(fit_mask.sum()),
        "reference_holdout_n": int(holdout.sum()),
        "reference_holdout_morph_distance_quantiles": np.quantile(ref_distances["morphology_texture"][holdout], [0, .2, .4, .5, .6, .8, 1]).tolist(),
        "targets": {},
    }
    order_rows = []
    for dataset in args.targets:
        frame = pd.read_csv(ROOT / "features" / f"{dataset}.prospective.csv")
        output = pd.DataFrame({"tile_id": frame.tile_id})
        for family, (cols, median, scale, centers) in fitted.items():
            output[family + "_distance"] = nearest_distance(transform(frame[cols].to_numpy(float), median, scale), centers)
        output["reference_percentile"] = ecdf_percentile(output.morphology_texture_distance.to_numpy(), ref_distances["morphology_texture"][holdout])
        levels, thresholds = assign_levels(output.morphology_texture_distance.to_numpy())
        output["axis_level"] = levels
        output.to_csv(ROOT / "features" / f"{dataset}.axis.csv", index=False)
        merged = output.join(frame.set_index("tile_id")[["focus_laplacian_var", "tissue_fraction", "white_fraction"]], on="tile_id")
        corr = merged[["morphology_texture_distance", "color_stain_distance", "quality_distance", "focus_laplacian_var", "tissue_fraction", "white_fraction"]].corr(method="spearman")
        level_summary = []
        for level, group in merged.groupby("axis_level"):
            row = {
                "dataset": dataset,
                "level": int(level),
                "n": int(len(group)),
                "morph_distance_mean": float(group.morphology_texture_distance.mean()),
                "morph_distance_median": float(group.morphology_texture_distance.median()),
                "reference_percentile_mean": float(group.reference_percentile.mean()),
                "color_distance_mean": float(group.color_stain_distance.mean()),
                "quality_distance_mean": float(group.quality_distance.mean()),
                "focus_mean": float(group.focus_laplacian_var.mean()),
                "tissue_fraction_mean": float(group.tissue_fraction.mean()),
                "white_fraction_mean": float(group.white_fraction.mean()),
            }
            order_rows.append(row)
            level_summary.append(row)
        diagnostics["targets"][dataset] = {
            "n": int(len(frame)),
            "level_thresholds": thresholds,
            "distance_min_max": [float(output.morphology_texture_distance.min()), float(output.morphology_texture_distance.max())],
            "spearman_correlations": corr.to_dict(),
            "levels": level_summary,
        }

    (ROOT / "features" / "axis_model.json").write_text(json.dumps(model, indent=2) + "\n")
    (ROOT / "features" / "axis_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    pd.DataFrame(order_rows).to_csv(ROOT / "results" / "measured_axis_order_blind.csv", index=False)
    print(json.dumps({"model": "features/axis_model.json", "diagnostics": "features/axis_diagnostics.json", "targets": diagnostics["targets"]}, indent=2))


if __name__ == "__main__":
    main()
