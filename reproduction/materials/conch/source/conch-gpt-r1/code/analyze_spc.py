#!/usr/bin/env python3
"""Compute SPCs, uncertainty, within-class effects, matching, and gate calls."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
METHODS = {
    "conch": "CONCH zero-shot",
    "handcrafted_logistic": "Color/texture logistic",
    "mobilenet_nearest_centroid": "ImageNet MobileNet centroid",
}
RNG_SEED = 20260714
B = 2000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def bacc(y: np.ndarray, pred: np.ndarray) -> float:
    present = sorted(set(y))
    return float(np.mean([np.mean(pred[y == c] == c) for c in present]))


def bootstrap_bacc(y: np.ndarray, pred: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    by_class = [np.flatnonzero(y == c) for c in sorted(set(y))]
    out = np.empty(B)
    for b in range(B):
        recalls = []
        for idx in by_class:
            sample = rng.choice(idx, size=len(idx), replace=True)
            recalls.append(np.mean(pred[sample] == y[sample]))
        out[b] = np.mean(recalls)
    return out


def load(dataset: str, method: str) -> pd.DataFrame:
    score = pd.read_csv(ROOT / "scores" / f"{dataset}.{method}.csv", keep_default_na=False)
    axis = pd.read_csv(ROOT / "features" / f"{dataset}.axis.csv")
    feat = pd.read_csv(ROOT / "features" / f"{dataset}.prospective.csv", usecols=["tile_id", "focus_laplacian_var", "tissue_fraction", "white_fraction"])
    frame = score.merge(axis, on="tile_id", validate="one_to_one").merge(feat, on="tile_id", validate="one_to_one")
    exclusions = pd.read_csv(ROOT / "results" / "dedup_exclusions.csv")
    frame = frame[~frame.tile_id.isin(exclusions[exclusions.dataset == dataset].tile_id)]
    population = json.loads((ROOT / "audit" / "analysis_population.json").read_text())
    eligible = population["datasets"][dataset]["primary_classes"]
    return frame[frame.label.isin(eligible)].copy()


def paired_extreme_bootstrap(frames: dict[str, pd.DataFrame], rng: np.random.Generator) -> dict:
    conch = frames["conch"].set_index("tile_id")
    by_class_level = {}
    for c in sorted(set(conch.label)):
        for level in (1, 5):
            by_class_level[(c, level)] = conch[(conch.label == c) & (conch.axis_level == level)].index.to_numpy()
    drops = {m: np.empty(B) for m in frames}
    indexed = {m: f.set_index("tile_id") for m, f in frames.items()}
    for b in range(B):
        for method, frame in indexed.items():
            per_class = []
            for c in sorted(set(conch.label)):
                rec = []
                for level in (1, 5):
                    ids = by_class_level[(c, level)]
                    sample = rng.choice(ids, size=len(ids), replace=True)
                    g = frame.loc[sample]
                    rec.append(np.mean(g.prediction.to_numpy() == g.label.to_numpy()))
                per_class.append(rec[0] - rec[1])
            drops[method][b] = np.mean(per_class)
    result = {}
    for method, values in drops.items():
        result[method] = {
            "drop_l1_minus_l5": float(np.mean(values)),
            "ci95": np.quantile(values, [.025, .975]).tolist(),
            "one_sided_p_no_degradation": float((np.sum(values <= 0) + 1) / (B + 1)),
        }
    for reference in ("handcrafted_logistic", "mobilenet_nearest_centroid"):
        diff = drops["conch"] - drops[reference]
        result[f"conch_minus_{reference}"] = {
            "difference_in_drop": float(np.mean(diff)),
            "ci95": np.quantile(diff, [.025, .975]).tolist(),
        }
    return result


def matched_controls(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    extreme = frame[frame.axis_level.isin([1, 5])].copy()
    for col in ("color_stain_distance", "focus_laplacian_var", "tissue_fraction"):
        extreme[col + "_bin"] = pd.qcut(extreme[col], 3, labels=False, duplicates="drop")
    keys = ["label", "color_stain_distance_bin", "focus_laplacian_var_bin", "tissue_fraction_bin"]
    pairs = []
    for _, g in extreme.groupby(keys, dropna=False):
        low = g[g.axis_level == 1]
        high = g[g.axis_level == 5]
        n = min(len(low), len(high))
        if n:
            low = low.sample(n=n, random_state=RNG_SEED)
            high = high.sample(n=n, random_state=RNG_SEED)
            pairs.extend((a, b) for a, b in zip((low.prediction == low.label).astype(float), (high.prediction == high.label).astype(float)))
    pairs = np.asarray(pairs, dtype=float)
    if not len(pairs):
        return {"matched_pairs": 0}
    delta = pairs[:, 0] - pairs[:, 1]
    boots = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(B)])
    return {"matched_pairs": int(len(delta)), "accuracy_drop": float(delta.mean()), "ci95": np.quantile(boots, [.025, .975]).tolist()}


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    spc_rows, class_rows = [], []
    summary = {"bootstrap_replicates": B, "uncertainty_unit": "tile; source records expose no patient/slide identifier", "datasets": {}}
    all_curves = {}
    datasets = ("crc_val_7k", "stanford_crc_val_small")
    for dataset in datasets:
        frames = {m: load(dataset, m) for m in METHODS}
        ids = [set(f.tile_id) for f in frames.values()]
        if not all(x == ids[0] for x in ids[1:]):
            raise RuntimeError(f"same-image invariant failed for {dataset}")
        dataset_summary = {"n_common_scored": len(ids[0]), "methods": {}, "primary_extreme_contrast": paired_extreme_bootstrap(frames, rng)}
        for method, frame in frames.items():
            method_summary = {"matched_confound_check": matched_controls(frame, rng)}
            curve = []
            for level, group in frame.groupby("axis_level"):
                y, pred = group.label.to_numpy(), group.prediction.to_numpy()
                boots = bootstrap_bacc(y, pred, rng)
                row = {
                    "dataset": dataset, "method": method, "level": int(level), "n": len(group),
                    "distance_mean": group.morphology_texture_distance.mean(),
                    "distance_median": group.morphology_texture_distance.median(),
                    "reference_percentile_mean": group.reference_percentile.mean(),
                    "balanced_accuracy": bacc(y, pred),
                    "bacc_ci_low": np.quantile(boots, .025), "bacc_ci_high": np.quantile(boots, .975),
                    "accuracy": np.mean(y == pred), "macro_f1": f1_score(y, pred, average="macro"),
                }
                spc_rows.append(row)
                curve.append(row)
                for c in sorted(set(y)):
                    mask = y == c
                    k, n = int(np.sum(pred[mask] == c)), int(mask.sum())
                    lo, hi = wilson(k, n)
                    class_rows.append({"dataset": dataset, "method": method, "level": int(level), "class": c, "n": n, "correct": k, "recall": k / n, "ci_low": lo, "ci_high": hi})
            method_summary["overall_balanced_accuracy"] = bacc(frame.label.to_numpy(), frame.prediction.to_numpy())
            method_summary["curve"] = curve
            dataset_summary["methods"][method] = method_summary
            all_curves[(dataset, method)] = curve
        conch_drop = dataset_summary["primary_extreme_contrast"]["conch"]["drop_l1_minus_l5"]
        ref_drop = dataset_summary["primary_extreme_contrast"]["handcrafted_logistic"]["drop_l1_minus_l5"]
        ref_cis = [dataset_summary["primary_extreme_contrast"][r]["ci95"] for r in ("handcrafted_logistic", "mobilenet_nearest_centroid")]
        dataset_summary["explicit_gate"] = {
            "question": "Does CONCH degrade at least as much as a model-independent reference along the frozen axis?",
            "answer_point_estimate_vs_handcrafted": bool(conch_drop >= ref_drop),
            "behavior": "shared" if any(ci[0] > 0 for ci in ref_cis) else "model-specific-or-reference-inconclusive",
            "shared_behavior_is_not_a_rejection": True,
        }
        summary["datasets"][dataset] = dataset_summary

    pd.DataFrame(spc_rows).to_csv(ROOT / "results" / "spc_table.csv", index=False)
    pd.DataFrame(class_rows).to_csv(ROOT / "results" / "within_class_spc.csv", index=False)
    (ROOT / "results" / "closure_statistics.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, dataset in zip(axes, datasets):
        for method, label in METHODS.items():
            curve = pd.DataFrame(all_curves[(dataset, method)]).sort_values("distance_mean")
            yerr = np.vstack([curve.balanced_accuracy - curve.bacc_ci_low, curve.bacc_ci_high - curve.balanced_accuracy])
            ax.errorbar(curve.distance_mean, curve.balanced_accuracy, yerr=yerr, marker="o", capsize=3, label=label)
        ax.set_title(dataset)
        ax.set_xlabel("NCT morphology-texture novelty distance")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Balanced accuracy (95% stratified tile bootstrap CI)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "plots" / "spectral_performance_curve.png", dpi=180)
    fig.savefig(ROOT / "plots" / "spectral_performance_curve.pdf")
    print(json.dumps({d: summary["datasets"][d]["explicit_gate"] for d in summary["datasets"]}, indent=2))


if __name__ == "__main__":
    main()
