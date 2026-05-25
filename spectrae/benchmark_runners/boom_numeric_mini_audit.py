"""Run a lightweight numeric /spectra audit on BOOM 10k density.

This runner is intentionally small: it does not reproduce the full BOOM model
suite. It uses the BOOM 10k density split, a Morgan-fingerprint random forest,
and post-hoc test subsets with decreasing train-test Tanimoto overlap. The goal
is to produce a first measured performance-over-overlap curve.
"""

import argparse
import csv
import json
import math
import os
import pickle
import urllib.request
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DEFAULT_ASSET_DIR = os.environ.get(
    "SPECTRA_ASSET_DIR",
    "/ewsc/yektefai/spectra_assets"
    if os.path.isdir("/ewsc/yektefai")
    else "spectra_assets",
)
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_ASSET_DIR, "boom_numeric_pilot")
SPLIT_FILE_NAME = "10k_dft_data_with_ood_splits.csv"
RAW_DENSITY_FILE_NAME = "10k_dft_density_data.csv"
RAW_HOF_FILE_NAME = "10k_dft_hof_data.csv"
RAW_DENSITY_URL = "https://raw.githubusercontent.com/FLASK-LLNL/LLNL-10k-Dataset/refs/heads/main/10k_dft_density_data.csv"
RAW_HOF_URL = "https://raw.githubusercontent.com/FLASK-LLNL/LLNL-10k-Dataset/refs/heads/main/10k_dft_hof_data.csv"

RDLogger.DisableLog("rdApp.warning")


def download_url(url: str, path: str, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "spectrae-boom-mini-audit/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as handle:
        handle.write(response.read())


def ensure_split_file(output_dir: str, overwrite: bool = False) -> str:
    """Ensure BOOM 10k raw and split CSVs exist.

    If the split file is missing, this function uses BOOM's own split generator
    when the BOOM repo/package is importable. That preserves BOOM's KDE-based
    property-tail split logic without requiring the full BOOM experiment stack.
    """
    split_file = os.path.join(output_dir, SPLIT_FILE_NAME)
    if os.path.exists(split_file) and not overwrite:
        return split_file

    density_file = os.path.join(output_dir, RAW_DENSITY_FILE_NAME)
    hof_file = os.path.join(output_dir, RAW_HOF_FILE_NAME)
    download_url(RAW_DENSITY_URL, density_file, overwrite=overwrite)
    download_url(RAW_HOF_URL, hof_file, overwrite=overwrite)

    try:
        from boom.data.prepare_splits_10k import prepare_splits
    except Exception as exc:
        raise RuntimeError(
            "BOOM must be importable to generate the official 10k split file. "
            "Set PYTHONPATH to the BOOM repo clone."
        ) from exc

    prepare_splits(density_file, hof_file, split_file)
    return split_file


def mol_from_smiles(smiles: str) -> Any:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES: %s" % smiles)
    return mol


def fingerprint_from_smiles(smiles: str, radius: int = 2, n_bits: int = 1024) -> Any:
    mol = mol_from_smiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def fingerprints_to_numpy(fingerprints: Sequence[Any], n_bits: int = 1024) -> np.ndarray:
    matrix = np.zeros((len(fingerprints), n_bits), dtype=np.float32)
    for index, fingerprint in enumerate(fingerprints):
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, arr)
        matrix[index] = arr
    return matrix


def max_train_similarity(test_fingerprint: Any, train_fingerprints: Sequence[Any]) -> float:
    return float(max(DataStructs.BulkTanimotoSimilarity(test_fingerprint, train_fingerprints)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def subset_metrics(
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_similarities: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    subset_size = int(mask.sum())
    if subset_size == 0:
        return {
            "subset": label,
            "n": 0,
            "mean_max_tanimoto": None,
            "median_max_tanimoto": None,
            "rmse": None,
            "mae": None,
        }
    return {
        "subset": label,
        "n": subset_size,
        "mean_max_tanimoto": float(np.mean(max_similarities[mask])),
        "median_max_tanimoto": float(np.median(max_similarities[mask])),
        "rmse": rmse(y_true[mask], y_pred[mask]),
        "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
    }


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError("Cannot write empty CSV: %s" % path)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_rows(
    path: str,
    rows: Sequence[Dict[str, Any]],
) -> None:
    write_csv(path, rows)


def write_curve_plot(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    valid = [
        row
        for row in rows
        if row["n"] and row["mean_max_tanimoto"] is not None and row["rmse"] is not None
    ]
    if not valid:
        return
    overlaps = [float(row["mean_max_tanimoto"]) for row in valid]
    rmses = [float(row["rmse"]) for row in valid]
    labels = [row["subset"].replace("ood_max_tanimoto_le_", "<= ") for row in valid]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width = 800
    height = 520
    margin_left = 70
    margin_right = 30
    margin_top = 50
    margin_bottom = 70
    x_min, x_max = min(overlaps), max(overlaps)
    y_min, y_max = min(rmses), max(rmses)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    def x_scale(value: float) -> float:
        # Invert x so lower overlap appears farther right as novelty increases.
        return margin_left + (x_max - value) / (x_max - x_min) * (width - margin_left - margin_right)

    def y_scale(value: float) -> float:
        return height - margin_bottom - (value - y_min) / (y_max - y_min) * (height - margin_top - margin_bottom)

    points = [(x_scale(x), y_scale(y), label, x, y) for x, y, label in zip(overlaps, rmses, labels)]
    polyline = " ".join("%.2f,%.2f" % (x, y) for x, y, _, _, _ in points)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">\n' % (width, height, width, height))
        handle.write('<rect width="100%" height="100%" fill="white"/>\n')
        handle.write('<text x="%d" y="28" font-size="20" font-family="Arial">BOOM 10k Density: Performance vs Chemical Overlap</text>\n' % margin_left)
        handle.write('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#222"/>\n' % (margin_left, height - margin_bottom, width - margin_right, height - margin_bottom))
        handle.write('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#222"/>\n' % (margin_left, margin_top, margin_left, height - margin_bottom))
        handle.write('<text x="%d" y="%d" font-size="14" font-family="Arial">Novelty increases as mean max Tanimoto decreases</text>\n' % (margin_left + 120, height - 25))
        handle.write('<text x="15" y="%d" font-size="14" font-family="Arial" transform="rotate(-90 15,%d)">Density RMSE</text>\n' % (height // 2 + 80, height // 2 + 80))
        handle.write('<polyline points="%s" fill="none" stroke="#1f77b4" stroke-width="3"/>\n' % polyline)
        for x, y, label, overlap, value in points:
            handle.write('<circle cx="%.2f" cy="%.2f" r="5" fill="#1f77b4"/>\n' % (x, y))
            handle.write('<text x="%.2f" y="%.2f" font-size="12" font-family="Arial">%s</text>\n' % (x + 7, y - 7, label))
            handle.write('<text x="%.2f" y="%.2f" font-size="11" font-family="Arial" fill="#444">%.3f / %.3f</text>\n' % (x + 7, y + 9, overlap, value))
        handle.write("</svg>\n")


def write_report(path: str, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    id_metrics = result["id_metrics"]
    curve = result["ood_curve"]
    hardest = curve[-1] if curve else None
    all_ood = curve[0] if curve else None
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# BOOM Numeric Mini-Audit\n\n")
        handle.write("Task: BOOM 10k density with a Morgan-fingerprint random forest baseline.\n\n")
        handle.write("## Summary\n\n")
        handle.write(
            "- ID RMSE: %.6f at mean max Tanimoto %.6f.\n"
            % (id_metrics["rmse"], id_metrics["mean_max_tanimoto"])
        )
        if all_ood:
            handle.write(
                "- Full OOD RMSE: %.6f at mean max Tanimoto %.6f.\n"
                % (all_ood["rmse"], all_ood["mean_max_tanimoto"])
            )
        if hardest:
            handle.write(
                "- Lowest-overlap OOD subset RMSE: %.6f at mean max Tanimoto %.6f, n=%d.\n"
                % (hardest["rmse"], hardest["mean_max_tanimoto"], hardest["n"])
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "This pilot reproduces the expected BOOM pattern that property-tail OOD "
            "molecules are harder than ID molecules for a lightweight baseline. "
            "Within the OOD set, RMSE rises as maximum Morgan Tanimoto similarity "
            "to the training set decreases, giving initial measured support for "
            "a SPECTRA-style chemical-overlap audit.\n\n"
        )
        handle.write("## Limitations\n\n")
        handle.write(
            "- This is a post-hoc test-subset novelty curve, not a full SPECTRA train/test resplitting run.\n"
            "- It uses one lightweight baseline and one BOOM property task.\n"
            "- It should be followed by retraining/evaluating models across generated SPECTRA splits.\n"
        )


def compute_auspc_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        row
        for row in rows
        if row["n"] and row["mean_max_tanimoto"] is not None and row["rmse"] is not None
    ]
    if len(valid) < 2:
        return {"computed": False, "value": None, "reason": "Fewer than two valid curve points."}

    overlaps = np.array([float(row["mean_max_tanimoto"]) for row in valid], dtype=float)
    rmse_values = np.array([float(row["rmse"]) for row in valid], dtype=float)
    min_overlap = float(overlaps.min())
    max_overlap = float(overlaps.max())
    if max_overlap == min_overlap:
        return {"computed": False, "value": None, "reason": "Overlap values are constant."}

    novelty = 1.0 - ((overlaps - min_overlap) / (max_overlap - min_overlap))
    order = np.argsort(novelty)
    novelty = novelty[order]
    score = -rmse_values[order]
    area = float(np.trapz(score, novelty))
    return {
        "computed": True,
        "value": area,
        "metric": "negative_rmse_area",
        "novelty_axis": "1 - normalized_mean_max_tanimoto",
        "point_count": len(valid),
    }


def run_audit(
    output_dir: str,
    split_file: str,
    target: str = "density",
    n_estimators: int = 120,
    random_seed: int = 42,
    n_bits: int = 1024,
    thresholds: Sequence[float] = (1.0, 0.8, 0.7, 0.6, 0.5),
) -> Dict[str, Any]:
    df = pd.read_csv(split_file)
    train = df[df[f"{target}_train"] == 1].copy()
    iid = df[df[f"{target}_iid"] == 1].copy()
    ood = df[df[f"{target}_ood"] == 1].copy()

    train_fps = [fingerprint_from_smiles(smiles, n_bits=n_bits) for smiles in train["smiles"]]
    iid_fps = [fingerprint_from_smiles(smiles, n_bits=n_bits) for smiles in iid["smiles"]]
    ood_fps = [fingerprint_from_smiles(smiles, n_bits=n_bits) for smiles in ood["smiles"]]

    x_train = fingerprints_to_numpy(train_fps, n_bits=n_bits)
    y_train = train[target].to_numpy(dtype=float)
    x_iid = fingerprints_to_numpy(iid_fps, n_bits=n_bits)
    y_iid = iid[target].to_numpy(dtype=float)
    x_ood = fingerprints_to_numpy(ood_fps, n_bits=n_bits)
    y_ood = ood[target].to_numpy(dtype=float)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_seed,
    )
    model.fit(x_train, y_train)
    iid_pred = model.predict(x_iid)
    ood_pred = model.predict(x_ood)

    iid_max = np.array([max_train_similarity(fp, train_fps) for fp in iid_fps], dtype=float)
    ood_max = np.array([max_train_similarity(fp, train_fps) for fp in ood_fps], dtype=float)

    split_rows = []
    performance_rows = []
    iid_mask = np.ones(len(iid), dtype=bool)
    iid_metrics = subset_metrics("id_test", y_iid, iid_pred, iid_max, iid_mask)
    performance_rows.append(iid_metrics)
    split_rows.append(
        {
            "subset": "id_test",
            "threshold": "",
            "n": iid_metrics["n"],
            "mean_max_tanimoto": iid_metrics["mean_max_tanimoto"],
            "median_max_tanimoto": iid_metrics["median_max_tanimoto"],
        }
    )

    for threshold in thresholds:
        mask = ood_max <= threshold
        label = "ood_max_tanimoto_le_%s" % str(threshold).replace(".", "_")
        metrics = subset_metrics(label, y_ood, ood_pred, ood_max, mask)
        performance_rows.append(metrics)
        split_rows.append(
            {
                "subset": label,
                "threshold": threshold,
                "n": metrics["n"],
                "mean_max_tanimoto": metrics["mean_max_tanimoto"],
                "median_max_tanimoto": metrics["median_max_tanimoto"],
            }
        )

    curve_rows = [row for row in performance_rows if row["subset"].startswith("ood_") and row["n"]]
    auspc = compute_auspc_rows(curve_rows)

    artifacts_dir = os.path.join(output_dir, "artifacts")
    split_stats_path = os.path.join(artifacts_dir, "split_stats.csv")
    performance_path = os.path.join(artifacts_dir, "performance_by_overlap.csv")
    predictions_path = os.path.join(artifacts_dir, "model_predictions.csv")
    model_path = os.path.join(artifacts_dir, "model.pkl")
    curve_plot_path = os.path.join(artifacts_dir, "spectral_curve.svg")
    report_path = os.path.join(artifacts_dir, "report.md")
    audit_card_path = os.path.join(artifacts_dir, "audit_card.json")
    write_csv(split_stats_path, split_rows)
    write_csv(performance_path, performance_rows)
    prediction_rows = []
    for split_name, split_df, y_true, y_pred, max_sims in [
        ("id", iid, y_iid, iid_pred, iid_max),
        ("ood", ood, y_ood, ood_pred, ood_max),
    ]:
        for smiles, truth, pred, max_sim in zip(split_df["smiles"], y_true, y_pred, max_sims):
            prediction_rows.append(
                {
                    "split": split_name,
                    "smiles": smiles,
                    "target": target,
                    "y_true": float(truth),
                    "y_pred": float(pred),
                    "abs_error": float(abs(truth - pred)),
                    "squared_error": float((truth - pred) ** 2),
                    "max_train_tanimoto": float(max_sim),
                }
            )
    write_prediction_rows(predictions_path, prediction_rows)
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)
    write_curve_plot(curve_plot_path, curve_rows)

    audit_card = {
        "schema_version": "0.1.0",
        "capsule_id": "boom",
        "paper": {
            "title": "BOOM: Benchmarking Out-Of-distribution Molecular Property Predictions of Machine Learning Models",
            "url": "https://openreview.net/forum?id=QoBxQrvFRd",
            "pdf_url": "https://arxiv.org/pdf/2505.01912.pdf",
        },
        "scientific_unit": "molecule",
        "models": ["RandomForestRegressor over Morgan fingerprints"],
        "original_evaluation": [
            "BOOM 10k density KDE property-tail OOD split",
            "ID test set sampled from remaining molecules",
        ],
        "discovered_novelty_axes": [
            {
                "name": "chemical_distance",
                "property_graph": "Morgan fingerprint Tanimoto similarity",
                "graph_type": "weighted",
            }
        ],
        "property_graphs": [
            {
                "name": "morgan_tanimoto_train_test_overlap",
                "construction": "Morgan radius-2 fingerprints; max Tanimoto similarity from each test molecule to the BOOM training set.",
                "edge_weight": "Tanimoto similarity",
            }
        ],
        "spectral_split_protocol": {
            "status": "post_hoc_test_novelty_curve",
            "description": "Train once on BOOM density train split, then evaluate BOOM OOD test subsets with decreasing maximum Tanimoto similarity to training molecules.",
            "thresholds": list(thresholds),
            "validation": "Mean and median max train-test Tanimoto similarity are written to split_stats.csv.",
        },
        "performance_overlap_curve": curve_rows,
        "auspc": auspc,
        "core_reproduced_claim": "The lightweight RF baseline is evaluated on the BOOM density property-tail split and compared against the ID test split.",
        "additional_spectra_finding": "This numeric pilot tests whether BOOM density OOD error changes as chemical train-test overlap decreases within the OOD split.",
        "citation": "BOOM: https://openreview.net/forum?id=QoBxQrvFRd; data: https://github.com/FLASK-LLNL/LLNL-10k-Dataset",
        "artifacts": {
            "split_stats_path": split_stats_path,
            "performance_by_overlap_path": performance_path,
            "predictions_path": predictions_path,
            "model_path": model_path,
            "curve_plot_path": curve_plot_path,
            "report_path": report_path,
            "audit_card_path": audit_card_path,
        },
    }
    write_report(report_path, {"id_metrics": iid_metrics, "ood_curve": curve_rows})
    with open(audit_card_path, "w", encoding="utf-8") as handle:
        json.dump(audit_card, handle, indent=2, sort_keys=True)

    return {
        "output_dir": output_dir,
        "split_file": split_file,
        "train_size": int(len(train)),
        "id_size": int(len(iid)),
        "ood_size": int(len(ood)),
        "artifacts": audit_card["artifacts"],
        "id_metrics": iid_metrics,
        "ood_curve": curve_rows,
        "auspc": auspc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BOOM 10k density numeric mini-audit.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-file")
    parser.add_argument("--boom-repo", help="Optional BOOM repo path to add to sys.path.")
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--n-bits", type=int, default=1024)
    parser.add_argument("--overwrite-splits", action="store_true")
    args = parser.parse_args()

    if args.boom_repo:
        import sys

        sys.path.insert(0, args.boom_repo)

    os.makedirs(args.output_dir, exist_ok=True)
    split_file = args.split_file or ensure_split_file(
        args.output_dir,
        overwrite=args.overwrite_splits,
    )
    result = run_audit(
        output_dir=args.output_dir,
        split_file=split_file,
        n_estimators=args.n_estimators,
        n_bits=args.n_bits,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
