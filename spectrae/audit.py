"""Deterministic SPECTRA audit engine.

This module contains package-backed audit logic that can run without an agent.
Agents should call this code, then explain and cite the generated artifacts.
"""

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = (1.0, 0.8, 0.7, 0.6, 0.5)


@dataclass
class SpectralAxisAuditConfig:
    eval_path: str
    output_dir: str
    target_col: str = "y_true"
    pred_col: str = "y_pred"
    axis_col: str = "spectral_axis"
    axis_type: str = "similarity"
    axis_name: str = "spectral_axis"
    domain: str = "generic"
    scientific_unit: str = "sample"
    train_path: Optional[str] = None
    train_target_col: Optional[str] = None
    unit_col: Optional[str] = None
    thresholds: Optional[Sequence[float]] = None
    quantile_bins: int = 5


@dataclass
class PairwiseSimilarityAuditConfig:
    eval_path: str
    similarity_path: str
    output_dir: str
    target_col: str = "y_true"
    pred_col: str = "y_pred"
    eval_id_col: str = "sample_id"
    similarity_eval_id_col: str = "sample_id"
    similarity_train_id_col: str = "train_id"
    similarity_col: str = "similarity"
    domain: str = "generic"
    scientific_unit: str = "sample"
    train_path: Optional[str] = None
    train_id_col: str = "train_id"
    train_target_col: Optional[str] = None
    axis_name: str = "agent_defined_pairwise_similarity"
    thresholds: Optional[Sequence[float]] = None
    quantile_bins: int = 5


@dataclass
class MoleculeAuditConfig:
    train_path: str
    eval_path: str
    output_dir: str
    smiles_col: str = "smiles"
    train_target_col: str = "y"
    eval_target_col: str = "y_true"
    pred_col: str = "y_pred"
    sample_id_col: Optional[str] = None
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS
    fingerprint_radius: int = 2
    fingerprint_bits: int = 1024
    prefer_rdkit: bool = True


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any) -> Optional[float]:
    return float(value) if _is_finite_number(value) else None


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) < 2:
        return None
    total = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if total == 0:
        return None
    residual = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - residual / total)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, Any]:
    """Compute deterministic regression metrics from raw predictions."""
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if len(truth) == 0:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "bias_mean_pred_minus_true": None,
            "median_abs_error": None,
            "p90_abs_error": None,
            "p95_abs_error": None,
            "max_abs_error": None,
        }

    abs_error = np.abs(truth - pred)
    return {
        "n": int(len(truth)),
        "mae": float(np.mean(abs_error)),
        "rmse": _rmse(truth, pred),
        "r2": _r2(truth, pred),
        "bias_mean_pred_minus_true": float(np.mean(pred - truth)),
        "median_abs_error": float(np.median(abs_error)),
        "p90_abs_error": float(np.quantile(abs_error, 0.90)),
        "p95_abs_error": float(np.quantile(abs_error, 0.95)),
        "max_abs_error": float(np.max(abs_error)),
    }


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError("Cannot write empty CSV: %s" % path)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _parse_thresholds(thresholds: Sequence[float]) -> List[float]:
    parsed = []
    for value in thresholds:
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Similarity thresholds must be between 0 and 1")
        parsed.append(threshold)
    return sorted(set(parsed), reverse=True)


def parse_threshold_string(value: str) -> List[float]:
    """Parse comma-separated similarity thresholds."""
    return _parse_thresholds([float(item.strip()) for item in value.split(",") if item.strip()])


def parse_optional_threshold_string(value: Optional[str]) -> Optional[List[float]]:
    """Parse optional comma-separated thresholds."""
    if value is None or not value.strip():
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _normalize_axis_type(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "overlap": "similarity",
        "sim": "similarity",
        "distance_from_train": "distance",
        "novelty": "distance",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"similarity", "distance"}:
        raise ValueError("axis_type must be similarity or distance")
    return normalized


def _axis_thresholds(values: np.ndarray, axis_type: str, thresholds: Optional[Sequence[float]], quantile_bins: int) -> List[float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("Axis column has no finite values")
    if thresholds is not None:
        parsed = sorted(set(float(value) for value in thresholds))
        return list(reversed(parsed)) if axis_type == "similarity" else parsed
    if quantile_bins < 2:
        raise ValueError("quantile_bins must be at least 2")

    if axis_type == "similarity":
        quantiles = np.linspace(1.0, 0.0, quantile_bins)
    else:
        quantiles = np.linspace(0.0, 1.0, quantile_bins)
    return [float(np.quantile(finite, quantile)) for quantile in quantiles]


def _axis_novelty(values: np.ndarray, axis_type: str) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("Axis column has no finite values")
    if axis_type == "distance":
        return values.astype(float)
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if 0.0 <= minimum and maximum <= 1.0:
        return 1.0 - values.astype(float)
    # Similarity-like axes outside [0, 1] still have inverse orientation.
    return -values.astype(float)


def _char_ngrams(value: str, lengths: Sequence[int] = (2, 3, 4)) -> set:
    normalized = str(value)
    features = set()
    for length in lengths:
        if len(normalized) < length:
            features.add(normalized)
        else:
            features.update(normalized[index : index + length] for index in range(len(normalized) - length + 1))
    return features


def _jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return float(len(left & right) / union)


def _string_max_similarities(train_smiles: Sequence[str], eval_smiles: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    train_features = [_char_ngrams(smiles) for smiles in train_smiles]
    max_values = []
    nearest_indices = []
    for smiles in eval_smiles:
        features = _char_ngrams(smiles)
        similarities = [_jaccard(features, train_feature) for train_feature in train_features]
        best_index = int(np.argmax(similarities)) if similarities else -1
        nearest_indices.append(best_index)
        max_values.append(float(similarities[best_index]) if best_index >= 0 else 0.0)
    return np.asarray(max_values, dtype=float), {
        "method": "smiles_char_ngram_jaccard",
        "nearest_train_indices": nearest_indices,
        "fallback": True,
        "warnings": [
            "RDKit was unavailable or disabled; similarity uses transparent SMILES character n-gram Jaccard, not chemical fingerprints."
        ],
    }


def _rdkit_max_similarities(
    train_smiles: Sequence[str],
    eval_smiles: Sequence[str],
    radius: int,
    n_bits: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from rdkit import Chem, DataStructs, RDLogger  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    RDLogger.DisableLog("rdApp.warning")

    train_mols = [Chem.MolFromSmiles(smiles) for smiles in train_smiles]
    eval_mols = [Chem.MolFromSmiles(smiles) for smiles in eval_smiles]
    invalid_train = [index for index, mol in enumerate(train_mols) if mol is None]
    invalid_eval = [index for index, mol in enumerate(eval_mols) if mol is None]
    if invalid_train or invalid_eval:
        raise ValueError(
            "RDKit could not parse %d training and %d eval SMILES"
            % (len(invalid_train), len(invalid_eval))
        )

    train_fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        for mol in train_mols
    ]
    eval_fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        for mol in eval_mols
    ]

    max_values = []
    nearest_indices = []
    for fingerprint in eval_fps:
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps)
        best_index = int(np.argmax(similarities)) if similarities else -1
        nearest_indices.append(best_index)
        max_values.append(float(similarities[best_index]) if best_index >= 0 else 0.0)

    train_canonical = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in train_mols]
    eval_canonical = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in eval_mols]
    return np.asarray(max_values, dtype=float), {
        "method": "morgan_tanimoto",
        "fingerprint_radius": radius,
        "fingerprint_bits": n_bits,
        "nearest_train_indices": nearest_indices,
        "fallback": False,
        "train_canonical_smiles": train_canonical,
        "eval_canonical_smiles": eval_canonical,
        "warnings": [],
    }


def compute_molecule_train_similarity(
    train_smiles: Sequence[str],
    eval_smiles: Sequence[str],
    radius: int = 2,
    n_bits: int = 1024,
    prefer_rdkit: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute max train similarity for each eval molecule."""
    if prefer_rdkit:
        try:
            return _rdkit_max_similarities(train_smiles, eval_smiles, radius, n_bits)
        except ImportError:
            return _string_max_similarities(train_smiles, eval_smiles)
    return _string_max_similarities(train_smiles, eval_smiles)


def _subset_metrics(
    subset: str,
    threshold: Optional[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_similarity: np.ndarray,
    mask: np.ndarray,
    train_size: int,
) -> Dict[str, Any]:
    metrics = regression_metrics(y_true[mask], y_pred[mask])
    n = int(mask.sum())
    mean_similarity = float(np.mean(max_similarity[mask])) if n else None
    median_similarity = float(np.median(max_similarity[mask])) if n else None
    row = {
        "subset": subset,
        "axis": "max_train_similarity",
        "axis_type": "similarity",
        "threshold": threshold,
        "threshold_rule": "similarity <= threshold",
        "spectral_parameter": None if threshold is None else float(1.0 - threshold),
        "train_size": int(train_size),
        "test_size": n,
        "mean_axis_value": mean_similarity,
        "median_axis_value": median_similarity,
        "mean_novelty": None if mean_similarity is None else float(1.0 - mean_similarity),
        "mean_max_train_similarity": mean_similarity,
        "median_max_train_similarity": median_similarity,
        "cross_split_overlap": mean_similarity,
        "mean_train_distance": None if mean_similarity is None else float(1.0 - mean_similarity),
    }
    row.update(metrics)
    return row


def validate_overlap_curve(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that cross-split overlap decreases as spectral parameter rises."""
    usable = [
        row
        for row in rows
        if _safe_float(row.get("spectral_parameter")) is not None
        and _safe_float(row.get("cross_split_overlap")) is not None
        and int(row.get("test_size") or 0) > 0
    ]
    usable.sort(key=lambda row: float(row["spectral_parameter"]))
    increases = []
    decreases = 0
    comparisons = 0
    for previous, current in zip(usable, usable[1:]):
        previous_overlap = float(previous["cross_split_overlap"])
        current_overlap = float(current["cross_split_overlap"])
        delta = current_overlap - previous_overlap
        comparisons += 1
        if delta <= 1e-12:
            decreases += 1
        else:
            increases.append(
                {
                    "from_subset": previous["subset"],
                    "to_subset": current["subset"],
                    "overlap_increase": delta,
                }
            )
    monotonic = len(increases) == 0
    return {
        "valid": bool(usable) and monotonic,
        "status": "valid" if usable and monotonic else "overlap_contract_failed",
        "monotonic_nonincreasing": monotonic,
        "decreasing_fraction": decreases / comparisons if comparisons else None,
        "row_count": len(usable),
        "overlap_increases": increases,
    }


def validate_novelty_curve(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that measured novelty increases across spectral subsets."""
    usable = [
        row
        for row in rows
        if _safe_float(row.get("spectral_parameter")) is not None
        and _safe_float(row.get("mean_novelty")) is not None
        and int(row.get("test_size") or 0) > 0
    ]
    usable.sort(key=lambda row: float(row["spectral_parameter"]))
    decreases = []
    increases = 0
    comparisons = 0
    for previous, current in zip(usable, usable[1:]):
        delta = float(current["mean_novelty"]) - float(previous["mean_novelty"])
        comparisons += 1
        if delta >= -1e-12:
            increases += 1
        else:
            decreases.append(
                {
                    "from_subset": previous["subset"],
                    "to_subset": current["subset"],
                    "novelty_decrease": delta,
                }
            )
    monotonic = len(decreases) == 0
    return {
        "valid": bool(usable) and monotonic,
        "status": "valid" if usable and monotonic else "novelty_contract_failed",
        "monotonic_nondecreasing_novelty": monotonic,
        "increasing_fraction": increases / comparisons if comparisons else None,
        "row_count": len(usable),
        "novelty_decreases": decreases,
    }


def compute_auspc_from_rows(
    rows: Sequence[Dict[str, Any]],
    performance_key: str = "rmse",
    novelty_key: str = "mean_novelty",
) -> Dict[str, Any]:
    valid = [
        row
        for row in rows
        if _safe_float(row.get(novelty_key)) is not None
        and _safe_float(row.get(performance_key)) is not None
        and int(row.get("test_size") or row.get("n") or 0) > 0
    ]
    if len(valid) < 2:
        return {
            "computed": False,
            "value": None,
            "reason": "Fewer than two valid curve points.",
        }
    novelty_values = np.asarray([float(row[novelty_key]) for row in valid], dtype=float)
    performance = np.asarray([float(row[performance_key]) for row in valid], dtype=float)
    min_novelty = float(np.min(novelty_values))
    max_novelty = float(np.max(novelty_values))
    if max_novelty == min_novelty:
        return {
            "computed": False,
            "value": None,
            "reason": "Novelty values are constant.",
        }
    novelty = (novelty_values - min_novelty) / (max_novelty - min_novelty)
    order = np.argsort(novelty)
    novelty = novelty[order]
    score = -performance[order]
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = getattr(np, "trapz")
    return {
        "computed": True,
        "value": float(trapezoid(score, novelty)),
        "metric": "negative_%s_area" % performance_key,
        "novelty_axis": "normalized_%s" % novelty_key,
        "point_count": int(len(valid)),
    }


def _write_curve_svg(path: str, rows: Sequence[Dict[str, Any]], metric: str = "rmse") -> None:
    valid = [
        row
        for row in rows
        if _safe_float(row.get("mean_novelty")) is not None
        and _safe_float(row.get(metric)) is not None
        and int(row.get("test_size") or row.get("n") or 0) > 0
    ]
    if not valid:
        return

    novelty_values = [float(row["mean_novelty"]) for row in valid]
    values = [float(row[metric]) for row in valid]
    labels = [str(row["subset"]) for row in valid]

    width = 860
    height = 540
    margin_left = 78
    margin_right = 36
    margin_top = 54
    margin_bottom = 76
    x_min = min(novelty_values)
    x_max = max(novelty_values)
    y_min = min(values)
    y_max = max(values)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    def x_scale(novelty: float) -> float:
        return margin_left + (novelty - x_min) / (x_max - x_min) * (width - margin_left - margin_right)

    def y_scale(value: float) -> float:
        return height - margin_bottom - (value - y_min) / (y_max - y_min) * (height - margin_top - margin_bottom)

    points = [
        (x_scale(novelty), y_scale(value), label, novelty, value)
        for novelty, value, label in zip(novelty_values, values, labels)
    ]
    polyline = " ".join("%.2f,%.2f" % (x, y) for x, y, _, _, _ in points)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">\n' % (width, height, width, height))
        handle.write('<rect width="100%" height="100%" fill="white"/>\n')
        handle.write('<text x="%d" y="30" font-size="20" font-family="Arial">SPECTRA Audit: Performance vs Train Distance</text>\n' % margin_left)
        handle.write('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#222"/>\n' % (margin_left, height - margin_bottom, width - margin_right, height - margin_bottom))
        handle.write('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#222"/>\n' % (margin_left, margin_top, margin_left, height - margin_bottom))
        handle.write('<text x="%d" y="%d" font-size="14" font-family="Arial">Measured novelty</text>\n' % (margin_left + 250, height - 28))
        handle.write('<text x="16" y="%d" font-size="14" font-family="Arial" transform="rotate(-90 16,%d)">%s</text>\n' % (height // 2 + 80, height // 2 + 80, xml_escape(metric.upper())))
        handle.write('<polyline points="%s" fill="none" stroke="#2563eb" stroke-width="3"/>\n' % polyline)
        for x, y, label, novelty, value in points:
            handle.write('<circle cx="%.2f" cy="%.2f" r="5" fill="#2563eb"/>\n' % (x, y))
            handle.write('<text x="%.2f" y="%.2f" font-size="11" font-family="Arial">%s</text>\n' % (x + 8, y - 8, xml_escape(label)))
            handle.write('<text x="%.2f" y="%.2f" font-size="10" font-family="Arial" fill="#444">novelty %.3f, %s %.3f</text>\n' % (x + 8, y + 8, novelty, xml_escape(metric), value))
        handle.write("</svg>\n")


def _write_report(
    path: str,
    audit_card: Dict[str, Any],
    aggregate_metrics: Dict[str, Any],
    performance_rows: Sequence[Dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hardest = performance_rows[-1] if performance_rows else None
    easiest = performance_rows[0] if performance_rows else None
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# SPECTRA Audit Report\n\n")
        handle.write("Domain: `%s`\n\n" % audit_card["domain"])
        handle.write("Scientific unit: `%s`\n\n" % audit_card["scientific_unit"])
        handle.write("## Aggregate Performance\n\n")
        handle.write("- n: `%s`\n" % aggregate_metrics["n"])
        handle.write("- MAE: `%.6f`\n" % aggregate_metrics["mae"])
        handle.write("- RMSE: `%.6f`\n" % aggregate_metrics["rmse"])
        if aggregate_metrics["r2"] is not None:
            handle.write("- R2: `%.6f`\n" % aggregate_metrics["r2"])
        handle.write("- Bias: `%.6f`\n\n" % aggregate_metrics["bias_mean_pred_minus_true"])

        handle.write("## Spectral Performance\n\n")
        if easiest and hardest:
            handle.write(
                "The easiest evaluated subset has mean max train similarity "
                "`%.6f` and RMSE `%.6f`. The hardest evaluated subset has mean "
                "max train similarity `%.6f` and RMSE `%.6f`.\n\n"
                % (
                    easiest["mean_max_train_similarity"],
                    easiest["rmse"],
                    hardest["mean_max_train_similarity"],
                    hardest["rmse"],
                )
            )
        handle.write("| Subset | n | Mean max train similarity | MAE | RMSE | Bias |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in performance_rows:
            handle.write(
                "| %s | %d | %.6f | %.6f | %.6f | %.6f |\n"
                % (
                    row["subset"],
                    row["test_size"],
                    row["mean_max_train_similarity"],
                    row["mae"],
                    row["rmse"],
                    row["bias_mean_pred_minus_true"],
                )
            )
        handle.write("\n## Overlap Validation\n\n")
        validation = audit_card["overlap_validation"]
        handle.write("- Status: `%s`\n" % validation["status"])
        handle.write("- Monotonic nonincreasing overlap: `%s`\n" % validation["monotonic_nonincreasing"])
        handle.write("\n## Artifacts\n\n")
        for name, artifact_path in audit_card["artifacts"].items():
            handle.write("- `%s`: `%s`\n" % (name, artifact_path))


def _axis_subset_metrics(
    subset: str,
    threshold: Optional[float],
    threshold_rule: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis_values: np.ndarray,
    novelty_values: np.ndarray,
    mask: np.ndarray,
    train_size: Optional[int],
    axis_name: str,
    axis_type: str,
) -> Dict[str, Any]:
    metrics = regression_metrics(y_true[mask], y_pred[mask])
    n = int(mask.sum())
    mean_axis = float(np.mean(axis_values[mask])) if n else None
    median_axis = float(np.median(axis_values[mask])) if n else None
    mean_novelty = float(np.mean(novelty_values[mask])) if n else None
    row = {
        "subset": subset,
        "axis": axis_name,
        "axis_type": axis_type,
        "threshold": threshold,
        "threshold_rule": threshold_rule,
        "spectral_parameter": mean_novelty,
        "train_size": train_size,
        "test_size": n,
        "mean_axis_value": mean_axis,
        "median_axis_value": median_axis,
        "mean_novelty": mean_novelty,
        "cross_split_overlap": mean_axis if axis_type == "similarity" else None,
    }
    row.update(metrics)
    return row


def _write_axis_report(
    path: str,
    audit_card: Dict[str, Any],
    aggregate_metrics: Dict[str, Any],
    performance_rows: Sequence[Dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    easiest = performance_rows[0] if performance_rows else None
    hardest = performance_rows[-1] if performance_rows else None
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# SPECTRA Audit Report\n\n")
        handle.write("Domain: `%s`\n\n" % audit_card["domain"])
        handle.write("Scientific unit: `%s`\n\n" % audit_card["scientific_unit"])
        handle.write("Spectral axis: `%s` (`%s`)\n\n" % (audit_card["spectral_axis"]["name"], audit_card["spectral_axis"]["axis_type"]))
        handle.write("## Aggregate Performance\n\n")
        handle.write("- n: `%s`\n" % aggregate_metrics["n"])
        handle.write("- MAE: `%.6f`\n" % aggregate_metrics["mae"])
        handle.write("- RMSE: `%.6f`\n" % aggregate_metrics["rmse"])
        if aggregate_metrics["r2"] is not None:
            handle.write("- R2: `%.6f`\n" % aggregate_metrics["r2"])
        handle.write("- Bias: `%.6f`\n\n" % aggregate_metrics["bias_mean_pred_minus_true"])
        handle.write("## Spectral Performance\n\n")
        if easiest and hardest:
            handle.write(
                "The easiest evaluated subset has mean novelty `%.6f` and RMSE "
                "`%.6f`. The hardest evaluated subset has mean novelty `%.6f` "
                "and RMSE `%.6f`.\n\n"
                % (
                    easiest["mean_novelty"],
                    easiest["rmse"],
                    hardest["mean_novelty"],
                    hardest["rmse"],
                )
            )
        handle.write("| Subset | n | Mean axis | Mean novelty | MAE | RMSE | Bias |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in performance_rows:
            handle.write(
                "| %s | %d | %.6f | %.6f | %.6f | %.6f | %.6f |\n"
                % (
                    row["subset"],
                    row["test_size"],
                    row["mean_axis_value"],
                    row["mean_novelty"],
                    row["mae"],
                    row["rmse"],
                    row["bias_mean_pred_minus_true"],
                )
            )
        handle.write("\n## Novelty Validation\n\n")
        validation = audit_card["novelty_validation"]
        handle.write("- Status: `%s`\n" % validation["status"])
        handle.write("- Monotonic nondecreasing novelty: `%s`\n" % validation["monotonic_nondecreasing_novelty"])
        handle.write("\n## Artifacts\n\n")
        for name, artifact_path in audit_card["artifacts"].items():
            handle.write("- `%s`: `%s`\n" % (name, artifact_path))


def run_axis_audit(config: SpectralAxisAuditConfig) -> Dict[str, Any]:
    """Run a domain-agnostic SPECTRA audit from an eval table and measured axis."""
    axis_type = _normalize_axis_type(config.axis_type)
    eval_df = pd.read_csv(config.eval_path)
    required = [config.target_col, config.pred_col, config.axis_col]
    missing = [column for column in required if column not in eval_df.columns]
    if missing:
        raise ValueError("%s is missing columns: %s" % (config.eval_path, ", ".join(missing)))

    train_df = pd.read_csv(config.train_path) if config.train_path else None
    train_size = int(len(train_df)) if train_df is not None else None
    y_true = eval_df[config.target_col].to_numpy(dtype=float)
    y_pred = eval_df[config.pred_col].to_numpy(dtype=float)
    axis_values = eval_df[config.axis_col].to_numpy(dtype=float)
    novelty_values = _axis_novelty(axis_values, axis_type)
    aggregate_metrics = regression_metrics(y_true, y_pred)

    thresholds = _axis_thresholds(
        axis_values,
        axis_type,
        config.thresholds,
        config.quantile_bins,
    )
    rows: List[Dict[str, Any]] = []
    all_mask = np.ones(len(eval_df), dtype=bool)
    rows.append(
        _axis_subset_metrics(
            "all_eval",
            None,
            "all",
            y_true,
            y_pred,
            axis_values,
            novelty_values,
            all_mask,
            train_size,
            config.axis_name,
            axis_type,
        )
    )
    for threshold in thresholds:
        if axis_type == "similarity":
            mask = axis_values <= threshold
            rule = "similarity <= threshold"
            subset = "similarity_le_%s" % str(threshold).replace(".", "_")
        else:
            mask = axis_values >= threshold
            rule = "distance >= threshold"
            subset = "distance_ge_%s" % str(threshold).replace(".", "_")
        if int(mask.sum()) == 0 or int(mask.sum()) == len(eval_df):
            continue
        rows.append(
            _axis_subset_metrics(
                subset,
                threshold,
                rule,
                y_true,
                y_pred,
                axis_values,
                novelty_values,
                mask,
                train_size,
                config.axis_name,
                axis_type,
            )
        )

    rows.sort(key=lambda row: float(row["mean_novelty"]) if row["mean_novelty"] is not None else -float("inf"))
    novelty_validation = validate_novelty_curve(rows)
    auspc = compute_auspc_from_rows(rows)

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    split_stats_path = os.path.join(output_dir, "split_stats.csv")
    performance_path = os.path.join(output_dir, "performance_by_axis.csv")
    eval_axis_path = os.path.join(output_dir, "eval_with_axis.csv")
    curve_path = os.path.join(output_dir, "spectral_curve.svg")
    report_path = os.path.join(output_dir, "report.md")
    audit_card_path = os.path.join(output_dir, "audit_card.json")

    split_rows = [
        {
            "subset": row["subset"],
            "axis": row["axis"],
            "axis_type": row["axis_type"],
            "threshold": row["threshold"],
            "threshold_rule": row["threshold_rule"],
            "spectral_parameter": row["spectral_parameter"],
            "train_size": row["train_size"],
            "test_size": row["test_size"],
            "mean_axis_value": row["mean_axis_value"],
            "median_axis_value": row["median_axis_value"],
            "mean_novelty": row["mean_novelty"],
            "cross_split_overlap": row["cross_split_overlap"],
        }
        for row in rows
    ]
    _write_csv(split_stats_path, split_rows)
    _write_csv(performance_path, rows)

    eval_rows = []
    for index, row in eval_df.reset_index(drop=True).iterrows():
        sample_id = (
            row[config.unit_col]
            if config.unit_col and config.unit_col in eval_df.columns
            else "eval_%05d" % index
        )
        eval_rows.append(
            {
                "sample_id": sample_id,
                "y_true": float(y_true[index]),
                "y_pred": float(y_pred[index]),
                "abs_error": float(abs(y_true[index] - y_pred[index])),
                "squared_error": float((y_true[index] - y_pred[index]) ** 2),
                "axis": config.axis_name,
                "axis_type": axis_type,
                "axis_value": float(axis_values[index]),
                "novelty": float(novelty_values[index]),
            }
        )
    _write_csv(eval_axis_path, eval_rows)
    _write_curve_svg(curve_path, rows)

    target_support: Dict[str, Any] = {"computed": False}
    if train_df is not None and config.train_target_col and config.train_target_col in train_df.columns:
        train_y = train_df[config.train_target_col].to_numpy(dtype=float)
        target_support = {
            "computed": True,
            "train_min": float(np.min(train_y)),
            "train_max": float(np.max(train_y)),
            "eval_min": float(np.min(y_true)),
            "eval_max": float(np.max(y_true)),
            "eval_below_train_range": int(np.sum(y_true < np.min(train_y))),
            "eval_above_train_range": int(np.sum(y_true > np.max(train_y))),
            "pred_below_train_range": int(np.sum(y_pred < np.min(train_y))),
            "pred_above_train_range": int(np.sum(y_pred > np.max(train_y))),
        }

    audit_card: Dict[str, Any] = {
        "schema_version": "0.2.0",
        "domain": config.domain,
        "scientific_unit": config.scientific_unit,
        "inputs": {
            "train_path": config.train_path,
            "eval_path": config.eval_path,
            "target_col": config.target_col,
            "pred_col": config.pred_col,
            "axis_col": config.axis_col,
            "unit_col": config.unit_col,
        },
        "spectral_axis": {
            "name": config.axis_name,
            "axis_type": axis_type,
            "source": "provided_eval_axis",
            "interpretation": (
                "Larger values mean more train-like samples."
                if axis_type == "similarity"
                else "Larger values mean more novel or farther-from-train samples."
            ),
        },
        "model_evaluation": {
            "metric_family": "regression",
            "aggregate_metrics": aggregate_metrics,
            "target_support": target_support,
        },
        "property_graphs": [
            {
                "name": config.axis_name,
                "type": "per_eval_spectral_axis",
                "construction": "Agent or upstream code supplied a measured train-test distance/overlap axis for each evaluation unit.",
            }
        ],
        "novelty_validation": novelty_validation,
        "performance_axis_curve": rows,
        "auspc": auspc,
        "warnings": [],
        "artifacts": {
            "audit_card": audit_card_path,
            "split_stats": split_stats_path,
            "performance_by_axis": performance_path,
            "eval_with_axis": eval_axis_path,
            "spectral_curve": curve_path,
            "report": report_path,
        },
    }
    _write_json(audit_card_path, audit_card)
    _write_axis_report(report_path, audit_card, aggregate_metrics, rows)
    return {
        "status": "completed",
        "domain": config.domain,
        "output_dir": output_dir,
        "aggregate_metrics": aggregate_metrics,
        "novelty_validation": novelty_validation,
        "auspc": auspc,
        "artifacts": audit_card["artifacts"],
    }


def run_pairwise_similarity_audit(config: PairwiseSimilarityAuditConfig) -> Dict[str, Any]:
    """Run SPECTRA from an agent/domain-defined pairwise train-eval similarity graph."""
    eval_df = pd.read_csv(config.eval_path)
    sim_df = pd.read_csv(config.similarity_path)
    for path, df, columns in [
        (config.eval_path, eval_df, [config.eval_id_col, config.target_col, config.pred_col]),
        (
            config.similarity_path,
            sim_df,
            [
                config.similarity_eval_id_col,
                config.similarity_train_id_col,
                config.similarity_col,
            ],
        ),
    ]:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError("%s is missing columns: %s" % (path, ", ".join(missing)))

    max_sim = (
        sim_df.groupby(config.similarity_eval_id_col)[config.similarity_col]
        .max()
        .rename("max_train_similarity")
        .reset_index()
    )
    max_sim = max_sim.rename(columns={config.similarity_eval_id_col: config.eval_id_col})
    merged = eval_df.merge(max_sim, on=config.eval_id_col, how="left")
    if merged["max_train_similarity"].isna().any():
        missing_count = int(merged["max_train_similarity"].isna().sum())
        raise ValueError("Pairwise similarity table is missing %d eval units" % missing_count)

    os.makedirs(config.output_dir, exist_ok=True)
    axis_eval_path = os.path.join(config.output_dir, "_eval_with_pairwise_axis_input.csv")
    merged.to_csv(axis_eval_path, index=False)
    result = run_axis_audit(
        SpectralAxisAuditConfig(
            eval_path=axis_eval_path,
            output_dir=config.output_dir,
            target_col=config.target_col,
            pred_col=config.pred_col,
            axis_col="max_train_similarity",
            axis_type="similarity",
            axis_name=config.axis_name,
            domain=config.domain,
            scientific_unit=config.scientific_unit,
            train_path=config.train_path,
            train_target_col=config.train_target_col,
            unit_col=config.eval_id_col,
            thresholds=config.thresholds,
            quantile_bins=config.quantile_bins,
        )
    )

    audit_card_path = result["artifacts"]["audit_card"]
    with open(audit_card_path, encoding="utf-8") as handle:
        audit_card = json.load(handle)
    audit_card["inputs"]["similarity_path"] = config.similarity_path
    audit_card["spectral_axis"]["source"] = "pairwise_train_eval_similarity_graph"
    audit_card["property_graphs"] = [
        {
            "name": config.axis_name,
            "type": "weighted_bipartite_train_eval_similarity_graph",
            "train_nodes": int(sim_df[config.similarity_train_id_col].nunique()),
            "eval_nodes": int(sim_df[config.similarity_eval_id_col].nunique()),
            "edge_count": int(len(sim_df)),
            "edge_weight": config.similarity_col,
            "construction": "Agent or domain adapter supplied pairwise train-eval similarities; SPECTRA reduced each eval unit to max train similarity for the spectral curve.",
        }
    ]
    audit_card["artifacts"]["pairwise_similarity"] = config.similarity_path
    _write_json(audit_card_path, audit_card)
    return result


def run_molecule_audit(config: MoleculeAuditConfig) -> Dict[str, Any]:
    """Run a deterministic molecule SPECTRA audit and write artifacts."""
    thresholds = _parse_thresholds(config.thresholds)
    train_df = pd.read_csv(config.train_path)
    eval_df = pd.read_csv(config.eval_path)

    for path, df, columns in [
        (config.train_path, train_df, [config.smiles_col]),
        (config.eval_path, eval_df, [config.smiles_col, config.eval_target_col, config.pred_col]),
    ]:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError("%s is missing columns: %s" % (path, ", ".join(missing)))

    train_smiles = train_df[config.smiles_col].astype(str).tolist()
    eval_smiles = eval_df[config.smiles_col].astype(str).tolist()
    max_similarity, similarity_meta = compute_molecule_train_similarity(
        train_smiles,
        eval_smiles,
        radius=config.fingerprint_radius,
        n_bits=config.fingerprint_bits,
        prefer_rdkit=config.prefer_rdkit,
    )

    y_true = eval_df[config.eval_target_col].to_numpy(dtype=float)
    y_pred = eval_df[config.pred_col].to_numpy(dtype=float)
    abs_error = np.abs(y_true - y_pred)
    squared_error = (y_true - y_pred) ** 2
    aggregate_metrics = regression_metrics(y_true, y_pred)

    rows: List[Dict[str, Any]] = []
    all_mask = np.ones(len(eval_df), dtype=bool)
    rows.append(
        _subset_metrics(
            "all_eval",
            1.0,
            y_true,
            y_pred,
            max_similarity,
            all_mask,
            len(train_df),
        )
    )
    for threshold in thresholds:
        if threshold == 1.0:
            continue
        mask = max_similarity <= threshold
        rows.append(
            _subset_metrics(
                "max_similarity_le_%s" % str(threshold).replace(".", "_"),
                threshold,
                y_true,
                y_pred,
                max_similarity,
                mask,
                len(train_df),
            )
        )
    rows = [row for row in rows if row["test_size"] > 0]

    overlap_validation = validate_overlap_curve(rows)
    auspc = compute_auspc_from_rows(rows)

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    split_stats_path = os.path.join(output_dir, "split_stats.csv")
    performance_path = os.path.join(output_dir, "performance_by_distance.csv")
    eval_distances_path = os.path.join(output_dir, "eval_with_distance.csv")
    curve_path = os.path.join(output_dir, "spectral_curve.svg")
    report_path = os.path.join(output_dir, "report.md")
    audit_card_path = os.path.join(output_dir, "audit_card.json")

    split_rows = [
        {
            "subset": row["subset"],
            "axis": row["axis"],
            "threshold": row["threshold"],
            "spectral_parameter": row["spectral_parameter"],
            "train_size": row["train_size"],
            "test_size": row["test_size"],
            "cross_split_overlap": row["cross_split_overlap"],
            "mean_max_train_similarity": row["mean_max_train_similarity"],
            "median_max_train_similarity": row["median_max_train_similarity"],
            "mean_train_distance": row["mean_train_distance"],
        }
        for row in rows
    ]
    _write_csv(split_stats_path, split_rows)
    _write_csv(performance_path, rows)

    eval_rows = []
    train_canonical = similarity_meta.get("train_canonical_smiles") or train_smiles
    eval_canonical = similarity_meta.get("eval_canonical_smiles") or eval_smiles
    train_canonical_set = set(train_canonical)
    for index, row in eval_df.reset_index(drop=True).iterrows():
        sample_id = (
            row[config.sample_id_col]
            if config.sample_id_col and config.sample_id_col in eval_df.columns
            else "eval_%05d" % index
        )
        eval_rows.append(
            {
                "sample_id": sample_id,
                "smiles": row[config.smiles_col],
                "canonical_or_raw_smiles": eval_canonical[index],
                "exact_train_smiles_overlap": eval_canonical[index] in train_canonical_set,
                "y_true": float(y_true[index]),
                "y_pred": float(y_pred[index]),
                "abs_error": float(abs_error[index]),
                "squared_error": float(squared_error[index]),
                "max_train_similarity": float(max_similarity[index]),
                "train_distance": float(1.0 - max_similarity[index]),
                "nearest_train_index": similarity_meta["nearest_train_indices"][index],
            }
        )
    _write_csv(eval_distances_path, eval_rows)
    _write_curve_svg(curve_path, rows)

    target_support: Dict[str, Any] = {"computed": False}
    if config.train_target_col in train_df.columns:
        train_y = train_df[config.train_target_col].to_numpy(dtype=float)
        target_support = {
            "computed": True,
            "train_min": float(np.min(train_y)),
            "train_max": float(np.max(train_y)),
            "eval_min": float(np.min(y_true)),
            "eval_max": float(np.max(y_true)),
            "eval_below_train_range": int(np.sum(y_true < np.min(train_y))),
            "eval_above_train_range": int(np.sum(y_true > np.max(train_y))),
            "pred_below_train_range": int(np.sum(y_pred < np.min(train_y))),
            "pred_above_train_range": int(np.sum(y_pred > np.max(train_y))),
        }

    audit_card: Dict[str, Any] = {
        "schema_version": "0.1.0",
        "domain": "molecules",
        "scientific_unit": "molecule",
        "inputs": {
            "train_path": config.train_path,
            "eval_path": config.eval_path,
            "smiles_col": config.smiles_col,
            "train_target_col": config.train_target_col,
            "eval_target_col": config.eval_target_col,
            "pred_col": config.pred_col,
        },
        "model_evaluation": {
            "metric_family": "regression",
            "aggregate_metrics": aggregate_metrics,
            "target_support": target_support,
        },
        "novelty_axes": [
            {
                "name": "max_train_similarity",
                "distance": "1 - max_train_similarity",
                "method": similarity_meta["method"],
                "scientific_rationale": "Held-out molecules with lower similarity to the training set are less covered by the training chemistry.",
            }
        ],
        "property_graphs": [
            {
                "name": "train_eval_similarity_graph",
                "type": "weighted_bipartite_nearest_neighbor_summary",
                "train_nodes": int(len(train_df)),
                "eval_nodes": int(len(eval_df)),
                "edge_weight": similarity_meta["method"],
                "construction": "For each eval molecule, compute maximum similarity to any training molecule.",
            }
        ],
        "overlap_validation": overlap_validation,
        "performance_distance_curve": rows,
        "auspc": auspc,
        "warnings": similarity_meta.get("warnings", []),
        "artifacts": {
            "audit_card": audit_card_path,
            "split_stats": split_stats_path,
            "performance_by_distance": performance_path,
            "eval_with_distance": eval_distances_path,
            "spectral_curve": curve_path,
            "report": report_path,
        },
    }
    _write_json(audit_card_path, audit_card)
    _write_report(report_path, audit_card, aggregate_metrics, rows)

    return {
        "status": "completed",
        "domain": "molecules",
        "output_dir": output_dir,
        "aggregate_metrics": aggregate_metrics,
        "overlap_validation": overlap_validation,
        "auspc": auspc,
        "artifacts": audit_card["artifacts"],
    }
