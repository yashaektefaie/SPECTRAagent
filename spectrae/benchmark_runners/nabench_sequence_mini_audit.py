"""Run a lightweight SPECTRA audit on one NABench sequence-fitness assay.

This runner is intentionally small. It does not reproduce the full NABench
foundation-model suite. It uses one public DMS assay from the NABench repository,
trains a position-aware ridge baseline, and evaluates a contiguous mutational
region holdout as an iterative SPECTRA sequence-novelty audit.
"""

import argparse
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from spectrae.audit import PairwiseSimilarityAuditConfig, run_pairwise_similarity_audit


DEFAULT_CACHE_ROOT = os.environ.get(
    "XDG_CACHE_HOME",
    os.path.join(os.path.expanduser("~"), ".cache"),
)
DEFAULT_REPO_DIR = os.environ.get(
    "NABENCH_REPO_DIR",
    os.path.join(DEFAULT_CACHE_ROOT, "spectra", "repos", "nabench"),
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "SPECTRA_NABENCH_OUTPUT_DIR",
    os.path.join(DEFAULT_CACHE_ROOT, "spectra", "runs", "nabench_sequence_mini_audit"),
)
DEFAULT_DATASET = "Martin_2018_myc_enhancer.csv"
DEFAULT_AXIS_ORDER = (
    "mutation_position_support_similarity",
    "sequence_identity_similarity",
    "mutation_centered_window_identity_similarity",
    "mutation_depth_support_similarity",
    "fitness_support_similarity",
    "position_fitness_composite_similarity",
)


def parse_mutation_positions(mutant: Any) -> List[int]:
    """Extract 1-indexed mutation positions from NABench mutant strings."""
    if mutant is None or (isinstance(mutant, float) and math.isnan(mutant)):
        return []
    return [int(value) for value in re.findall(r"\d+", str(mutant))]


def mutation_center(positions: Sequence[int]) -> Optional[float]:
    if not positions:
        return None
    return float(np.mean(np.asarray(positions, dtype=float)))


def sequence_tokens(sequence: str) -> str:
    """Represent a sequence as position-aware character tokens."""
    return " ".join("pos%d_%s" % (index + 1, base) for index, base in enumerate(str(sequence)))


def sequence_identity(left: str, right: str) -> float:
    """Compute ungapped positional sequence identity with length penalty."""
    left_value = str(left)
    right_value = str(right)
    denominator = max(len(left_value), len(right_value))
    if denominator == 0:
        return 1.0
    shared_length = min(len(left_value), len(right_value))
    matches = sum(1 for index in range(shared_length) if left_value[index] == right_value[index])
    return float(matches / denominator)


def centered_window(sequence: str, center: float, radius: int = 10) -> str:
    """Extract a mutation-centered sequence window with boundary clipping."""
    value = str(sequence)
    if not value:
        return ""
    index = max(0, min(len(value) - 1, int(round(float(center))) - 1))
    start = max(0, index - radius)
    end = min(len(value), index + radius + 1)
    return value[start:end]


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, Any]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(truth)),
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(math.sqrt(mean_squared_error(truth, pred))),
        "r2": float(r2_score(truth, pred)) if len(truth) > 1 else None,
        "bias_mean_pred_minus_true": float(np.mean(pred - truth)),
    }


def fit_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame, alpha: float) -> np.ndarray:
    train_tokens = [sequence_tokens(value) for value in train_df["sequence"].astype(str)]
    eval_tokens = [sequence_tokens(value) for value in eval_df["sequence"].astype(str)]
    model = make_pipeline(
        CountVectorizer(token_pattern=r"[^ ]+", lowercase=False),
        Ridge(alpha=alpha),
    )
    model.fit(train_tokens, train_df["y"].to_numpy(dtype=float))
    return model.predict(eval_tokens)


def choose_holdout_interval(
    positions: np.ndarray,
    holdout_fraction: float,
) -> Tuple[float, float]:
    """Choose a central contiguous interval over observed mutation positions."""
    if not 0.05 <= holdout_fraction <= 0.8:
        raise ValueError("holdout_fraction must be between 0.05 and 0.8")
    unique_positions = np.unique(positions.astype(float))
    minimum = float(np.min(unique_positions))
    maximum = float(np.max(unique_positions))
    width = max(1.0, (maximum - minimum) * holdout_fraction)
    midpoint = minimum + (maximum - minimum) / 2.0
    start = midpoint - width / 2.0
    end = midpoint + width / 2.0
    if start == end:
        raise ValueError("Could not choose a non-empty contiguous holdout interval")
    return start, end


def nearest_position_similarity_rows(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    length_scale: float,
) -> List[Dict[str, Any]]:
    """Compute exact nearest-position similarity edges for eval units."""
    train_positions = train_df["mutation_center"].to_numpy(dtype=float)
    train_ids = train_df["sample_id"].astype(str).tolist()
    rows = []
    for _, eval_row in eval_df.iterrows():
        eval_position = float(eval_row["mutation_center"])
        distances = np.abs(train_positions - eval_position)
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        similarity = float(math.exp(-nearest_distance / length_scale))
        rows.append(
            {
                "sample_id": str(eval_row["sample_id"]),
                "train_id": train_ids[nearest_index],
                "similarity": similarity,
                "position_distance": nearest_distance,
            }
        )
    return rows


def nearest_sequence_identity_rows(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Compute exact max train sequence identity for each eval unit."""
    train_sequences = train_df["sequence"].astype(str).tolist()
    train_ids = train_df["sample_id"].astype(str).tolist()
    rows = []
    for _, eval_row in eval_df.iterrows():
        eval_sequence = str(eval_row["sequence"])
        similarities = [sequence_identity(eval_sequence, train_sequence) for train_sequence in train_sequences]
        nearest_index = int(np.argmax(similarities))
        rows.append(
            {
                "sample_id": str(eval_row["sample_id"]),
                "train_id": train_ids[nearest_index],
                "similarity": float(similarities[nearest_index]),
            }
        )
    return rows


def nearest_centered_window_identity_rows(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    radius: int,
) -> List[Dict[str, Any]]:
    """Compute max identity between mutation-centered local sequence windows."""
    train_windows = [
        centered_window(row.sequence, row.mutation_center, radius=radius)
        for row in train_df.itertuples(index=False)
    ]
    train_ids = train_df["sample_id"].astype(str).tolist()
    rows = []
    for _, eval_row in eval_df.iterrows():
        eval_window = centered_window(
            str(eval_row["sequence"]),
            float(eval_row["mutation_center"]),
            radius=radius,
        )
        similarities = [sequence_identity(eval_window, train_window) for train_window in train_windows]
        nearest_index = int(np.argmax(similarities))
        rows.append(
            {
                "sample_id": str(eval_row["sample_id"]),
                "train_id": train_ids[nearest_index],
                "similarity": float(similarities[nearest_index]),
                "window_radius": int(radius),
            }
        )
    return rows


def nearest_mutation_depth_rows(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    depth_scale: float,
) -> List[Dict[str, Any]]:
    """Compute similarity by number of mutated positions."""
    train_depths = train_df["mutation_depth"].to_numpy(dtype=float)
    train_ids = train_df["sample_id"].astype(str).tolist()
    rows = []
    for _, eval_row in eval_df.iterrows():
        distances = np.abs(train_depths - float(eval_row["mutation_depth"]))
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        rows.append(
            {
                "sample_id": str(eval_row["sample_id"]),
                "train_id": train_ids[nearest_index],
                "similarity": float(math.exp(-nearest_distance / depth_scale)),
                "mutation_depth_distance": nearest_distance,
            }
        )
    return rows


def nearest_fitness_support_rows(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    fitness_scale: float,
) -> List[Dict[str, Any]]:
    """Compute post-hoc target-support similarity from observed fitness values."""
    train_y = train_df["y"].to_numpy(dtype=float)
    train_ids = train_df["sample_id"].astype(str).tolist()
    rows = []
    for _, eval_row in eval_df.iterrows():
        distances = np.abs(train_y - float(eval_row["y"]))
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        rows.append(
            {
                "sample_id": str(eval_row["sample_id"]),
                "train_id": train_ids[nearest_index],
                "similarity": float(math.exp(-nearest_distance / fitness_scale)),
                "fitness_distance": nearest_distance,
            }
        )
    return rows


def nearest_position_fitness_composite_rows(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    length_scale: float,
    fitness_scale: float,
) -> List[Dict[str, Any]]:
    """Compute a composite of position support and post-hoc fitness support."""
    train_positions = train_df["mutation_center"].to_numpy(dtype=float)
    train_y = train_df["y"].to_numpy(dtype=float)
    train_ids = train_df["sample_id"].astype(str).tolist()
    rows = []
    for _, eval_row in eval_df.iterrows():
        position_distances = np.abs(train_positions - float(eval_row["mutation_center"]))
        fitness_distances = np.abs(train_y - float(eval_row["y"]))
        position_similarities = np.exp(-position_distances / length_scale)
        fitness_similarities = np.exp(-fitness_distances / fitness_scale)
        similarities = np.sqrt(position_similarities * fitness_similarities)
        nearest_index = int(np.argmax(similarities))
        rows.append(
            {
                "sample_id": str(eval_row["sample_id"]),
                "train_id": train_ids[nearest_index],
                "similarity": float(similarities[nearest_index]),
                "position_distance": float(position_distances[nearest_index]),
                "fitness_distance": float(fitness_distances[nearest_index]),
            }
        )
    return rows


def load_assay(path: str, max_rows: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in ["mutant", "DMS_score", "sequence"] if column not in df.columns]
    if missing:
        raise ValueError("%s is missing required columns: %s" % (path, ", ".join(missing)))
    df = df[["mutant", "DMS_score", "sequence"]].copy()
    df = df.dropna(subset=["DMS_score", "sequence"])
    df["mutation_positions"] = df["mutant"].apply(parse_mutation_positions)
    df["mutation_center"] = df["mutation_positions"].apply(mutation_center)
    df = df.dropna(subset=["mutation_center"]).reset_index(drop=True)
    df["mutation_depth"] = df["mutation_positions"].apply(len).astype(float)
    df["y"] = df["DMS_score"].astype(float)
    df["sample_id"] = ["seq_%05d" % index for index in range(len(df))]
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=13).sort_index().reset_index(drop=True)
        df["sample_id"] = ["seq_%05d" % index for index in range(len(df))]
    return df


def robust_positive_scale(values: Sequence[float], fallback: float = 1.0) -> float:
    """Return a positive scale for exponential kernels."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return fallback
    q25, q75 = np.quantile(array, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= 0:
        scale = float(np.std(array))
    if scale <= 0:
        scale = fallback
    return max(float(scale), 1e-12)


def axis_metadata(axis_id: str) -> Dict[str, Any]:
    metadata = {
        "mutation_position_support_similarity": {
            "name": "mutation_position_support_similarity",
            "hypothesis": "Performance degrades when eval mutations are farther from training mutation positions.",
            "leakage_risk": "none",
            "computation_strategy": "exact_nearest_neighbor_over_mutation_positions",
            "next_if_rejected": "Try sequence identity or target-support axes; position alone may not capture functional novelty.",
        },
        "sequence_identity_similarity": {
            "name": "sequence_identity_similarity",
            "hypothesis": "Performance degrades when eval sequences have lower identity to any training sequence.",
            "leakage_risk": "none",
            "computation_strategy": "exact_all_train_candidates_max_sequence_identity",
            "next_if_rejected": "Try mutation-depth or fitness-support axes; raw sequence identity may saturate for single-mutant assays.",
        },
        "mutation_centered_window_identity_similarity": {
            "name": "mutation_centered_window_identity_similarity",
            "hypothesis": "Performance degrades when the local sequence context around an eval mutation has lower identity to any training mutation-centered context.",
            "leakage_risk": "none",
            "computation_strategy": "exact_all_train_candidates_max_centered_window_identity",
            "next_if_rejected": "Try mutation-position, motif-disruption, or assay-specific regulatory annotations; local windows may be too short or too conserved.",
        },
        "mutation_depth_support_similarity": {
            "name": "mutation_depth_support_similarity",
            "hypothesis": "Performance degrades when eval variants have mutation depths not represented in training.",
            "leakage_risk": "none",
            "computation_strategy": "exact_nearest_neighbor_over_mutation_depth",
            "next_if_rejected": "Try target-support or composite axes; mutation count may be constant in single-mutant assays.",
        },
        "fitness_support_similarity": {
            "name": "fitness_support_similarity",
            "hypothesis": "Performance degrades when eval fitness values lie farther from training fitness support.",
            "leakage_risk": "post_hoc_uses_eval_labels",
            "computation_strategy": "exact_nearest_neighbor_over_observed_fitness",
            "next_if_rejected": "Try a composite axis or a richer biological feature axis.",
        },
        "position_fitness_composite_similarity": {
            "name": "position_fitness_composite_similarity",
            "hypothesis": "Performance degrades when eval variants are jointly distant in mutation position and observed fitness support.",
            "leakage_risk": "post_hoc_uses_eval_labels",
            "computation_strategy": "exact_composite_nearest_neighbor_position_and_fitness",
            "next_if_rejected": "Escalate to richer domain features such as motif disruption, structure, or assay family.",
        },
    }
    if axis_id not in metadata:
        raise ValueError("Unknown axis id: %s" % axis_id)
    return metadata[axis_id].copy()


def build_similarity_rows(
    axis_id: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    length_scale: float,
    fitness_scale: float,
    window_radius: int = 10,
) -> List[Dict[str, Any]]:
    if axis_id == "mutation_position_support_similarity":
        return nearest_position_similarity_rows(train_df, eval_df, length_scale=length_scale)
    if axis_id == "sequence_identity_similarity":
        return nearest_sequence_identity_rows(train_df, eval_df)
    if axis_id == "mutation_centered_window_identity_similarity":
        return nearest_centered_window_identity_rows(train_df, eval_df, radius=window_radius)
    if axis_id == "mutation_depth_support_similarity":
        return nearest_mutation_depth_rows(train_df, eval_df, depth_scale=1.0)
    if axis_id == "fitness_support_similarity":
        return nearest_fitness_support_rows(train_df, eval_df, fitness_scale=fitness_scale)
    if axis_id == "position_fitness_composite_similarity":
        return nearest_position_fitness_composite_rows(
            train_df,
            eval_df,
            length_scale=length_scale,
            fitness_scale=fitness_scale,
        )
    raise ValueError("Unknown axis id: %s" % axis_id)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if len(left) < 2 or len(right) < 2:
        return None
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def score_curve(
    performance_path: str,
    min_subset_size: int,
) -> Dict[str, Any]:
    curve_df = pd.read_csv(performance_path)
    curve_rows = curve_df.to_dict(orient="records")
    usable_rows = [
        row
        for row in curve_rows
        if row.get("rmse") is not None
        and not pd.isna(row.get("rmse"))
        and int(row.get("test_size") or 0) >= min_subset_size
    ]
    if not usable_rows:
        usable_rows = [
            row
            for row in curve_rows
            if row.get("rmse") is not None and not pd.isna(row.get("rmse"))
        ]

    all_eval_row = curve_rows[0] if curve_rows else {}
    all_eval_rmse = float(all_eval_row["rmse"]) if all_eval_row else None
    if len(usable_rows) < 2:
        return {
            "status": "not_evaluable",
            "reason": "Fewer than two usable curve points.",
            "all_eval_rmse": all_eval_rmse,
            "lowest_overlap_rmse": all_eval_rmse,
            "lowest_overlap_delta_rmse": 0.0,
            "max_rmse": all_eval_rmse,
            "max_rmse_subset": str(all_eval_row.get("subset", "all_eval")) if all_eval_row else None,
            "max_rmse_delta": 0.0,
            "rmse_novelty_correlation": None,
            "slope": None,
            "point_count": len(usable_rows),
            "rows": curve_rows,
        }

    novelty = np.asarray([float(row["mean_novelty"]) for row in usable_rows], dtype=float)
    rmse = np.asarray([float(row["rmse"]) for row in usable_rows], dtype=float)
    order = np.argsort(novelty)
    novelty = novelty[order]
    rmse = rmse[order]
    ordered_rows = [usable_rows[index] for index in order]

    correlation = _safe_corr(novelty, rmse)
    if len(np.unique(novelty)) >= 2:
        slope = float(np.polyfit(novelty, rmse, 1)[0])
    else:
        slope = None

    lowest_overlap_row = ordered_rows[-1]
    max_rmse_row = max(ordered_rows, key=lambda row: float(row["rmse"]))
    lowest_overlap_rmse = float(lowest_overlap_row["rmse"])
    max_rmse = float(max_rmse_row["rmse"])
    lowest_delta = lowest_overlap_rmse - float(all_eval_rmse)
    max_delta = max_rmse - float(all_eval_rmse)
    tolerance = max(1e-12, abs(float(all_eval_rmse)) * 0.02)

    if correlation is not None and correlation >= 0.5 and lowest_delta > tolerance:
        status = "monotonic_supported"
        reason = "RMSE rises with novelty and the lowest-overlap subset is harder than all eval."
    elif max_delta > tolerance:
        status = "localized_supported"
        reason = "A lower-support region is harder, but the hardest subset is not the final curve point."
    elif correlation is not None and correlation > 0.25:
        status = "weak_supported"
        reason = "RMSE weakly increases with novelty but the effect is not strong enough to rely on."
    else:
        status = "not_explanatory"
        reason = "This similarity axis did not explain a monotonic or localized performance drop."

    return {
        "status": status,
        "reason": reason,
        "all_eval_rmse": float(all_eval_rmse),
        "lowest_overlap_rmse": lowest_overlap_rmse,
        "lowest_overlap_test_size": int(lowest_overlap_row["test_size"]),
        "lowest_overlap_delta_rmse": float(lowest_delta),
        "max_rmse": max_rmse,
        "max_rmse_subset": str(max_rmse_row["subset"]),
        "max_rmse_test_size": int(max_rmse_row["test_size"]),
        "max_rmse_mean_novelty": float(max_rmse_row["mean_novelty"]),
        "max_rmse_delta": float(max_delta),
        "rmse_novelty_correlation": correlation,
        "slope": slope,
        "point_count": len(usable_rows),
        "min_subset_size": int(min_subset_size),
        "rows": curve_rows,
    }


def run_one_similarity_hypothesis(
    axis_id: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_path: str,
    train_path: str,
    output_dir: str,
    length_scale: float,
    fitness_scale: float,
    window_radius: int,
    quantile_bins: int,
    min_subset_size: int,
) -> Dict[str, Any]:
    meta = axis_metadata(axis_id)
    axis_dir = os.path.join(output_dir, "similarity_hypotheses", axis_id)
    audit_dir = os.path.join(output_dir, "spectra_audits", axis_id)
    os.makedirs(axis_dir, exist_ok=True)
    pairwise_path = os.path.join(axis_dir, "pairwise_similarity.csv")
    pairwise_rows = build_similarity_rows(
        axis_id,
        train_df,
        eval_df,
        length_scale=length_scale,
        fitness_scale=fitness_scale,
        window_radius=window_radius,
    )
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)
    result = run_pairwise_similarity_audit(
        PairwiseSimilarityAuditConfig(
            eval_path=eval_path,
            similarity_path=pairwise_path,
            output_dir=audit_dir,
            target_col="y_true",
            pred_col="y_pred",
            eval_id_col="sample_id",
            similarity_eval_id_col="sample_id",
            similarity_train_id_col="train_id",
            similarity_col="similarity",
            domain="nucleotide_fitness",
            scientific_unit="mutated_sequence",
            train_path=train_path,
            train_id_col="sample_id",
            train_target_col="y",
            axis_name=axis_id,
            quantile_bins=quantile_bins,
        )
    )
    curve_summary = score_curve(
        result["artifacts"]["performance_by_axis"],
        min_subset_size=min_subset_size,
    )
    return {
        "axis_id": axis_id,
        "metadata": meta,
        "pairwise_similarity_path": pairwise_path,
        "spectra_result": result,
        "curve_summary": curve_summary,
        "decision": {
            "status": curve_summary["status"],
            "reason": curve_summary["reason"],
            "next_if_rejected": meta["next_if_rejected"],
        },
    }


def choose_best_axis(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    priority = {
        "monotonic_supported": 4,
        "localized_supported": 3,
        "weak_supported": 2,
        "not_explanatory": 1,
        "not_evaluable": 0,
    }

    def key(result: Dict[str, Any]) -> Tuple[int, int, float, float]:
        summary = result["curve_summary"]
        leakage_rank = 1 if result["metadata"].get("leakage_risk") == "none" else 0
        return (
            priority.get(summary["status"], 0),
            leakage_rank,
            float(summary.get("max_rmse_delta") or 0.0),
            float(summary.get("lowest_overlap_delta_rmse") or 0.0),
        )

    return max(results, key=key)


def build_iteration_trace(results: Sequence[Dict[str, Any]], selected_axis_id: str) -> List[Dict[str, Any]]:
    trace = []
    for index, result in enumerate(results, start=1):
        summary = result["curve_summary"]
        selected = result["axis_id"] == selected_axis_id
        if selected:
            action = "select_axis_for_reporting"
        elif summary["status"] in {"monotonic_supported", "localized_supported"}:
            action = "retain_as_supported_secondary_axis"
        else:
            action = "try_next_similarity_hypothesis"
        trace.append(
            {
                "step": index,
                "axis_id": result["axis_id"],
                "status": summary["status"],
                "reason": summary["reason"],
                "action": action,
                "next_if_rejected": result["metadata"]["next_if_rejected"] if action == "try_next_similarity_hypothesis" else None,
            }
        )
    return trace


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_summary_report(path: str, summary: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    random_metrics = summary["random_split_metrics"]
    contiguous_metrics = summary["contiguous_holdout_metrics"]
    spectra = summary["spectra_result"]
    curve_summary = summary["curve_summary"]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# NABench Sequence Mini-Audit\n\n")
        handle.write("Dataset: `%s`\n\n" % summary["dataset"])
        handle.write("Scientific unit: mutated DNA/RNA sequence.\n\n")
        handle.write("## Paperclip Selection\n\n")
        handle.write(
            "Paperclip identified NABench as a strong sequence-fitness depth demo because "
            "the paper explicitly contrasts random cross-validation with contiguous "
            "cross-validation for extrapolation to unseen mutational regions, and the "
            "repository ships public DMS CSV files.\n\n"
        )
        handle.write("## Setup\n\n")
        handle.write("- Train split: mutations outside a central contiguous position interval.\n")
        handle.write("- Eval split: mutations inside that interval.\n")
        handle.write("- Model: position-aware ridge regression over sequence character tokens.\n")
        handle.write("- Similarity mode: iterative hypothesis loop over multiple candidate axes.\n")
        handle.write("- Initial axis: mutation-position support.\n")
        handle.write("- Follow-up axes: sequence identity, local window identity, mutation depth, fitness support, and position-fitness composite.\n\n")
        handle.write("Holdout interval: `%.3f` to `%.3f`.\n\n" % (summary["holdout_start"], summary["holdout_end"]))
        handle.write("## Random Baseline\n\n")
        handle.write("- n: `%d`\n" % random_metrics["n"])
        handle.write("- RMSE: `%.6f`\n" % random_metrics["rmse"])
        handle.write("- MAE: `%.6f`\n" % random_metrics["mae"])
        handle.write("- R2: `%.6f`\n\n" % random_metrics["r2"])
        handle.write("## Contiguous Holdout\n\n")
        handle.write("- n: `%d`\n" % contiguous_metrics["n"])
        handle.write("- RMSE: `%.6f`\n" % contiguous_metrics["rmse"])
        handle.write("- MAE: `%.6f`\n" % contiguous_metrics["mae"])
        handle.write("- R2: `%.6f`\n\n" % contiguous_metrics["r2"])
        handle.write("## Iterative Similarity Search\n\n")
        handle.write("| Step | Axis | Leakage risk | Status | Action |\n")
        handle.write("| ---: | --- | --- | --- | --- |\n")
        for step in summary["iteration_trace"]:
            axis_result = next(
                result
                for result in summary["iterative_similarity_results"]
                if result["axis_id"] == step["axis_id"]
            )
            handle.write(
                "| %d | `%s` | `%s` | `%s` | `%s` |\n"
                % (
                    step["step"],
                    step["axis_id"],
                    axis_result["metadata"]["leakage_risk"],
                    step["status"],
                    step["action"],
                )
            )
        handle.write("\n")
        handle.write("Selected axis: `%s`\n\n" % summary["selected_similarity_axis"])
        if summary["similarity_definition"]["leakage_risk"] != "none":
            handle.write(
                "Selected-axis caveat: this axis is post-hoc and uses evaluation "
                "labels. It is appropriate for explaining an observed failure mode, "
                "but not for prospective split design before labels are available.\n\n"
            )

        handle.write("## Selected SPECTRA Result\n\n")
        handle.write("- AUSPC: `%s`\n" % spectra["auspc"].get("value"))
        handle.write("- Novelty validation: `%s`\n" % spectra["novelty_validation"]["status"])
        handle.write("- Curve status: `%s`\n" % curve_summary["status"])
        handle.write("- Curve reason: %s\n" % curve_summary["reason"])
        handle.write("- All-eval curve RMSE: `%.6f`\n" % curve_summary["all_eval_rmse"])
        handle.write("- Lowest-overlap subset RMSE: `%.6f`\n" % curve_summary["lowest_overlap_rmse"])
        handle.write("- Max curve RMSE: `%.6f` in `%s`\n" % (curve_summary["max_rmse"], curve_summary["max_rmse_subset"]))
        handle.write("- Audit directory: `%s`\n\n" % summary["audit_output_dir"])
        handle.write("## Interpretation\n\n")
        handle.write(summary["interpretation"] + "\n\n")
        handle.write("## Artifacts\n\n")
        for name, artifact_path in spectra["artifacts"].items():
            handle.write("- `%s`: `%s`\n" % (name, artifact_path))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = os.path.join(args.repo_dir, "data", args.dataset)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    df = load_assay(dataset_path, args.max_rows)
    positions = df["mutation_center"].to_numpy(dtype=float)
    holdout_start, holdout_end = choose_holdout_interval(positions, args.holdout_fraction)
    eval_mask = (df["mutation_center"] >= holdout_start) & (df["mutation_center"] <= holdout_end)
    train_df = df.loc[~eval_mask].copy().reset_index(drop=True)
    eval_df = df.loc[eval_mask].copy().reset_index(drop=True)
    if len(train_df) < 10 or len(eval_df) < 10:
        raise ValueError("Contiguous split is too small: train=%d eval=%d" % (len(train_df), len(eval_df)))

    contiguous_pred = fit_predict(train_df, eval_df, alpha=args.alpha)
    eval_df["y_true"] = eval_df["y"].astype(float)
    eval_df["y_pred"] = contiguous_pred.astype(float)
    contiguous_metrics = regression_metrics(eval_df["y_true"], eval_df["y_pred"])

    random_train, random_eval = train_test_split(
        df,
        test_size=len(eval_df),
        random_state=13,
        shuffle=True,
    )
    random_pred = fit_predict(random_train.reset_index(drop=True), random_eval.reset_index(drop=True), alpha=args.alpha)
    random_metrics = regression_metrics(random_eval["y"].to_numpy(dtype=float), random_pred)

    length_scale = args.length_scale
    if length_scale is None:
        sequence_lengths = df["sequence"].astype(str).map(len).to_numpy(dtype=float)
        length_scale = max(1.0, float(np.median(sequence_lengths)) * 0.10)
    fitness_scale = args.fitness_scale
    if fitness_scale is None:
        fitness_scale = robust_positive_scale(train_df["y"].to_numpy(dtype=float), fallback=1.0)

    train_path = os.path.join(output_dir, "train.csv")
    eval_path = os.path.join(output_dir, "eval_predictions.csv")

    train_df[["sample_id", "mutant", "sequence", "mutation_center", "mutation_depth", "y"]].to_csv(train_path, index=False)
    eval_df[["sample_id", "mutant", "sequence", "mutation_center", "mutation_depth", "y_true", "y_pred"]].to_csv(eval_path, index=False)

    axis_order = [
        axis_id.strip()
        for axis_id in args.axis_order.split(",")
        if axis_id.strip()
    ]
    if not axis_order:
        axis_order = list(DEFAULT_AXIS_ORDER)

    hypothesis_results = []
    for axis_id in axis_order:
        hypothesis_results.append(
            run_one_similarity_hypothesis(
                axis_id=axis_id,
                train_df=train_df,
                eval_df=eval_df,
                eval_path=eval_path,
                train_path=train_path,
                output_dir=output_dir,
                length_scale=length_scale,
                fitness_scale=fitness_scale,
                window_radius=args.window_radius,
                quantile_bins=args.quantile_bins,
                min_subset_size=args.min_subset_size,
            )
        )
    selected_result = choose_best_axis(hypothesis_results)
    selected_axis_id = selected_result["axis_id"]
    spectra_result = selected_result["spectra_result"]
    curve_summary = selected_result["curve_summary"]
    iteration_trace = build_iteration_trace(hypothesis_results, selected_axis_id)

    if curve_summary["status"] == "monotonic_supported":
        interpretation = (
            "The iterative SPECTRA loop found a similarity definition whose curve "
            "supports monotonic degradation: RMSE rises with measured novelty and "
            "the lowest-overlap subset is harder than the full contiguous evaluation set."
        )
    elif curve_summary["status"] == "localized_supported":
        interpretation = (
            "The iterative SPECTRA loop found a localized failure axis: at least "
            "one lower-support region is harder than the full contiguous evaluation "
            "set, but the hardest region is not the final novelty point."
        )
    elif curve_summary["status"] == "weak_supported":
        interpretation = (
            "The iterative SPECTRA loop found only weak evidence for this selected "
            "axis. It should be treated as a candidate explanation rather than a "
            "confirmed novelty driver."
        )
    else:
        interpretation = (
            "The iterative SPECTRA loop did not find a strong explanatory axis in "
            "this candidate set. That is a useful audit result: the next decision "
            "should be to add richer sequence biology features such as motif "
            "disruption, structural region, or assay-family context."
        )
    interpretation += (
        " It is a depth-demo pilot, not a full NABench reproduction: it uses one "
        "assay and one lightweight baseline rather than the full NFM suite."
    )

    summary = {
        "status": "completed",
        "paperclip_search_ids": {
            "sequence": "s_7f98e379",
        },
        "paper": {
            "title": "NABench: Large-Scale Benchmarks of Nucleotide Foundation Models for Fitness Prediction",
            "paperclip_id": "arx_2511.02888",
            "repo": "https://github.com/mrzzmrzz/NABench",
        },
        "dataset": args.dataset,
        "dataset_path": dataset_path,
        "row_count_after_filtering": int(len(df)),
        "train_count": int(len(train_df)),
        "eval_count": int(len(eval_df)),
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "holdout_fraction": args.holdout_fraction,
        "length_scale": length_scale,
        "fitness_scale": fitness_scale,
        "window_radius": args.window_radius,
        "similarity_definition": {
            **selected_result["metadata"],
        },
        "similarity_hypothesis_order": axis_order,
        "selected_similarity_axis": selected_axis_id,
        "iterative_similarity_results": hypothesis_results,
        "iteration_trace": iteration_trace,
        "computation_strategy": selected_result["metadata"]["computation_strategy"],
        "operating_point": "contiguous_mutational_region_split",
        "random_split_metrics": random_metrics,
        "contiguous_holdout_metrics": contiguous_metrics,
        "spectra_result": spectra_result,
        "curve_summary": curve_summary,
        "interpretation": interpretation,
        "audit_output_dir": spectra_result["output_dir"],
        "artifacts": {
            "train": train_path,
            "eval_predictions": eval_path,
            "selected_pairwise_similarity": selected_result["pairwise_similarity_path"],
            "selected_spectra_audit": spectra_result["output_dir"],
            "all_similarity_hypotheses": os.path.join(output_dir, "similarity_hypotheses"),
            "all_spectra_audits": os.path.join(output_dir, "spectra_audits"),
            "summary_json": os.path.join(output_dir, "summary.json"),
            "summary_report": os.path.join(output_dir, "summary.md"),
        },
    }
    write_json(summary["artifacts"]["summary_json"], summary)
    write_summary_report(summary["artifacts"]["summary_report"], summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a NABench SPECTRA sequence mini-audit")
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--length-scale", type=float)
    parser.add_argument("--fitness-scale", type=float)
    parser.add_argument("--window-radius", type=int, default=10)
    parser.add_argument("--quantile-bins", type=int, default=5)
    parser.add_argument("--min-subset-size", type=int, default=10)
    parser.add_argument("--axis-order", default=",".join(DEFAULT_AXIS_ORDER))
    parser.add_argument("--max-rows", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
