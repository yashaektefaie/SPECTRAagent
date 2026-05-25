"""Run benchmark-mode SPECTRA on the hidden motif sequence bundle.

This script is intentionally agent-facing and sequence-generic. It does not use
hidden motif family labels. Given labeled DNA sequences, it tests candidate
sequence-similarity axes, generates measured-overlap train/test splits, trains a
fresh lightweight model for each split, and reports performance versus
train-test overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def read_labeled_sequences(input_dir: Path) -> pd.DataFrame:
    all_sequences = input_dir / "all_sequences.csv"
    if all_sequences.exists():
        df = pd.read_csv(all_sequences)
    else:
        train = pd.read_csv(input_dir / "train.csv")
        train["source_split"] = "original_train"
        eval_df = pd.read_csv(input_dir / "eval_predictions.csv")
        eval_df["source_split"] = "original_eval"
        df = pd.concat(
            [
                train[["sample_id", "sequence", "y_true", "source_split"]],
                eval_df[["sample_id", "sequence", "y_true", "source_split"]],
            ],
            ignore_index=True,
        )
    required = {"sample_id", "sequence", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: %s" % ", ".join(sorted(missing)))
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["sequence"] = df["sequence"].astype(str).str.upper()
    df["y_true"] = df["y_true"].astype(int)
    if "source_split" not in df.columns:
        df["source_split"] = "all_labeled"
    if df["y_true"].nunique() != 2:
        raise ValueError("Benchmark runner expects a binary target with both classes present.")
    return df.reset_index(drop=True)


def central_window(sequences: pd.Series, width: int = 30) -> list[str]:
    windows = []
    for seq in sequences:
        start = max(0, (len(seq) - width) // 2)
        windows.append(seq[start:start + width])
    return windows


def central_minimizer_groups(sequences: pd.Series, k: int = 6, width: int = 30) -> np.ndarray:
    groups = []
    for window in central_window(sequences, width=width):
        kmers = [window[index:index + k] for index in range(max(1, len(window) - k + 1))]
        groups.append(min(kmers) if kmers else window)
    return np.asarray(groups, dtype=object)


def vectorize_for_similarity(sequences: pd.Series, axis_name: str) -> sparse.csr_matrix:
    if axis_name == "central_5mer_jaccard":
        corpus = central_window(sequences, width=34)
        ngram_range = (5, 5)
    elif axis_name == "global_6mer_jaccard":
        corpus = sequences.tolist()
        ngram_range = (6, 6)
    else:
        raise ValueError("Unknown axis: %s" % axis_name)
    vectorizer = CountVectorizer(analyzer="char", ngram_range=ngram_range, lowercase=False, binary=True)
    return vectorizer.fit_transform(corpus).astype(np.int8).tocsr()


def mean_max_jaccard(features: sparse.csr_matrix, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, float]:
    x_train = features[train_idx]
    x_test = features[test_idx]
    train_sums = np.asarray(x_train.getnnz(axis=1), dtype=float)
    test_sums = np.asarray(x_test.getnnz(axis=1), dtype=float)
    intersections = (x_test @ x_train.T).tocsr()
    max_values = np.zeros(x_test.shape[0], dtype=float)
    for row_id in range(intersections.shape[0]):
        start = intersections.indptr[row_id]
        end = intersections.indptr[row_id + 1]
        if start == end:
            continue
        cols = intersections.indices[start:end]
        data = intersections.data[start:end].astype(float)
        denom = test_sums[row_id] + train_sums[cols] - data
        valid = denom > 0
        if np.any(valid):
            max_values[row_id] = float(np.max(data[valid] / denom[valid]))
    return {
        "mean_max_train_similarity": float(np.mean(max_values)),
        "median_max_train_similarity": float(np.median(max_values)),
        "p10_max_train_similarity": float(np.quantile(max_values, 0.10)),
        "p90_max_train_similarity": float(np.quantile(max_values, 0.90)),
        "fraction_test_with_exact_neighbor": float(np.mean(max_values >= 0.999)),
    }


def candidate_splits(df: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    y = df["y_true"].to_numpy()
    groups = central_minimizer_groups(df["sequence"])
    candidates: list[dict[str, Any]] = []

    for repeat, (_, test_idx) in enumerate(
        StratifiedShuffleSplit(n_splits=8, test_size=0.30, random_state=seed).split(np.zeros(len(df)), y)
    ):
        test_idx = np.asarray(test_idx, dtype=int)
        train_idx = np.setdiff1d(np.arange(len(df)), test_idx)
        candidates.append({
            "split_family": "random_iid",
            "repeat": repeat,
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    splitter = GroupShuffleSplit(n_splits=40, test_size=0.30, random_state=seed + 17)
    for repeat, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(df)), y, groups=groups)):
        candidates.append({
            "split_family": "central_minimizer_group_holdout",
            "repeat": repeat,
            "train_idx": np.asarray(train_idx, dtype=int),
            "test_idx": np.asarray(test_idx, dtype=int),
        })

    return candidates


def select_overlap_levels(candidates: list[dict[str, Any]], features: sparse.csr_matrix, axis_name: str) -> list[dict[str, Any]]:
    scored = []
    for candidate in candidates:
        stats = mean_max_jaccard(features, candidate["train_idx"], candidate["test_idx"])
        enriched = dict(candidate)
        enriched.update(stats)
        enriched["axis_name"] = axis_name
        scored.append(enriched)

    scored.sort(key=lambda item: item["mean_max_train_similarity"], reverse=True)
    if len(scored) < 3:
        raise ValueError("Need at least three candidate splits.")
    selected_indices = [0, len(scored) // 2, len(scored) - 1]
    selected = []
    labels = ["high_overlap", "middle_overlap", "low_overlap"]
    for label, index in zip(labels, selected_indices):
        item = dict(scored[index])
        item["overlap_level"] = label
        item["spectral_parameter"] = {"high_overlap": 0.0, "middle_overlap": 0.5, "low_overlap": 1.0}[label]
        selected.append(item)
    return selected


def train_and_evaluate(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> dict[str, Any]:
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(5, 6), lowercase=False, binary=True)
    x_train = vectorizer.fit_transform(train["sequence"])
    x_test = vectorizer.transform(test["sequence"])
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0015,
        max_iter=100,
        tol=1e-4,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x_train, train["y_true"].to_numpy())
    y_true = test["y_true"].to_numpy()
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "positive_rate_test": float(np.mean(y_true)),
        "mean_predicted_positive_probability": float(np.mean(y_score)),
        "n_features": int(x_train.shape[1]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    novelty = np.asarray([1.0 - float(row["cross_split_overlap"]) for row in rows], dtype=float)
    difficulty = np.asarray([-float(row["auroc"]) for row in rows], dtype=float)
    if len(rows) < 2 or np.var(novelty) == 0 or np.var(difficulty) == 0:
        corr = None
    else:
        corr = float(np.corrcoef(novelty, difficulty)[0, 1])
    easiest = max(rows, key=lambda row: float(row["cross_split_overlap"]))
    hardest = min(rows, key=lambda row: float(row["cross_split_overlap"]))
    degradation = float(easiest["auroc"]) - float(hardest["auroc"])
    if corr is not None and corr >= 0.5 and degradation > 0:
        status = "monotonic_supported"
    elif degradation > 0:
        status = "localized_or_weak_supported"
    else:
        status = "not_explanatory"
    return {
        "status": status,
        "difficulty_novelty_correlation": corr,
        "auroc_degradation_high_to_low_overlap": degradation,
        "easiest_split": easiest["split_id"],
        "hardest_split": hardest["split_id"],
    }


def run(input_dir: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    df = read_labeled_sequences(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidate_splits(df, seed=seed)
    all_split_rows: list[dict[str, Any]] = []
    all_perf_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    hypothesis_scores: dict[str, Any] = {}

    for axis_name in ["central_5mer_jaccard", "global_6mer_jaccard"]:
        features = vectorize_for_similarity(df["sequence"], axis_name)
        selected = select_overlap_levels(candidates, features, axis_name)
        perf_rows_for_axis: list[dict[str, Any]] = []
        for split_number, split in enumerate(selected):
            split_id = f"{axis_name}_{split['overlap_level']}"
            train_ids = df.iloc[split["train_idx"]]["sample_id"].tolist()
            test_ids = df.iloc[split["test_idx"]]["sample_id"].tolist()
            (split_dir / f"{split_id}_train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
            (split_dir / f"{split_id}_test_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
            metrics = train_and_evaluate(
                df,
                split["train_idx"],
                split["test_idx"],
                seed=seed + split_number,
            )
            split_row = {
                "split_id": split_id,
                "axis_name": axis_name,
                "split_family": split["split_family"],
                "overlap_level": split["overlap_level"],
                "spectral_parameter": split["spectral_parameter"],
                "cross_split_overlap": split["mean_max_train_similarity"],
                "median_max_train_similarity": split["median_max_train_similarity"],
                "p10_max_train_similarity": split["p10_max_train_similarity"],
                "p90_max_train_similarity": split["p90_max_train_similarity"],
                "fraction_test_with_exact_neighbor": split["fraction_test_with_exact_neighbor"],
                "train_size": int(len(train_ids)),
                "test_size": int(len(test_ids)),
            }
            perf_row = dict(split_row)
            perf_row.update(metrics)
            all_split_rows.append(split_row)
            all_perf_rows.append(perf_row)
            perf_rows_for_axis.append(perf_row)
            manifest_rows.append({
                "split_id": split_id,
                "axis_name": axis_name,
                "train_ids_path": str(split_dir / f"{split_id}_train_ids.txt"),
                "test_ids_path": str(split_dir / f"{split_id}_test_ids.txt"),
                "model_type": "SGDClassifier(log_loss) over train-only 5/6-mer CountVectorizer",
                "fresh_model_trained": True,
                "random_seed": seed + split_number,
                "model_artifact_path": "not_saved_reproducible_from_manifest",
            })
        hypothesis_scores[axis_name] = score_curve(perf_rows_for_axis)

    best_axis = max(
        hypothesis_scores,
        key=lambda axis: hypothesis_scores[axis]["auroc_degradation_high_to_low_overlap"],
    )
    write_csv(output_dir / "split_stats.csv", all_split_rows)
    write_csv(output_dir / "performance_by_overlap.csv", all_perf_rows)
    write_csv(output_dir / "retraining_manifest.csv", manifest_rows)
    (output_dir / "similarity_hypothesis_scores.json").write_text(
        json.dumps(hypothesis_scores, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_card = {
        "mode": "benchmark",
        "scientific_unit": "DNA sequence",
        "task": "binary regulatory activity prediction",
        "model_family": "lightweight k-mer logistic classifier",
        "split_construction": "measured train-test k-mer Jaccard overlap with random and central-minimizer group holdout candidates",
        "fresh_model_per_split": True,
        "candidate_axes": list(hypothesis_scores),
        "selected_axis": best_axis,
        "similarity_hypothesis_scores": hypothesis_scores,
        "artifacts": {
            "split_stats": str(output_dir / "split_stats.csv"),
            "performance_by_overlap": str(output_dir / "performance_by_overlap.csv"),
            "retraining_manifest": str(output_dir / "retraining_manifest.csv"),
        },
    }
    (output_dir / "audit_card.json").write_text(json.dumps(audit_card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Benchmark-Mode SPECTRA Hidden Motif Audit",
        "",
        "Mode: benchmark. Raw labels and a trainable baseline are available, so fixed-prediction binning is not used as primary evidence.",
        "",
        f"Selected axis: `{best_axis}`.",
        "",
        "| axis | status | AUROC degradation high-to-low overlap | difficulty-novelty correlation |",
        "|---|---:|---:|---:|",
    ]
    for axis, score in hypothesis_scores.items():
        corr = score["difficulty_novelty_correlation"]
        corr_text = "NA" if corr is None else f"{corr:.4f}"
        report.append(
            f"| {axis} | {score['status']} | {score['auroc_degradation_high_to_low_overlap']:.4f} | {corr_text} |"
        )
    report.extend([
        "",
        "Each curve point trained a fresh SGD logistic classifier with a train-only 5/6-mer vocabulary.",
        "",
    ])
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return audit_card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()
    audit_card = run(args.input_dir, args.output_dir, args.seed)
    print(json.dumps({"output_dir": str(args.output_dir), "selected_axis": audit_card["selected_axis"]}, indent=2))


if __name__ == "__main__":
    main()
