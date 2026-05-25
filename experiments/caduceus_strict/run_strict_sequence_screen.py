"""Quick benchmark-mode SPECTRA screen for strict Caduceus sequence tasks.

This is not a Caduceus fine-tuning reproduction. It is a lightweight audit
screen over the same original evaluation task data: train fresh k-mer probes on
similarity-controlled splits and report whether simple raw-sequence novelty
axes explain probe performance.
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
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


DEFAULT_TASKS = [
    "nucleotide_transformer:promoter_tata",
    "nucleotide_transformer:enhancers",
    "nucleotide_transformer:H3K4me3",
    "genomic_benchmarks:dummy_mouse_enhancers_ensembl",
    "genomic_benchmarks:human_nontata_promoters",
    "genomic_benchmarks:human_enhancers_cohn",
]


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


def load_task(manifest: pd.DataFrame, task_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    suite, task = task_key.split(":", 1)
    row = manifest[(manifest["source_suite"] == suite) & (manifest["task_name"] == task)]
    if row.empty:
        raise ValueError(f"Task not found in manifest: {task_key}")
    info = row.iloc[0].to_dict()
    train = pd.read_csv(info["train_path"])
    test = pd.read_csv(info["test_path"])
    df = pd.concat([train, test], ignore_index=True)
    return df, info


def stratified_cap(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy().reset_index(drop=True)
    parts = []
    rng = np.random.default_rng(seed)
    for _, group in df.groupby(["original_split", "label"], observed=True):
        take = max(1, int(round(max_rows * len(group) / len(df))))
        parts.append(group.iloc[rng.choice(len(group), size=min(take, len(group)), replace=False)])
    capped = pd.concat(parts, ignore_index=True)
    if len(capped) > max_rows:
        capped = capped.sample(n=max_rows, random_state=seed, stratify=capped["label"])
    return capped.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def central_window(sequences: pd.Series, width: int = 80) -> list[str]:
    windows = []
    for seq in sequences.astype(str):
        start = max(0, (len(seq) - width) // 2)
        windows.append(seq[start:start + width])
    return windows


def minimizer_groups(sequences: pd.Series, k: int = 6, width: int = 80) -> np.ndarray:
    groups = []
    for window in central_window(sequences, width=width):
        kmers = [window[index:index + k] for index in range(max(1, len(window) - k + 1))]
        groups.append(min(kmers) if kmers else window)
    return np.asarray(groups, dtype=object)


def similarity_features(sequences: pd.Series, axis_name: str) -> sparse.csr_matrix:
    if axis_name == "central_6mer_jaccard":
        corpus = central_window(sequences, width=80)
    elif axis_name == "global_6mer_jaccard":
        corpus = sequences.astype(str).tolist()
    else:
        raise ValueError(axis_name)
    vec = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False, binary=True)
    return vec.fit_transform(corpus).astype(np.int8).tocsr()


def mean_max_jaccard(features: sparse.csr_matrix, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, float]:
    x_train = features[train_idx]
    x_test = features[test_idx]
    train_sums = np.asarray(x_train.getnnz(axis=1), dtype=float)
    test_sums = np.asarray(x_test.getnnz(axis=1), dtype=float)
    dots = (x_test @ x_train.T).tocsr()
    max_values = np.zeros(x_test.shape[0], dtype=float)
    for row_id in range(dots.shape[0]):
        start = dots.indptr[row_id]
        end = dots.indptr[row_id + 1]
        if start == end:
            continue
        cols = dots.indices[start:end]
        vals = dots.data[start:end].astype(float)
        denom = test_sums[row_id] + train_sums[cols] - vals
        valid = denom > 0
        if np.any(valid):
            max_values[row_id] = float(np.max(vals[valid] / denom[valid]))
    return {
        "cross_split_overlap": float(np.mean(max_values)),
        "median_max_train_similarity": float(np.median(max_values)),
        "p90_max_train_similarity": float(np.quantile(max_values, 0.90)),
    }


def evaluate_probe(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict[str, Any]:
    y_train = train["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()
    classes = np.unique(np.concatenate([y_train, y_test]))
    vec = CountVectorizer(analyzer="char", ngram_range=(5, 6), lowercase=False, binary=True, max_features=120000)
    x_train = vec.fit_transform(train["sequence"].astype(str))
    x_test = vec.transform(test["sequence"].astype(str))
    loss = "log_loss" if len(classes) == 2 else "modified_huber"
    model = SGDClassifier(
        loss=loss,
        alpha=0.0005,
        max_iter=60,
        tol=1e-3,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    out = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "mcc": float(matthews_corrcoef(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "n_features": int(x_train.shape[1]),
    }
    if len(classes) == 2 and hasattr(model, "predict_proba"):
        score = model.predict_proba(x_test)[:, 1]
        out["auroc"] = float(roc_auc_score(y_test, score))
    else:
        out["auroc"] = ""
    return out


def candidate_splits(df: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    y = df["label"].astype(int).to_numpy()
    groups = minimizer_groups(df["sequence"])
    candidates = []
    for repeat, (_, test_idx) in enumerate(
        StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=seed).split(np.zeros(len(df)), y)
    ):
        train_idx = np.setdiff1d(np.arange(len(df)), np.asarray(test_idx))
        candidates.append({"family": "random_iid", "repeat": repeat, "train_idx": train_idx, "test_idx": np.asarray(test_idx)})
    splitter = GroupShuffleSplit(n_splits=20, test_size=0.25, random_state=seed + 13)
    for repeat, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(df)), y, groups=groups)):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        candidates.append({"family": "central_minimizer_group_holdout", "repeat": repeat, "train_idx": train_idx, "test_idx": test_idx})
    return candidates


def run_task(task_key: str, df: pd.DataFrame, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    capped = stratified_cap(df, max_rows=16000, seed=seed)
    split_rows = []
    perf_rows = []
    for axis_name in ["central_6mer_jaccard", "global_6mer_jaccard"]:
        features = similarity_features(capped["sequence"], axis_name)
        scored = []
        for candidate in candidate_splits(capped, seed):
            stats = mean_max_jaccard(features, candidate["train_idx"], candidate["test_idx"])
            item = dict(candidate)
            item.update(stats)
            scored.append(item)
        scored.sort(key=lambda item: item["cross_split_overlap"], reverse=True)
        if len(scored) < 3:
            continue
        selected = [
            ("high_overlap", 0.0, scored[0]),
            ("middle_overlap", 0.5, scored[len(scored) // 2]),
            ("low_overlap", 1.0, scored[-1]),
        ]
        for level, spectral_parameter, split in selected:
            train = capped.iloc[split["train_idx"]]
            test = capped.iloc[split["test_idx"]]
            split_id = f"{task_key}:{axis_name}:{level}"
            split_row = {
                "split_id": split_id,
                "task_key": task_key,
                "axis_name": axis_name,
                "overlap_level": level,
                "spectral_parameter": spectral_parameter,
                "split_family": split["family"],
                "cross_split_overlap": split["cross_split_overlap"],
                "median_max_train_similarity": split["median_max_train_similarity"],
                "p90_max_train_similarity": split["p90_max_train_similarity"],
                "train_size": int(len(train)),
                "test_size": int(len(test)),
            }
            metrics = evaluate_probe(train, test, seed=seed + int(spectral_parameter * 10))
            perf_row = dict(split_row)
            perf_row.update(metrics)
            split_rows.append(split_row)
            perf_rows.append(perf_row)
    return split_rows, perf_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()
    manifest = pd.read_csv(args.bundle_dir / "tasks_manifest.csv")
    split_rows = []
    perf_rows = []
    original_rows = []
    for offset, task_key in enumerate(args.tasks):
        df, info = load_task(manifest, task_key)
        train = df[df["original_split"] == "train"]
        test = df[df["original_split"] == "test"]
        original = {
            "task_key": task_key,
            "split_id": "original_train_test",
            "train_size": int(len(train)),
            "test_size": int(len(test)),
        }
        original.update(evaluate_probe(stratified_cap(train, 20000, args.seed + offset), test, seed=args.seed + offset))
        original_rows.append(original)
        task_splits, task_perf = run_task(task_key, df, seed=args.seed + offset)
        split_rows.extend(task_splits)
        perf_rows.extend(task_perf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "original_split_probe_metrics.csv", original_rows)
    write_csv(args.output_dir / "split_stats.csv", split_rows)
    write_csv(args.output_dir / "performance_by_overlap.csv", perf_rows)
    scores = {}
    for (task_key, axis_name), group in pd.DataFrame(perf_rows).groupby(["task_key", "axis_name"], observed=True):
        ordered = group.sort_values("cross_split_overlap", ascending=False)
        high = ordered.iloc[0]
        low = ordered.iloc[-1]
        scores[f"{task_key}:{axis_name}"] = {
            "mcc_high_overlap": float(high["mcc"]),
            "mcc_low_overlap": float(low["mcc"]),
            "mcc_degradation_high_to_low": float(high["mcc"] - low["mcc"]),
            "accuracy_high_overlap": float(high["accuracy"]),
            "accuracy_low_overlap": float(low["accuracy"]),
            "cross_split_overlap_high": float(high["cross_split_overlap"]),
            "cross_split_overlap_low": float(low["cross_split_overlap"]),
            "status": "degradation" if float(high["mcc"] - low["mcc"]) > 0 else "not_explanatory",
        }
    (args.output_dir / "similarity_hypothesis_scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "mode": "benchmark",
                "model": "fresh lightweight k-mer downstream probe, not Caduceus fine-tuning",
                "tasks": args.tasks,
                "scores": scores,
                "artifacts": {
                    "original_split_probe_metrics": str(args.output_dir / "original_split_probe_metrics.csv"),
                    "split_stats": str(args.output_dir / "split_stats.csv"),
                    "performance_by_overlap": str(args.output_dir / "performance_by_overlap.csv"),
                    "similarity_hypothesis_scores": str(args.output_dir / "similarity_hypothesis_scores.json"),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
