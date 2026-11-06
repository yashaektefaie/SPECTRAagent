#!/usr/bin/env python3
"""Fit NCT-only fixed references and write same-image target scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def labels(dataset: str) -> pd.DataFrame:
    with (ROOT / "data" / "manifests" / f"{dataset}.sealed.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    mapping = json.loads((ROOT / "audit" / "label_maps.json").read_text())[dataset]
    return pd.DataFrame({"tile_id": [r["tile_id"] for r in rows], "label": [mapping.get(r["raw_label"]) for r in rows]})


def write_scores(dataset: str, method: str, ids: np.ndarray, y: np.ndarray, logits: np.ndarray, classes: list[str]) -> None:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    pred = np.asarray(classes)[probs.argmax(axis=1)]
    frame = pd.DataFrame({"tile_id": ids, "dataset": dataset, "label": y, "prediction": pred})
    for j, c in enumerate(classes):
        frame[f"logit_{c}"] = logits[:, j]
        frame[f"prob_{c}"] = probs[:, j]
    frame.to_csv(ROOT / "scores" / f"{dataset}.{method}.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default="nct_crc_100k")
    p.add_argument("--targets", nargs="+", default=["crc_val_7k", "stanford_crc_val_small"])
    args = p.parse_args()
    state = json.loads((ROOT / "audit" / "state.json").read_text())
    if not state["target_labels_accessible"]:
        raise SystemExit("controller guard: fixed references require post-freeze label release")

    train = pd.read_csv(ROOT / "features" / f"{args.reference}.prospective.csv").merge(labels(args.reference), on="tile_id")
    exclude = {"path", "tile_id", "dhash64", "label"}
    columns = [c for c in train.columns if c not in exclude and np.issubdtype(train[c].dtype, np.number)]
    handcrafted = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=20260714),
    )
    handcrafted.fit(train[columns], train.label)
    joblib.dump({"columns": columns, "pipeline": handcrafted}, ROOT / "references" / "handcrafted_logistic.joblib")
    for dataset in args.targets:
        frame = pd.read_csv(ROOT / "features" / f"{dataset}.prospective.csv").merge(labels(dataset), on="tile_id")
        logits = handcrafted.decision_function(frame[columns])
        write_scores(dataset, "handcrafted_logistic", frame.tile_id.to_numpy(), frame.label.to_numpy(), logits, handcrafted.classes_.tolist())

    train_m = np.load(ROOT / "features" / f"{args.reference}.mobilenet_v3_small_imagenet1k_v1.npz")
    train_labels = labels(args.reference).set_index("tile_id").loc[train_m["tile_id"]]["label"].to_numpy()
    x = train_m["embedding"].astype(float)
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    centroids = np.stack([x[train_labels == c].mean(axis=0) for c in CLASSES])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-12)
    np.savez_compressed(ROOT / "references" / "mobilenet_nearest_centroid.npz", classes=np.asarray(CLASSES), centroids=centroids)
    for dataset in args.targets:
        target = np.load(ROOT / "features" / f"{dataset}.mobilenet_v3_small_imagenet1k_v1.npz")
        tx = target["embedding"].astype(float)
        tx /= np.linalg.norm(tx, axis=1, keepdims=True).clip(1e-12)
        y = labels(dataset).set_index("tile_id").loc[target["tile_id"]]["label"].to_numpy()
        write_scores(dataset, "mobilenet_nearest_centroid", target["tile_id"], y, tx @ centroids.T, CLASSES)
    print("fixed references fit on NCT only; same-image target scores written")


if __name__ == "__main__":
    main()
