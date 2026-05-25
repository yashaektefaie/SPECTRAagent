"""Build a full Open Problems PBMC blinded model-evaluation bundle.

The input files are the public NeurIPS 2023 Open Problems differential
expression H5AD train/test files. The agent-visible bundle intentionally hides
SMILES so chemical structure is not handed to the agent as an obvious schema
field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _read_layer_matrix(path: Path, layer: str) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    obj = ad.read_h5ad(path)
    try:
        obs = obj.obs.copy()
        genes = [str(x) for x in obj.var_names]
        values = np.asarray(obj.layers[layer], dtype=np.float32)
    finally:
        obj.file.close() if getattr(obj, "isbacked", False) else None
    return obs, genes, values


def _visible_metadata(obs: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = ["cell_type", "sm_lincs_id", "sm_name", "dose_uM"]
    out = obs.loc[:, keep].copy()
    out.insert(0, "sample_id", [f"{prefix}_{i:04d}" for i in range(len(out))])
    return out


def build_bundle(train_h5ad: Path, test_h5ad: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    visible = output_dir / "agent_visible"
    hidden = output_dir / "hidden_for_scoring"
    visible.mkdir(exist_ok=True)
    hidden.mkdir(exist_ok=True)

    train_obs, genes, train_y = _read_layer_matrix(train_h5ad, "logFC")
    test_obs, test_genes, test_y = _read_layer_matrix(test_h5ad, "logFC")
    if genes != test_genes:
        raise ValueError("Train and test gene axes do not match")

    train_visible = _visible_metadata(train_obs, "train")
    eval_visible = _visible_metadata(test_obs, "eval")
    train_visible.to_csv(visible / "train_metadata.csv", index=False)
    eval_visible.to_csv(visible / "eval_metadata.csv", index=False)
    pd.DataFrame({"gene": genes}).to_csv(visible / "genes.csv", index=False)

    # Drug-mean baseline: for each held-out row, predict the mean logFC vector
    # of the same compound across available training cell types. This uses full
    # training data but does not use held-out labels.
    train_drugs = train_obs["sm_lincs_id"].astype(str).to_numpy()
    test_drugs = test_obs["sm_lincs_id"].astype(str).to_numpy()
    global_mean = train_y.mean(axis=0)
    drug_to_mean: dict[str, np.ndarray] = {}
    for drug in sorted(set(train_drugs)):
        mask = train_drugs == drug
        drug_to_mean[drug] = train_y[mask].mean(axis=0)
    pred_y = np.vstack([drug_to_mean.get(drug, global_mean) for drug in test_drugs]).astype(np.float32)

    np.savez_compressed(visible / "train_logfc.npz", logfc=train_y)
    np.savez_compressed(visible / "eval_true_logfc.npz", logfc=test_y.astype(np.float32))
    np.savez_compressed(visible / "eval_pred_logfc.npz", logfc=pred_y)

    model_description = {
        "model_name": "compound_mean_logfc_baseline",
        "training_data": "full Open Problems NeurIPS 2023 PBMC differential-expression train set",
        "prediction_rule": (
            "For each held-out compound/cell-type row, predict the mean logFC vector "
            "for the same compound across all available training rows; use the global "
            "training mean only if a compound is absent from training."
        ),
        "target": "gene-wise log fold-change vector",
        "n_train_rows": int(train_y.shape[0]),
        "n_eval_rows": int(test_y.shape[0]),
        "n_genes": int(train_y.shape[1]),
    }
    (visible / "model_description.json").write_text(
        json.dumps(model_description, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = """# Open Problems PBMC Model Evaluation Bundle

You are given artifacts for a trained scientific prediction model.

Files:

- `model_description.json`: model and prediction-rule metadata.
- `train_metadata.csv`: metadata for training rows.
- `train_logfc.npz`: training target matrix, key `logfc`, rows aligned to `train_metadata.csv`.
- `eval_metadata.csv`: metadata for held-out rows.
- `eval_true_logfc.npz`: held-out true target matrix, key `logfc`, rows aligned to `eval_metadata.csv`.
- `eval_pred_logfc.npz`: model predictions for held-out rows, key `logfc`, rows aligned to `eval_metadata.csv`.
- `genes.csv`: gene axis for all matrices.

All matrices contain gene-wise log fold-change values.
"""
    (visible / "README.md").write_text(readme, encoding="utf-8")

    # Hidden fields are excluded from the agent-visible bundle, but retained so
    # we can score whether agents asked for or reconstructed chemical structure.
    train_hidden = train_obs.copy()
    test_hidden = test_obs.copy()
    train_hidden.insert(0, "sample_id", train_visible["sample_id"].to_numpy())
    test_hidden.insert(0, "sample_id", eval_visible["sample_id"].to_numpy())
    train_hidden.to_csv(hidden / "train_metadata_with_smiles.csv", index=False)
    test_hidden.to_csv(hidden / "eval_metadata_with_smiles.csv", index=False)

    metrics = {
        "overall_rmse": float(np.sqrt(np.mean((pred_y - test_y) ** 2))),
        "overall_mae": float(np.mean(np.abs(pred_y - test_y))),
        "per_cell_type": {},
    }
    for cell_type in sorted(test_obs["cell_type"].astype(str).unique()):
        mask = test_obs["cell_type"].astype(str).to_numpy() == cell_type
        metrics["per_cell_type"][cell_type] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean((pred_y[mask] - test_y[mask]) ** 2))),
            "mae": float(np.mean(np.abs(pred_y[mask] - test_y[mask]))),
        }
    (hidden / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5ad", required=True, type=Path)
    parser.add_argument("--test-h5ad", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    build_bundle(args.train_h5ad, args.test_h5ad, args.output_dir)


if __name__ == "__main__":
    main()
