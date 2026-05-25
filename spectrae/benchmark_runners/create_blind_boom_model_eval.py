"""Create a blinded molecular model-evaluation bundle from the BOOM mini-audit.

Agent-facing files do not mention BOOM and do not expose ID/OOD labels. Hidden
labels are written separately for grading.
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


DEFAULT_ASSET_DIR = os.environ.get(
    "SPECTRA_ASSET_DIR",
    "/ewsc/yektefai/spectra_assets"
    if os.path.isdir("/ewsc/yektefai")
    else "spectra_assets",
)
DEFAULT_SOURCE_DIR = os.path.join(DEFAULT_ASSET_DIR, "boom_numeric_pilot")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_ASSET_DIR, "blind_molecular_model_eval")


def write_rows(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_bundle(source_dir: str, output_dir: str, hide_derived_metrics: bool = False) -> Dict[str, Any]:
    split_path = os.path.join(source_dir, "10k_dft_data_with_ood_splits.csv")
    predictions_path = os.path.join(source_dir, "artifacts", "model_predictions.csv")
    model_path = os.path.join(source_dir, "artifacts", "model.pkl")
    if not os.path.exists(split_path):
        raise FileNotFoundError(split_path)
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(predictions_path)

    split_df = pd.read_csv(split_path)
    pred_df = pd.read_csv(predictions_path)
    target = "density"
    train_df = split_df[split_df[f"{target}_train"] == 1].copy()

    train_rows = []
    for index, row in train_df.reset_index(drop=True).iterrows():
        train_rows.append(
            {
                "train_id": "train_%05d" % index,
                "smiles": row["smiles"],
                "target": target,
                "y": float(row[target]),
            }
        )

    eval_rows = []
    hidden_rows = []
    for index, row in pred_df.reset_index(drop=True).iterrows():
        sample_id = "eval_%05d" % index
        visible_row = {
            "sample_id": sample_id,
            "smiles": row["smiles"],
            "target": row["target"],
            "y_true": float(row["y_true"]),
            "y_pred": float(row["y_pred"]),
        }
        if not hide_derived_metrics:
            visible_row.update(
                {
                    "abs_error": float(row["abs_error"]),
                    "squared_error": float(row["squared_error"]),
                    "max_train_tanimoto": float(row["max_train_tanimoto"]),
                }
            )
        eval_rows.append(visible_row)
        hidden_rows.append(
            {
                "sample_id": sample_id,
                "hidden_split": row["split"],
                "abs_error": float(row["abs_error"]),
                "squared_error": float(row["squared_error"]),
                "max_train_tanimoto": float(row["max_train_tanimoto"]),
            }
        )

    train_out = os.path.join(output_dir, "agent_visible", "train.csv")
    eval_out = os.path.join(output_dir, "agent_visible", "eval_predictions.csv")
    metadata_out = os.path.join(output_dir, "agent_visible", "metadata.json")
    hidden_out = os.path.join(output_dir, "grading_only", "hidden_eval_labels.csv")
    manifest_out = os.path.join(output_dir, "manifest.json")

    write_rows(train_out, train_rows, ["train_id", "smiles", "target", "y"])
    eval_fields = [
        "sample_id",
        "smiles",
        "target",
        "y_true",
        "y_pred",
    ]
    if not hide_derived_metrics:
        eval_fields.extend(["abs_error", "squared_error", "max_train_tanimoto"])
    write_rows(eval_out, eval_rows, eval_fields)
    write_rows(hidden_out, hidden_rows, ["sample_id", "hidden_split", "abs_error", "squared_error", "max_train_tanimoto"])

    metadata = {
        "bundle_id": "anonymous_molecular_model_eval_v1",
        "description": "A trained molecular property model and held-out prediction table. The task is to evaluate model generalization from the provided artifacts.",
        "model_type": "random forest over Morgan fingerprints",
        "target": target,
        "derived_metrics_hidden": hide_derived_metrics,
        "agent_visible_files": {
            "train": train_out,
            "eval_predictions": eval_out,
            "metadata": metadata_out,
            "model": model_path,
        },
        "columns": {
            "train.csv": {
                "smiles": "Molecule string.",
                "y": "Training target value.",
            },
            "eval_predictions.csv": {
                "y_true": "Held-out target value.",
                "y_pred": "Model prediction.",
                **(
                    {}
                    if hide_derived_metrics
                    else {
                        "abs_error": "Absolute prediction error.",
                        "squared_error": "Squared prediction error.",
                        "max_train_tanimoto": "Maximum Morgan-fingerprint Tanimoto similarity from this held-out molecule to any training molecule.",
                    }
                ),
            },
        },
        "do_not_use_for_agent_prompt": [
            hidden_out,
        ],
    }
    with open(metadata_out, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    manifest = {
        "source_dir": source_dir,
        "output_dir": output_dir,
        "agent_visible": {
            "train": train_out,
            "eval_predictions": eval_out,
            "metadata": metadata_out,
            "model": model_path,
        },
        "grading_only": {
            "hidden_eval_labels": hidden_out,
        },
        "model_path": model_path,
        "derived_metrics_hidden": hide_derived_metrics,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
    }
    with open(manifest_out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create blind molecular model-evaluation bundle.")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hide-derived-metrics", action="store_true")
    args = parser.parse_args()
    print(json.dumps(create_bundle(args.source_dir, args.output_dir, args.hide_derived_metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
