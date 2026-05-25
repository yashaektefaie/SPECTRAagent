"""Create blinded model-generalizability bundles for the cross-domain agent trial."""

import json
import os
import shutil
from typing import Dict

import pandas as pd


RUN_ROOT = "/ewsc/yektefai/spectra_agent_ablation_20260513"
BOOM_SOURCE = "/ewsc/yektefai/spectra_assets/hard_blind_molecular_model_eval/agent_visible"
NABENCH_SOURCE = "/ewsc/yektefai/spectra_paper_results_20260513/deterministic/nabench_Martin_2018_myc_enhancer_prospective"
PERTURBENCH_SOURCE = "/ewsc/yektefai/spectra_paper_results_20260513/deterministic/perturbench_devel_component_support"


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def create_molecule_bundle(root: str) -> Dict:
    out_dir = os.path.join(root, "bundles", "molecules")
    visible = os.path.join(out_dir, "agent_visible")
    grading = os.path.join(out_dir, "grading_only")
    os.makedirs(visible, exist_ok=True)
    os.makedirs(grading, exist_ok=True)

    for name in ["train.csv", "eval_predictions.csv", "metadata.json"]:
        shutil.copy2(os.path.join(BOOM_SOURCE, name), os.path.join(visible, name))

    metadata = {
        "bundle_id": "cross_domain_molecules_v1",
        "domain": "molecules",
        "description": "Anonymous molecular property model. Evaluate generalization using only visible train data, held-out predictions, and metadata.",
        "scientific_unit_hint": "molecule",
        "agent_visible_files": {
            "train": os.path.join(visible, "train.csv"),
            "eval_predictions": os.path.join(visible, "eval_predictions.csv"),
            "metadata": os.path.join(visible, "metadata.json")
        },
        "forbidden_context": [
            "BOOM paper or repository context",
            "grading_only files",
            "precomputed train-test similarity",
            "precomputed split labels"
        ]
    }
    write_json(os.path.join(visible, "trial_metadata.json"), metadata)
    return metadata


def create_sequence_bundle(root: str) -> Dict:
    out_dir = os.path.join(root, "bundles", "sequence_fitness")
    visible = os.path.join(out_dir, "agent_visible")
    grading = os.path.join(out_dir, "grading_only")
    os.makedirs(visible, exist_ok=True)
    os.makedirs(grading, exist_ok=True)

    train = pd.read_csv(os.path.join(NABENCH_SOURCE, "train.csv"))
    eval_df = pd.read_csv(os.path.join(NABENCH_SOURCE, "eval_predictions.csv"))
    train_visible = train[["sample_id", "mutant", "sequence", "y"]].rename(columns={"sample_id": "train_id"})
    eval_visible = eval_df[["sample_id", "mutant", "sequence", "y_true", "y_pred"]]
    train_visible.to_csv(os.path.join(visible, "train.csv"), index=False)
    eval_visible.to_csv(os.path.join(visible, "eval_predictions.csv"), index=False)

    grading_payload = {
        "source": NABENCH_SOURCE,
        "hidden_columns": [
            "mutation_center",
            "mutation_depth",
            "selected_similarity_axis",
            "pairwise_similarity",
            "spectra_summary"
        ],
        "deterministic_selected_axis": "mutation_centered_window_identity_similarity"
    }
    write_json(os.path.join(grading, "hidden_reference.json"), grading_payload)
    metadata = {
        "bundle_id": "cross_domain_sequence_fitness_v1",
        "domain": "sequence_fitness",
        "description": "Anonymous nucleotide variant fitness model. Evaluate generalization using only visible train data and held-out predictions.",
        "scientific_unit_hint": "mutated nucleotide sequence",
        "agent_visible_files": {
            "train": os.path.join(visible, "train.csv"),
            "eval_predictions": os.path.join(visible, "eval_predictions.csv"),
            "metadata": os.path.join(visible, "trial_metadata.json")
        },
        "forbidden_context": [
            "NABench paper or repository context",
            "grading_only files",
            "precomputed mutation centers or depths",
            "precomputed SPECTRA summary"
        ]
    }
    write_json(os.path.join(visible, "trial_metadata.json"), metadata)
    return metadata


def create_perturbation_bundle(root: str) -> Dict:
    out_dir = os.path.join(root, "bundles", "perturbation_biology")
    visible = os.path.join(out_dir, "agent_visible")
    grading = os.path.join(out_dir, "grading_only")
    os.makedirs(visible, exist_ok=True)
    os.makedirs(grading, exist_ok=True)

    train = pd.read_csv(os.path.join(PERTURBENCH_SOURCE, "train_single_gene_responses.csv"))
    eval_df = pd.read_csv(os.path.join(PERTURBENCH_SOURCE, "eval_combination_gene_predictions.csv"))
    eval_visible = eval_df[["sample_id", "condition", "gene", "cell_type", "y_true", "y_pred"]]
    train.to_csv(os.path.join(visible, "train.csv"), index=False)
    eval_visible.to_csv(os.path.join(visible, "eval_predictions.csv"), index=False)

    grading_payload = {
        "source": PERTURBENCH_SOURCE,
        "hidden_columns": [
            "component_support_similarity",
            "component_support_pairwise_similarity",
            "profile_support_curve"
        ],
        "deterministic_selected_axis": "component_support_similarity"
    }
    write_json(os.path.join(grading, "hidden_reference.json"), grading_payload)
    metadata = {
        "bundle_id": "cross_domain_perturbation_biology_v1",
        "domain": "perturbation_biology",
        "description": "Anonymous perturbation-response model. Evaluate generalization using only visible single-perturbation training responses and held-out combination predictions.",
        "scientific_unit_hint": "perturbation-condition expression response profile",
        "agent_visible_files": {
            "train": os.path.join(visible, "train.csv"),
            "eval_predictions": os.path.join(visible, "eval_predictions.csv"),
            "metadata": os.path.join(visible, "trial_metadata.json")
        },
        "forbidden_context": [
            "PerturBench paper or repository context",
            "grading_only files",
            "precomputed component-support similarity",
            "precomputed SPECTRA summary"
        ]
    }
    write_json(os.path.join(visible, "trial_metadata.json"), metadata)
    return metadata


def main() -> int:
    os.makedirs(RUN_ROOT, exist_ok=True)
    manifest = {
        "run_root": RUN_ROOT,
        "bundles": {
            "molecules": create_molecule_bundle(RUN_ROOT),
            "sequence_fitness": create_sequence_bundle(RUN_ROOT),
            "perturbation_biology": create_perturbation_bundle(RUN_ROOT)
        },
        "excluded_domains": {
            "regulatory_dna": "DART-Eval requires Synapse-hosted data and genome references not available in this local run."
        }
    }
    write_json(os.path.join(RUN_ROOT, "manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
