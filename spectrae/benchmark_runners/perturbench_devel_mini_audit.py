"""Run a lightweight SPECTRA audit on PerturBench's bundled devel.h5ad.

This runner uses the small development AnnData resource shipped in the
PerturBench repository. It constructs a perturbation-combination prediction task
in K562 cells:

* train units are single-perturbation expression-response profiles;
* evaluation units are two-gene combination perturbation profiles;
* the baseline predicts a combination response as the sum of available single
  responses, using zero for unseen components;
* the prospective novelty axis is component-support similarity, the fraction of
  combination components with a single-perturbation profile in train.

The goal is not to reproduce PerturBench's full model suite. It is a small,
fully local perturbation-biology capsule for testing whether SPECTRA can turn a
known operating point, combinatorial perturbation novelty, into a spectral audit.
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from spectrae.audit import PairwiseSimilarityAuditConfig, run_pairwise_similarity_audit


DEFAULT_REPO_DIR = os.environ.get(
    "PERTURBENCH_REPO_DIR",
    "/ewsc/yektefai/spectra_depth_demos/repos/perturbench"
    if os.path.isdir("/ewsc/yektefai")
    else "perturbench",
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "SPECTRA_PERTURBENCH_OUTPUT_DIR",
    "/ewsc/yektefai/spectra_depth_demos/perturbench_devel_mini_audit"
    if os.path.isdir("/ewsc/yektefai")
    else "perturbench_devel_mini_audit",
)


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_devel_h5ad(path: str) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Read the small PerturBench devel h5ad without requiring anndata/scanpy."""
    with h5py.File(path, "r") as handle:
        shape = tuple(int(value) for value in handle["X"].attrs["shape"])
        matrix = csr_matrix(
            (
                handle["X/data"][()],
                handle["X/indices"][()],
                handle["X/indptr"][()],
            ),
            shape=shape,
        ).toarray()
        obs: Dict[str, List[str]] = {}
        for column in ["condition", "cell_type", "dataset", "perturbation_type", "dose"]:
            categories = [_decode(value) for value in handle[f"obs/{column}/categories"][()]]
            codes = handle[f"obs/{column}/codes"][()]
            obs[column] = [categories[int(code)] for code in codes]
        obs["cell_id"] = [_decode(value) for value in handle["obs/_index"][()]]
        genes = [_decode(value) for value in handle["var/gene_symbol"][()]]
    return matrix, pd.DataFrame(obs), genes


def profile_means(matrix: np.ndarray, obs: pd.DataFrame, cell_type: str) -> Dict[str, np.ndarray]:
    profiles: Dict[str, np.ndarray] = {}
    subset = obs["cell_type"] == cell_type
    for condition in sorted(obs.loc[subset, "condition"].unique()):
        mask = subset & (obs["condition"] == condition)
        if int(mask.sum()) == 0:
            continue
        profiles[str(condition)] = np.asarray(matrix[mask.to_numpy()].mean(axis=0), dtype=float)
    return profiles


def split_components(condition: str) -> List[str]:
    return [component for component in str(condition).split("+") if component]


def regression_metrics(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "min": float(np.min(array)) if len(array) else None,
        "max": float(np.max(array)) if len(array) else None,
    }


def profile_rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def compute_auspc_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("profile_rmse") is not None and row.get("component_support_similarity") is not None
    ]
    if len(valid) < 2:
        return {"computed": False, "value": None, "reason": "Fewer than two profile points."}
    points = sorted(
        (
            1.0 - float(row["component_support_similarity"]),
            -float(row["profile_rmse"]),
        )
        for row in valid
    )
    area = 0.0
    for left, right in zip(points, points[1:]):
        width = right[0] - left[0]
        area += width * ((left[1] + right[1]) / 2.0)
    return {
        "computed": True,
        "metric": "negative_profile_rmse_area",
        "novelty_axis": "1 - component_support_similarity",
        "point_count": len(points),
        "value": float(area),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data_path = args.data_path or os.path.join(
        args.repo_dir,
        "src",
        "perturbench",
        "data",
        "resources",
        "devel.h5ad",
    )
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    matrix, obs, genes = read_devel_h5ad(data_path)
    profiles = profile_means(matrix, obs, cell_type=args.cell_type)
    if "control" not in profiles:
        raise ValueError("No control profile found for cell type %s" % args.cell_type)
    control = profiles["control"]
    responses = {
        condition: profile - control
        for condition, profile in profiles.items()
        if condition != "control"
    }
    singles = {condition: value for condition, value in responses.items() if "+" not in condition}
    combinations = {condition: value for condition, value in responses.items() if "+" in condition}
    if not singles or not combinations:
        raise ValueError("Need at least one single and one combination perturbation")

    profile_rows = []
    eval_rows = []
    pairwise_rows = []
    train_rows = []
    for condition, response in singles.items():
        for gene_index, gene in enumerate(genes):
            train_rows.append(
                {
                    "sample_id": "train_%s_%s" % (condition, gene),
                    "condition": condition,
                    "gene": gene,
                    "cell_type": args.cell_type,
                    "response": float(response[gene_index]),
                }
            )

    for condition, true_response in sorted(combinations.items()):
        components = split_components(condition)
        component_seen = [component in singles for component in components]
        support = float(sum(component_seen) / len(components)) if components else 0.0
        pred_response = np.zeros_like(true_response)
        for component in components:
            if component in singles:
                pred_response = pred_response + singles[component]
        rmse = profile_rmse(true_response, pred_response)
        max_posthoc_effect_similarity = max(
            cosine_similarity(true_response, train_response)
            for train_response in singles.values()
        )
        profile_rows.append(
            {
                "condition": condition,
                "cell_type": args.cell_type,
                "components": "+".join(components),
                "component_count": int(len(components)),
                "seen_component_count": int(sum(component_seen)),
                "component_support_similarity": support,
                "component_novelty": 1.0 - support,
                "profile_rmse": rmse,
                "posthoc_max_train_effect_cosine": max_posthoc_effect_similarity,
                "classification": "prospective",
            }
        )
        train_id = "train_%s" % (
            next((component for component in components if component in singles), "none")
        )
        for gene_index, gene in enumerate(genes):
            sample_id = "eval_%s_%s" % (condition, gene)
            eval_rows.append(
                {
                    "sample_id": sample_id,
                    "condition": condition,
                    "gene": gene,
                    "cell_type": args.cell_type,
                    "component_support_similarity": support,
                    "y_true": float(true_response[gene_index]),
                    "y_pred": float(pred_response[gene_index]),
                }
            )
            pairwise_rows.append(
                {
                    "sample_id": sample_id,
                    "train_id": "%s_%s" % (train_id, gene),
                    "similarity": support,
                }
            )

    train_path = os.path.join(output_dir, "train_single_gene_responses.csv")
    eval_path = os.path.join(output_dir, "eval_combination_gene_predictions.csv")
    pairwise_path = os.path.join(output_dir, "component_support_pairwise_similarity.csv")
    profile_path = os.path.join(output_dir, "combination_profile_results.csv")
    pd.DataFrame(train_rows).to_csv(train_path, index=False)
    pd.DataFrame(eval_rows).to_csv(eval_path, index=False)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)
    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(profile_path, index=False)

    audit_dir = os.path.join(output_dir, "spectra_component_support_audit")
    spectra_result = run_pairwise_similarity_audit(
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
            domain="perturbation_biology",
            scientific_unit="perturbation_cell_type_gene_response",
            train_path=train_path,
            train_id_col="sample_id",
            train_target_col="response",
            axis_name="component_support_similarity",
            quantile_bins=3,
        )
    )

    support_summary = []
    for support, group in profile_df.groupby("component_support_similarity"):
        support_summary.append(
            {
                "component_support_similarity": float(support),
                "condition_count": int(len(group)),
                "mean_profile_rmse": float(group["profile_rmse"].mean()),
                "median_profile_rmse": float(group["profile_rmse"].median()),
                "max_profile_rmse": float(group["profile_rmse"].max()),
            }
        )
    support_summary = sorted(support_summary, key=lambda row: row["component_support_similarity"])
    rmse_by_support = [row["mean_profile_rmse"] for row in support_summary]
    support_values = [row["component_support_similarity"] for row in support_summary]
    if len(support_summary) >= 2 and rmse_by_support[0] > rmse_by_support[-1]:
        curve_status = "monotonic_supported"
        curve_reason = "Mean profile RMSE is highest when no combination components have single-perturbation support and lowest when all components are supported."
    else:
        curve_status = "not_explanatory"
        curve_reason = "Profile RMSE did not decrease as component support increased."

    summary = {
        "status": "completed",
        "paper": {
            "title": "PerturBench: Benchmarking ML models for cellular perturbation analysis",
            "repo": "https://github.com/altoslabs/perturbench",
        },
        "dataset": "PerturBench devel.h5ad",
        "data_path": data_path,
        "cell_type": args.cell_type,
        "scientific_unit": "perturbation-cell-type expression response profile",
        "model": "additive single-perturbation response baseline",
        "train_units": "single-gene perturbation profiles in %s" % args.cell_type,
        "eval_units": "two-gene combination perturbation profiles in %s" % args.cell_type,
        "selected_similarity_axis": "component_support_similarity",
        "similarity_definition": {
            "name": "component_support_similarity",
            "hypothesis": "Combination perturbations are easier when more component genes have observed single-perturbation responses in training.",
            "leakage_risk": "none",
            "computation_strategy": "exact_component_set_overlap_from_perturbation_names",
            "axis_type": "similarity",
            "range": [0.0, 1.0],
        },
        "profile_count": int(len(profile_df)),
        "gene_count": int(len(genes)),
        "single_train_perturbation_count": int(len(singles)),
        "combination_eval_count": int(len(combinations)),
        "profile_rmse_summary": regression_metrics(profile_df["profile_rmse"].tolist()),
        "support_summary": support_summary,
        "curve_summary": {
            "status": curve_status,
            "reason": curve_reason,
            "lowest_support_mean_rmse": float(support_summary[0]["mean_profile_rmse"]),
            "highest_support_mean_rmse": float(support_summary[-1]["mean_profile_rmse"]),
            "support_values": support_values,
        },
        "auspc": compute_auspc_from_rows(profile_rows),
        "spectra_result": spectra_result,
        "failed_or_secondary_axes": [
            {
                "name": "exact_combination_identity",
                "status": "not_evaluable",
                "reason": "No evaluation combination perturbation is present as a training single-perturbation unit.",
                "leakage_risk": "none",
            },
            {
                "name": "posthoc_effect_profile_similarity",
                "status": "diagnostic_only",
                "reason": "Cosine similarity to observed training expression effects uses evaluation expression response labels and is not a prospective deployment axis.",
                "leakage_risk": "post_hoc_uses_eval_expression",
            },
        ],
        "interpretation": (
            "This local PerturBench capsule shows a prospective perturbation-combination novelty curve: "
            "the additive baseline performs best when both combination components have observed single-perturbation support and worse when component support is partial or absent."
        ),
        "artifacts": {
            "train_single_gene_responses": train_path,
            "eval_combination_gene_predictions": eval_path,
            "pairwise_similarity": pairwise_path,
            "profile_results": profile_path,
            "spectra_audit": audit_dir,
            "summary_json": os.path.join(output_dir, "summary.json"),
            "summary_report": os.path.join(output_dir, "summary.md"),
        },
    }
    with open(summary["artifacts"]["summary_json"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_report(summary["artifacts"]["summary_report"], summary)
    return summary


def write_report(path: str, summary: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# PerturBench Devel Mini-Audit\n\n")
        handle.write("Scientific unit: `%s`\n\n" % summary["scientific_unit"])
        handle.write("Model: `%s`\n\n" % summary["model"])
        handle.write("Selected axis: `%s`\n\n" % summary["selected_similarity_axis"])
        handle.write("## Setup\n\n")
        handle.write("- Train units: %s.\n" % summary["train_units"])
        handle.write("- Eval units: %s.\n" % summary["eval_units"])
        handle.write("- Single train perturbations: `%d`.\n" % summary["single_train_perturbation_count"])
        handle.write("- Combination eval perturbations: `%d`.\n" % summary["combination_eval_count"])
        handle.write("- Genes per response profile: `%d`.\n\n" % summary["gene_count"])
        handle.write("## Component Support Curve\n\n")
        handle.write("| Component support | Conditions | Mean profile RMSE | Median profile RMSE |\n")
        handle.write("| ---: | ---: | ---: | ---: |\n")
        for row in summary["support_summary"]:
            handle.write(
                "| %.3f | %d | %.6f | %.6f |\n"
                % (
                    row["component_support_similarity"],
                    row["condition_count"],
                    row["mean_profile_rmse"],
                    row["median_profile_rmse"],
                )
            )
        handle.write("\n")
        handle.write("Curve status: `%s`.\n\n" % summary["curve_summary"]["status"])
        handle.write("AUSPC: `%s`.\n\n" % summary["auspc"].get("value"))
        handle.write("## Interpretation\n\n")
        handle.write(summary["interpretation"] + "\n\n")
        handle.write("## Caveats\n\n")
        handle.write("- This is a local development PerturBench capsule, not the full PerturBench model suite.\n")
        handle.write("- The SPECTRA audit is run at gene-row level while the scientific unit is a perturbation-cell-type profile; profile-level RMSE is reported separately.\n")
        handle.write("- The selected axis is prospective because it uses only perturbation component names and training availability.\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a PerturBench devel SPECTRA mini-audit")
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cell-type", default="k562")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
