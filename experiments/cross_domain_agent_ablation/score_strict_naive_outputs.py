"""Score strict-naive vanilla reports against the same 24-point rubric."""

import csv
import json
import os
from typing import Dict, List


RUN_ROOT = "/ewsc/yektefai/spectra_agent_ablation_20260513"
OUTPUT_ROOT = os.path.join(RUN_ROOT, "scoring_strict_naive")

DIMENSIONS = [
    "scientific_unit",
    "aggregate_error",
    "axis_discovery",
    "pairwise_similarity",
    "overlap_validation",
    "performance_curve",
    "auspc_or_summary",
    "failed_axis_reporting",
    "adaptive_iteration",
    "leakage_classification",
    "domain_interpretation",
    "artifact_quality",
]


SCORES: Dict[str, Dict[str, Dict[str, object]]] = {
    "molecules": {
        "strict_naive": {
            "report": f"{RUN_ROOT}/agent_outputs_strict_naive/molecules_naive/report.json",
            "scores": {
                "scientific_unit": 2,
                "aggregate_error": 2,
                "axis_discovery": 2,
                "pairwise_similarity": 2,
                "overlap_validation": 2,
                "performance_curve": 2,
                "auspc_or_summary": 1,
                "failed_axis_reporting": 1,
                "adaptive_iteration": 1,
                "leakage_classification": 2,
                "domain_interpretation": 2,
                "artifact_quality": 2,
            },
            "key_findings": [
                "Eval MAE 0.155, RMSE 0.193, R2 0.155.",
                "SMILES 4-gram novelty curve was monotonic: MAE rose from 0.084 to 0.219.",
                "Target-support shift was the strongest failure: 57.1% of eval molecules above train target max, MAE 0.228.",
            ],
            "rationale": "Even the strict-naive prompt induced a strong molecular novelty and target-support audit; the schema made the natural axes visible.",
        },
        "spectra": {
            "report": f"{RUN_ROOT}/agent_outputs/molecules_spectra/report.json",
            "scores": {
                "scientific_unit": 2,
                "aggregate_error": 2,
                "axis_discovery": 2,
                "pairwise_similarity": 2,
                "overlap_validation": 2,
                "performance_curve": 2,
                "auspc_or_summary": 2,
                "failed_axis_reporting": 2,
                "adaptive_iteration": 2,
                "leakage_classification": 2,
                "domain_interpretation": 2,
                "artifact_quality": 2,
            },
            "key_findings": [
                "Selected composition_weighted_jaccard as prospective axis.",
                "MAE rose from 0.0493 to 0.3054 across selected-axis novelty bins.",
                "Marked Morgan/scaffold axes not evaluable because RDKit was unavailable.",
            ],
            "rationale": "The /spectra report remains more standardized and complete, mainly through AUSPC, failed-axis accounting, and audit-card artifacts.",
        },
    },
    "sequence_fitness": {
        "strict_naive": {
            "report": f"{RUN_ROOT}/agent_outputs_strict_naive/sequence_fitness_naive/report.json",
            "scores": {
                "scientific_unit": 2,
                "aggregate_error": 2,
                "axis_discovery": 2,
                "pairwise_similarity": 2,
                "overlap_validation": 2,
                "performance_curve": 2,
                "auspc_or_summary": 1,
                "failed_axis_reporting": 2,
                "adaptive_iteration": 1,
                "leakage_classification": 2,
                "domain_interpretation": 2,
                "artifact_quality": 2,
            },
            "key_findings": [
                "Held-out predictions had one unique value.",
                "RMSE 0.07631 and R2 -0.00809.",
                "No exact mutant, sequence, or mutation-position overlap; eval was contiguous positions 241-360.",
                "Position-distance curve was non-explanatory; Hamming curve not evaluable.",
            ],
            "rationale": "The strict-naive sequence agent independently produced nearly the SPECTRA interpretation, including failed/non-evaluable axes.",
        },
        "spectra": {
            "report": f"{RUN_ROOT}/agent_outputs/sequence_fitness_spectra/report.json",
            "scores": {
                "scientific_unit": 2,
                "aggregate_error": 2,
                "axis_discovery": 2,
                "pairwise_similarity": 2,
                "overlap_validation": 2,
                "performance_curve": 2,
                "auspc_or_summary": 2,
                "failed_axis_reporting": 2,
                "adaptive_iteration": 1,
                "leakage_classification": 2,
                "domain_interpretation": 2,
                "artifact_quality": 2,
            },
            "key_findings": [
                "Selected mutation_position_exponential_30bp.",
                "Overlap validation was valid, but the curve was non-explanatory.",
                "Dominant failure was globally poor constant prediction.",
            ],
            "rationale": "The /spectra sequence agent adds reusable pairwise/curve/AUSPC/audit-card artifacts, but the strict-naive agent found the scientific conclusion.",
        },
    },
    "perturbation_biology": {
        "strict_naive": {
            "report": f"{RUN_ROOT}/agent_outputs_strict_naive/perturbation_biology_naive/report.json",
            "scores": {
                "scientific_unit": 2,
                "aggregate_error": 2,
                "axis_discovery": 2,
                "pairwise_similarity": 2,
                "overlap_validation": 2,
                "performance_curve": 2,
                "auspc_or_summary": 1,
                "failed_axis_reporting": 2,
                "adaptive_iteration": 1,
                "leakage_classification": 2,
                "domain_interpretation": 2,
                "artifact_quality": 2,
            },
            "key_findings": [
                "Exact eval-combination overlap was 0/15.",
                "All predictions matched a visible component-sum baseline.",
                "RMSE improved from 0.1070 with zero seen components to 0.0201 with two seen components.",
            ],
            "rationale": "The strict-naive perturbation agent found component support and leakage-aware/post-hoc distinctions from the visible schema alone.",
        },
        "spectra": {
            "report": f"{RUN_ROOT}/agent_outputs/perturbation_biology_spectra/report.json",
            "scores": {
                "scientific_unit": 2,
                "aggregate_error": 2,
                "axis_discovery": 2,
                "pairwise_similarity": 2,
                "overlap_validation": 2,
                "performance_curve": 2,
                "auspc_or_summary": 2,
                "failed_axis_reporting": 2,
                "adaptive_iteration": 2,
                "leakage_classification": 2,
                "domain_interpretation": 2,
                "artifact_quality": 2,
            },
            "key_findings": [
                "Selected component_support_fraction.",
                "AUSPC -0.09876.",
                "Gene/cell support was not evaluable; response-profile similarity was post-hoc/leaky.",
            ],
            "rationale": "The /spectra report adds standardized artifacts and AUSPC, but the strict-naive scientific conclusion is similar.",
        },
    },
}


def load_report(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def total_score(score_map: Dict[str, int]) -> int:
    return sum(int(score_map[dimension]) for dimension in DIMENSIONS)


def main() -> int:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    rows: List[Dict[str, object]] = []
    scored = {
        "run_root": RUN_ROOT,
        "rubric_dimensions": DIMENSIONS,
        "domains": {},
        "interpretation": {
            "main_result": "With the strict-naive prompt, vanilla agents still found the main generalization failures in all three domains and often produced distance/overlap analyses without being asked.",
            "supported_claim": "/spectra remains useful as a standardization layer for AUSPC, audit cards, consistent artifacts, and explicit protocol compliance.",
            "weakened_claim": "These tasks do not show that registries or /spectra are necessary for agents to discover obvious schema-exposed axes.",
        },
    }
    for domain, condition_map in SCORES.items():
        scored["domains"][domain] = {}
        for condition, payload in condition_map.items():
            report = load_report(str(payload["report"]))
            scores = payload["scores"]
            score_total = total_score(scores)
            scored["domains"][domain][condition] = {
                "report": payload["report"],
                "condition_from_report": report.get("condition") or report.get("task"),
                "scores": scores,
                "total": score_total,
                "max_total": 2 * len(DIMENSIONS),
                "key_findings": payload["key_findings"],
                "rationale": payload["rationale"],
            }
            row = {
                "domain": domain,
                "condition": condition,
                "total": score_total,
                "max_total": 2 * len(DIMENSIONS),
                "report": payload["report"],
            }
            for dimension in DIMENSIONS:
                row[dimension] = scores[dimension]
            rows.append(row)

        scored["domains"][domain]["delta_spectra_minus_strict_naive"] = (
            scored["domains"][domain]["spectra"]["total"]
            - scored["domains"][domain]["strict_naive"]["total"]
        )

    scores_path = os.path.join(OUTPUT_ROOT, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as handle:
        json.dump(scored, handle, indent=2, sort_keys=True)

    csv_path = os.path.join(OUTPUT_ROOT, "scores.csv")
    fieldnames = ["domain", "condition", "total", "max_total"] + DIMENSIONS + ["report"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = os.path.join(OUTPUT_ROOT, "summary.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("# Strict-Naive Cross-Domain Agent Ablation Scores\n\n")
        handle.write("| Domain | Strict-naive vanilla | /spectra | Delta |\n")
        handle.write("| --- | ---: | ---: | ---: |\n")
        for domain, condition_map in scored["domains"].items():
            naive_total = condition_map["strict_naive"]["total"]
            spectra_total = condition_map["spectra"]["total"]
            delta = condition_map["delta_spectra_minus_strict_naive"]
            handle.write("| %s | %d / 24 | %d / 24 | %+d |\n" % (domain, naive_total, spectra_total, delta))
        handle.write("\n## Interpretation\n\n")
        handle.write(scored["interpretation"]["main_result"] + "\n\n")
        handle.write(scored["interpretation"]["supported_claim"] + "\n\n")
        handle.write(scored["interpretation"]["weakened_claim"] + "\n")

    print(json.dumps({"scores_json": scores_path, "scores_csv": csv_path, "summary_md": markdown_path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
