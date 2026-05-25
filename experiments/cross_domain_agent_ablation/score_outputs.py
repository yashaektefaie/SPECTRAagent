"""Score the fresh cross-domain agent-ablation reports against the fixed rubric."""

import csv
import json
import os
from typing import Dict, List


RUN_ROOT = "/ewsc/yektefai/spectra_agent_ablation_20260513"
OUTPUT_ROOT = os.path.join(RUN_ROOT, "scoring")

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
        "vanilla": {
            "report": f"{RUN_ROOT}/agent_outputs/molecules_vanilla/report.json",
            "scores": {
                "scientific_unit": 2,
                "aggregate_error": 2,
                "axis_discovery": 2,
                "pairwise_similarity": 2,
                "overlap_validation": 1,
                "performance_curve": 2,
                "auspc_or_summary": 1,
                "failed_axis_reporting": 1,
                "adaptive_iteration": 0,
                "leakage_classification": 0,
                "domain_interpretation": 2,
                "artifact_quality": 2,
            },
            "rationale": {
                "scientific_unit": "Correctly identified molecules/SMILES with density target.",
                "aggregate_error": "Computed MAE, RMSE, R2, rank correlations, bias, and baselines.",
                "axis_discovery": "Constructed SMILES q-gram and nearest feature-distance novelty axes.",
                "pairwise_similarity": "Computed nearest-train similarity/distance axes from visible train/eval data, though not a long-form pairwise graph.",
                "overlap_validation": "Reported novelty bins and nearest-train summaries but did not explicitly validate threshold monotonicity.",
                "performance_curve": "Reported performance by similarity and novelty bins.",
                "auspc_or_summary": "Provided correlations and binned deltas but no AUSPC/area summary.",
                "failed_axis_reporting": "Mentioned unavailable chemistry-toolkit/scaffold analysis but did not systematically report failed axes.",
                "adaptive_iteration": "Did not use a failed curve to motivate a next similarity hypothesis.",
                "leakage_classification": "Did not classify prospective versus post-hoc axes.",
                "domain_interpretation": "Interpreted density underprediction, target shift, and molecular novelty coherently.",
                "artifact_quality": "Produced report, code, novelty metrics, performance table, and shift summary.",
            },
            "key_findings": [
                "MAE 0.1551, RMSE 0.1929, R2 0.1548.",
                "MAE increased from 0.0914 in least-novel quantile to 0.2270 in most-novel quantile.",
                "Mean prediction bias was -0.1144, indicating underprediction of shifted high-density molecules.",
            ],
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
            "rationale": {
                "scientific_unit": "Correctly identified molecule/SMILES rows and density target.",
                "aggregate_error": "Computed overall error and spectral bin errors.",
                "axis_discovery": "Tried q-gram, composition, descriptor, post-hoc target support, and documented unavailable Morgan/scaffold axes.",
                "pairwise_similarity": "Wrote retained top-k train-eval pairwise similarity CSVs for multiple axes.",
                "overlap_validation": "Wrote overlap validation tables showing decreasing measured overlap.",
                "performance_curve": "Reported per-axis performance curves.",
                "auspc_or_summary": "Computed area-style spectral summaries and novelty-binned deltas.",
                "failed_axis_reporting": "Explicitly marked Morgan Tanimoto and Bemis-Murcko scaffold as not evaluable.",
                "adaptive_iteration": "Selected a leakage-aware computable axis after evaluating unavailable and post-hoc alternatives.",
                "leakage_classification": "Classified structure axes versus post-hoc target support.",
                "domain_interpretation": "Connected molecular composition/descriptor novelty to density failure.",
                "artifact_quality": "Produced report, audit card, pairwise files, curves, overlap tables, and pretty JSON.",
            },
            "key_findings": [
                "Selected composition_weighted_jaccard as the prospective axis.",
                "MAE increased from 0.0493 in lowest-novelty bin to 0.3054 in highest-novelty bin.",
                "Morgan/scaffold axes were correctly marked not evaluable because RDKit was unavailable.",
            ],
        },
    },
    "sequence_fitness": {
        "vanilla": {
            "report": f"{RUN_ROOT}/agent_outputs/sequence_fitness_vanilla/report.json",
            "scores": {
                "scientific_unit": 2,
                "aggregate_error": 2,
                "axis_discovery": 2,
                "pairwise_similarity": 2,
                "overlap_validation": 1,
                "performance_curve": 2,
                "auspc_or_summary": 1,
                "failed_axis_reporting": 1,
                "adaptive_iteration": 0,
                "leakage_classification": 0,
                "domain_interpretation": 2,
                "artifact_quality": 2,
            },
            "rationale": {
                "scientific_unit": "Correctly identified single mutated nucleotide sequence.",
                "aggregate_error": "Computed MAE, RMSE, negative R2, baselines, and bootstrap CIs.",
                "axis_discovery": "Identified contiguous mutation-position novelty and Hamming saturation.",
                "pairwise_similarity": "Computed nearest trained mutation-position distance and min Hamming distance from visible data.",
                "overlap_validation": "Showed zero position overlap and contiguous interval but did not validate a threshold curve.",
                "performance_curve": "Reported performance by distance bins, mutation position, base, and substitution.",
                "auspc_or_summary": "Provided grouped quantitative summaries but no AUSPC/area.",
                "failed_axis_reporting": "Noted full-sequence Hamming is saturated but not as a formal failed-axis report.",
                "adaptive_iteration": "No explicit iterative next-axis behavior.",
                "leakage_classification": "No prospective/post-hoc leakage classification.",
                "domain_interpretation": "Correctly concluded constant-predictor failure on a held-out mutation-position block.",
                "artifact_quality": "Produced report, analysis script, aggregate metrics, split summary, and grouped performance tables.",
            },
            "key_findings": [
                "All 359 held-out predictions were identical.",
                "RMSE 0.0763 and R2 -0.0081.",
                "Eval mutations were a contiguous held-out position block 241-360 with zero train position overlap.",
            ],
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
            "rationale": {
                "scientific_unit": "Correctly identified single mutated 600 nt sequence.",
                "aggregate_error": "Reported aggregate MAE, RMSE, R2, prediction/truth distribution.",
                "axis_discovery": "Defined five prospective sequence axes.",
                "pairwise_similarity": "Wrote pairwise train-eval similarity CSVs for each axis.",
                "overlap_validation": "Validated overlap decrease for supported axes and identified saturated axes.",
                "performance_curve": "Wrote performance curves for all axes.",
                "auspc_or_summary": "Computed negative-RMSE AUSPC where evaluable.",
                "failed_axis_reporting": "Reported non-explanatory and not-evaluable axes.",
                "adaptive_iteration": "Evaluated alternatives and reported failed axes, but the next-axis sequence was mostly preplanned rather than clearly adaptive.",
                "leakage_classification": "Classified axes as prospective/low leakage.",
                "domain_interpretation": "Correctly concluded global constant-prediction failure rather than novelty-specific degradation.",
                "artifact_quality": "Produced report, audit card, pairwise files, curves, SVGs, and script.",
            },
            "key_findings": [
                "Selected mutation_position_exponential_30bp as the direct prospective axis.",
                "Overlap validation was valid, but the curve was non-explanatory.",
                "Dominant failure was globally poor constant prediction, RMSE 0.0763, R2 -0.0081.",
            ],
        },
    },
    "perturbation_biology": {
        "vanilla": {
            "report": f"{RUN_ROOT}/agent_outputs/perturbation_biology_vanilla/report.json",
            "scores": {
                "scientific_unit": 2,
                "aggregate_error": 2,
                "axis_discovery": 2,
                "pairwise_similarity": 2,
                "overlap_validation": 1,
                "performance_curve": 2,
                "auspc_or_summary": 1,
                "failed_axis_reporting": 1,
                "adaptive_iteration": 0,
                "leakage_classification": 1,
                "domain_interpretation": 2,
                "artifact_quality": 2,
            },
            "rationale": {
                "scientific_unit": "Correctly identified perturbation-condition expression response profile.",
                "aggregate_error": "Computed row-level RMSE, MAE, R2, correlations, detection metrics, and baselines.",
                "axis_discovery": "Identified component-support and visible-component-response magnitude analyses.",
                "pairwise_similarity": "Computed component support from visible train conditions, equivalent to a train-eval support axis.",
                "overlap_validation": "Stratified by support but did not explicitly validate threshold monotonicity.",
                "performance_curve": "Reported RMSE/MAE by component-support group.",
                "auspc_or_summary": "Provided grouped quantitative summary but no AUSPC.",
                "failed_axis_reporting": "Mentioned post-hoc profile similarity limitation but did not systematically report failed axes.",
                "adaptive_iteration": "No explicit iterative next-axis behavior.",
                "leakage_classification": "Identified post-hoc profile similarity as descriptive, but did not classify all axes.",
                "domain_interpretation": "Correctly interpreted component-reuse failure and limited extrapolation.",
                "artifact_quality": "Produced report, analysis script, condition metrics, component support metrics, and magnitude bins.",
            },
            "key_findings": [
                "Aggregate RMSE 0.08594, MAE 0.00944, R2 0.233.",
                "Predictions exactly matched a visible-component sum baseline.",
                "RMSE worsened from 0.0201 with two components supported to 0.1070 with no components supported.",
            ],
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
            "rationale": {
                "scientific_unit": "Correctly identified perturbation-condition expression response profile.",
                "aggregate_error": "Reported aggregate RMSE and curve metrics.",
                "axis_discovery": "Defined component support, gene/cell support, and post-hoc response-profile similarity.",
                "pairwise_similarity": "Wrote pairwise similarity artifacts for component support and post-hoc response-profile axes.",
                "overlap_validation": "Validated decreasing overlap for component support.",
                "performance_curve": "Reported performance-overlap curves for each axis.",
                "auspc_or_summary": "Computed negative-RMSE AUSPC for component support and post-hoc axis.",
                "failed_axis_reporting": "Reported gene/cell support as not evaluable and post-hoc profile similarity as secondary/leaky.",
                "adaptive_iteration": "Selected the prospective component-support axis after evaluating saturated and post-hoc alternatives.",
                "leakage_classification": "Explicitly classified low-leakage and high-leakage axes.",
                "domain_interpretation": "Correctly interpreted coarse monotonic component-novelty degradation.",
                "artifact_quality": "Produced report, audit card, Markdown report, pairwise files, curves, split stats, and hypothesis scores.",
            },
            "key_findings": [
                "Selected component_support_fraction as a low-leakage prospective axis.",
                "RMSE rose from 0.08594 overall to 0.10699 at support 0.",
                "AUSPC was -0.09876.",
            ],
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
        "score_scale": {
            "0": "missing/invalid/misleading",
            "1": "partial/qualitative/weak",
            "2": "quantitative/valid/reproducible/useful",
        },
        "domains": {},
        "interpretation": {
            "main_result": "Vanilla agents found the main generalization failures in all executable domains. /spectra mainly improved standardized spectral artifacts, AUSPC, failed-axis reporting, leakage classification, and audit-card completeness.",
            "unsupported_claim": "The experiment does not support the claim that vanilla agents cannot discover broad generalization failures.",
            "supported_claim": "The experiment supports the claim that /spectra makes agent-executed generalization audits more complete, reproducible, and leakage-aware.",
        },
        "excluded_domains": {
            "regulatory_dna": "DART-Eval was not run because task data and genome references are Synapse-hosted and unavailable locally."
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
                "condition_from_report": report.get("condition"),
                "scores": scores,
                "total": score_total,
                "max_total": 2 * len(DIMENSIONS),
                "rationale": payload["rationale"],
                "key_findings": payload["key_findings"],
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

    for domain, condition_map in scored["domains"].items():
        if "vanilla" in condition_map and "spectra" in condition_map:
            condition_map["delta_spectra_minus_vanilla"] = (
                condition_map["spectra"]["total"] - condition_map["vanilla"]["total"]
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
        handle.write("# Cross-Domain Agent Ablation Scores\n\n")
        handle.write("| Domain | Vanilla | /spectra | Delta |\n")
        handle.write("| --- | ---: | ---: | ---: |\n")
        for domain, condition_map in scored["domains"].items():
            vanilla_total = condition_map["vanilla"]["total"]
            spectra_total = condition_map["spectra"]["total"]
            delta = condition_map["delta_spectra_minus_vanilla"]
            handle.write("| %s | %d / 24 | %d / 24 | %+d |\n" % (domain, vanilla_total, spectra_total, delta))
        handle.write("\n## Interpretation\n\n")
        handle.write(scored["interpretation"]["main_result"] + "\n\n")
        handle.write("Unsupported claim: %s\n\n" % scored["interpretation"]["unsupported_claim"])
        handle.write("Supported claim: %s\n" % scored["interpretation"]["supported_claim"])

    print(json.dumps({"scores_json": scores_path, "scores_csv": csv_path, "summary_md": markdown_path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
