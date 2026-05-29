# Generalizability Analysis Procedure

Version: 0.5.1

## Purpose

SPECTRA evaluates whether model performance changes as prospective train-test
or pretraining-test similarity decreases. The primary product is a spectral
performance curve (SPC), split or similarity-bin statistics, baseline behavior,
target-model behavior, and a validity decision.

When the user supplies a generalization question and a model paper/reference,
call `start_spectra_audit_session` first. The returned contract defines these
roles:

- SPECTRA Distiller
- SPECTRA Investigator
- SPECTRA Dataset Scout
- SPECTRA Dataset Fetcher
- SPECTRA Auditor
- final SPECTRA Synthesis Distiller

Clients with subagent support should spawn the roles directed by the contract.
Single-agent clients should execute the same role passes sequentially and
persist the same artifacts. Do not execute roles implicitly just because a
session was prepared.

## Role Contract

SPECTRA Distiller:
Turn user generalizability questions into SPECTRA analyses. Identify or
validate a prospective similarity axis that can yield a meaningful SPC:
performance as train-test or pretraining-test similarity decreases. Use papers,
claims, metadata, schema, and domain knowledge to propose candidate axes.
Interpret Investigator and Auditor results, distinguish valid from exploratory
axes, avoid overclaiming, and return a clear answer about where the model
generalizes or fails.

SPECTRA Investigator:
Execute the SPECTRA protocol for a given model, dataset, task, metric, and
similarity axis. Compute prospective similarities without using target-model
errors. Construct at least three nontrivial split levels or similarity bins when
feasible. Verify train-test or pretraining-test similarity decreases. Validate
the axis with a simple fixed baseline when labels exist. Only then evaluate the
model of interest. Return the SPC, split statistics, baseline results, model
results, commands, blockers, and validity self-assessment.

SPECTRA Dataset Scout:
Find datasets suitable for SPECTRA. Prioritize datasets with labels, metadata,
prospective similarity features, enough examples for multiple split levels,
clear access, and low leakage risk. Query the portable dataset catalog before
open-ended search. Return candidate datasets, access routes, available fields,
possible similarity axes, risks, and suitability rankings.

SPECTRA Dataset Fetcher:
Retrieve and package datasets for SPECTRA. Load data, inspect schema, retain
inputs, labels, metadata, and prospective similarity features, handle duplicates
and missingness, and create SPECTRA-ready artifacts. For very large pretraining
datasets, use scalable filtering or approximate retrieval to estimate
pretraining proximity rather than exhaustive comparison.

SPECTRA Auditor:
Check whether the SPC supports the claim. Look for target-error leakage,
test-label leakage, tiny splits, non-decreasing similarity, unstable baselines,
confounding, metric-direction errors, and post-hoc axis selection. Mark analyses
as valid, weak, invalid, or exploratory only.

## Required Workflow

1. Identify the scientific unit, dataset, prediction target, model, task, and
   metric.
2. Extract the paper's stated and implicit generalization claims when a paper is
   provided.
3. Choose prospective similarity axes from fields available before target-model
   evaluation.
4. If the dataset is missing or unsuitable, route to Dataset Scout or Dataset
   Fetcher before model evaluation.
5. Decide whether benchmark mode or audit fallback mode applies.
6. Plan the SPC with at least three split levels or similarity bins when
   feasible.
7. Compute train-test or pretraining-test similarity for the planned levels.
8. Verify that measured similarity decreases across levels.
9. If labels exist, run a simple fixed baseline across the same levels.
10. Evaluate the target model only after the split contract is valid or
    explicitly labeled exploratory.
11. Send the SPC, split statistics, baseline results, and model results to the
    Auditor.
12. Return a final answer that states whether the SPC is valid, weak, invalid,
    or exploratory and what claim boundary follows.

## Validity Rules

An SPC can be `valid`, `weak`, `invalid`, or `exploratory`.

Mark it invalid or exploratory if:

- the axis uses target-model errors, prediction/reference errors, or target
  confidence derived from the evaluated model;
- held-out labels define split membership;
- train-test or pretraining-test similarity does not decrease;
- split levels are tiny or degenerate;
- a fixed baseline is omitted despite available labels and no justification;
- the chosen axis was selected post-hoc because it correlated with model error;
- confounders explain the curve better than the declared axis.

Failed or non-monotonic prospective axes are findings. Report them directly and
either try the next prospective axis or state what data/features are missing.
Do not replace a failed prospective axis with a circular post-hoc error metric.

## Required Artifacts

- `spectra_analysis_plan.json`
- `candidate_similarity_axes.json`
- `axis_leakage_classification.json`
- `dataset_manifest.json` when data are fetched or packaged
- `schema_report.json` when data are fetched or packaged
- `split_assignments/`
- `split_statistics.json`
- `similarity_progression.csv`
- `baseline_results.csv` or `baseline_omission_reason.md`
- `model_results.csv`
- `spectral_performance_curve.csv`
- `validity_self_assessment.json`
- `audit_report.md`
- `validity_decision.json`
- `risk_register.json`
- `required_fixes.json`
- `claim_boundary.json`
- `overclaim_guardrails.md`
- `commands_run.json`

## Recommended Tool Sequence

1. `start_spectra_audit_session`
2. `get_procedure`
3. `suggest_dataset_catalog_entries`
4. `start_generalizability_analysis`
5. `select_spectra_execution_mode`
6. `plan_spectral_performance_curve`
7. `suggest_similarity_definitions`
8. `suggest_similarity_computation_strategies`
9. `plan_similarity_computation`
10. `run_spectra_audit` or an equivalent split-based evaluator
11. `score_similarity_hypothesis_curve`
12. `validate_split_stats`
13. Auditor review through the role contract
14. final synthesis only after the validity decision is clear

## Reporting

The final report should state:

- the model, dataset, task, metric, and scientific unit;
- the similarity axis and why it is prospective;
- the split or similarity-bin levels;
- measured similarity progression;
- baseline behavior;
- target-model behavior;
- SPC status: valid, weak, invalid, or exploratory;
- leakage, split-size, confounding, and metric-direction risks;
- the exact claim supported by the SPC.
