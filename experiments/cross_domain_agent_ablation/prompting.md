# Cross-Domain Agent-Ablation Prompting

This file records the condition-level prompts used for the 2026-05-13 fresh
agent-ablation experiment. Domain-specific paths are injected into these
templates before launching each agent.

## Vanilla Condition

You are an analysis agent in a blinded scientific model-generalizability
experiment. You are given only the agent-visible files listed below. Do not use
paper context, benchmark repository context, grading-only files, hidden
reference files, precomputed SPECTRA summaries, precomputed split labels, or
precomputed train-test similarity. Do not use `/tmp`; write all outputs under
the assigned output directory.

Task: evaluate whether the trained model generalizes from the training data to
the held-out predictions. Choose whatever analyses you think are scientifically
appropriate from the visible files. Run code if needed. Produce a JSON report at
`{output_dir}/report.json` and any supporting artifacts under `{output_dir}`.

The JSON report should include:

- `condition`
- `domain`
- `scientific_unit`
- `analysis_performed`
- `aggregate_performance`
- `generalization_evidence`
- `similarity_or_novelty_analyses`
- `performance_by_group_or_novelty`
- `failure_modes`
- `limitations`
- `recommended_next_steps`
- `artifacts`

## SPECTRA Condition

You are an analysis agent in a blinded scientific model-generalizability
experiment. You are given only the agent-visible files listed below. Do not use
paper context, benchmark repository context, grading-only files, hidden
reference files, precomputed deterministic benchmark summaries, precomputed
split labels, or precomputed train-test similarity. Do not use `/tmp`; write all
outputs under the assigned output directory.

Task: evaluate whether the trained model generalizes using the `/spectra`
workflow. Use the SPECTRA audit loop: identify the scientific unit, propose
candidate train-test similarity axes, compute or approximate pairwise train-eval
similarity from visible data, validate overlap/novelty, report performance as a
function of measured novelty, compute AUSPC or an equivalent spectral summary,
classify leakage risk for each axis, report failed/non-evaluable axes, and write
a reusable audit card.

You may use the local SPECTRA package, CLI, procedure docs, and registries in
this repository. The generic pairwise audit accepts `eval_predictions.csv` plus
a long-form `pairwise_similarity.csv`; use that if appropriate.

Produce a JSON report at `{output_dir}/report.json` and any supporting artifacts
under `{output_dir}`.

The JSON report should include:

- `condition`
- `domain`
- `scientific_unit`
- `candidate_similarity_axes`
- `pairwise_similarity_artifacts`
- `overlap_validation`
- `performance_curves`
- `auspc_or_spectral_summary`
- `failed_or_secondary_axes`
- `leakage_risk_classification`
- `selected_axis`
- `generalization_conclusion`
- `failure_modes`
- `limitations`
- `recommended_next_steps`
- `artifacts`
