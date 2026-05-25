# Iterative Agent Trial Results

Date: 2026-05-12

This trial ran the updated vanilla versus `/spectra` comparison on an anonymous
sequence-fitness task derived from the Martin 2018 NABench assay. Both agents
received only train rows, held-out labels/predictions, and metadata. Neither
agent was given the NABench paper or prior SPECTRA rerun outputs.

Input/output root:

`/ewsc/yektefai/spectra_agent_trials/iterative_martin_20260512`

## Conditions

Vanilla condition:

- Used normal scientific model-evaluation workflow.
- Was prohibited from using SPECTRA docs, registries, runners, or previous
  SPECTRA outputs.
- Output report:
  `/ewsc/yektefai/spectra_agent_trials/iterative_martin_20260512/vanilla/report.json`

`/spectra` condition:

- Used the current SPECTRA protocol.
- Was instructed to treat similarity definitions as hypotheses, validate
  overlap, score each curve, report failed axes, and classify leakage/post-hoc
  risks.
- Output report:
  `/ewsc/yektefai/spectra_agent_trials/iterative_martin_20260512/spectra/report.json`

## Shared Finding

Both agents found the aggregate model behavior:

- Held-out RMSE: about `0.0763`
- Held-out R2: about `-0.008`
- Prediction variance: effectively zero

This means the supplied model behaves like a near-constant predictor. That is a
major caveat for interpreting any novelty-dependent curve, because error changes
can reflect changes in the distribution of `y_true` rather than a model-specific
failure mode.

## Vanilla Behavior

The vanilla agent produced a competent standard evaluation. It identified the
scientific unit, checked exact sequence/mutant overlap, computed aggregate
metrics, and considered several novelty axes:

- exact sequence or mutant overlap,
- mutation center seen/unseen,
- nearest mutation-center distance,
- nearest Hamming distance,
- local 21-mer context seen in train,
- absolute mutation-position quartiles.

Its main conclusion was conservative: performance degradation under novelty was
not convincingly evaluable. Several axes collapsed:

- no exact train/eval sequence overlap,
- no eval rows with seen mutation centers,
- no local 21-mer contexts seen in train,
- all eval rows had nearest Hamming distance 2.

The nearest-center-distance bins were weak/non-monotonic rather than a clean
degradation curve.

## `/spectra` Behavior

The `/spectra` agent produced a more structured iterative audit. It generated
pairwise similarity files and per-axis SPECTRA artifacts for:

- `mutation_position_support_similarity`,
- `sequence_identity_similarity`,
- `mutation_centered_window_identity_similarity`,
- `mutation_depth_support_similarity`,
- `local_gc_support_similarity`,
- `fitness_support_similarity`,
- an invalid absolute-error axis.

It scored the initial position-support axis as `not_explanatory`, scored
whole-sequence identity and mutation depth as `not_evaluable`, classified
fitness support as post-hoc explanatory, and rejected absolute error as an
invalid novelty axis.

The selected prospective axis was:

`mutation_centered_window_identity_similarity`

For that axis:

- All-eval RMSE: `0.076314`
- Lowest-overlap subset RMSE: `0.089943`
- Curve status: `monotonic_supported`
- AUSPC/equivalent negative RMSE area: `-0.093598`

## Score

Scores use `experiments/iterative_agent_behavior/rubric.json`.

| Dimension | Vanilla | `/spectra` |
| --- | ---: | ---: |
| scientific_unit | 2 | 2 |
| initial_similarity_hypothesis | 2 | 2 |
| property_graph_specificity | 1 | 2 |
| overlap_validation | 2 | 2 |
| performance_curve | 2 | 2 |
| curve_scoring | 2 | 2 |
| failed_axis_reporting | 2 | 2 |
| adaptive_next_axis | 0 | 2 |
| leakage_risk_classification | 1 | 2 |
| selected_axis_defensibility | 1 | 2 |
| metric_reporting | 0 | 2 |
| audit_artifact_quality | 1 | 2 |
| Total | 17 / 24 | 24 / 24 |

## Interpretation

This trial supports the narrower, defensible SPECTRA claim:

> `/spectra` changes an agent's behavior from a static generalization analysis
> into an iterative similarity-hypothesis audit with failed-axis reporting,
> leakage classification, AUSPC, and reusable audit artifacts.

It does not support the stronger claim that vanilla agents cannot identify the
core generalization issue. The vanilla agent found the main aggregate failure and
several failed novelty axes. The added value was protocol completeness,
iteration, artifact quality, and prospective/post-hoc classification.

The next trial should repeat this comparison on at least two more domains:

- BOOM-style molecular predictions, where chemical similarity should give a
  stronger prospective curve.
- Perturbation or regulatory sequence data, where the similarity definition is
  less obvious and the registry/MCP guidance should matter more.
