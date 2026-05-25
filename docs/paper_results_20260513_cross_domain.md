# Cross-Domain Paper Results Run

Date: 2026-05-13

This document records the follow-up run executed to move the paper beyond a
BOOM-centered result set. All new generated artifacts were written under:

`/ewsc/yektefai/spectra_paper_results_20260513`

The prior BOOM artifacts remain under:

`/ewsc/yektefai/spectra_paper_results_20260512`

## Why the Earlier NABench Result Was Weak

The first NABench runner did not fail because SPECTRA is molecule-specific. It
failed because the candidate sequence axes and the selection rule were too
narrow:

- whole-sequence identity was often saturated for single-mutant assays;
- mutation-position support was useful for Gregory but not sufficient for all
  assays;
- the first selection rule preferred effect size even when the strongest axis
  used evaluation labels;
- the runner did not include a local sequence-context axis around the mutation.

The revised runner adds `mutation_centered_window_identity_similarity`, marks
fitness-based axes as post-hoc, and prefers leakage-free supported axes over
post-hoc axes when support status is tied.

## Executed Experiments

1. Prospective NABench rerun on three assays:
   - `Gregory_2018_mRNA.csv`
   - `Martin_2018_myc_enhancer.csv`
   - `Pitt_2010_ribozyme.csv`
2. Local PerturBench development capsule using bundled `devel.h5ad`.
3. DART-Eval data-access inspection.
4. Cross-domain result aggregation.

## Deterministic Cross-Domain Results

| Domain | Task | Model | Selected axis | Axis type | Reference performance | High-novelty performance | AUSPC | Interpretation |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Molecules | BOOM 10k density | RF Morgan fingerprints | max Morgan Tanimoto to train | prospective | 0.0443 ID RMSE; 0.2296 OOD RMSE | 0.2534 lowest-overlap OOD RMSE | -0.2410 | Chemical OOD error increases as max train Tanimoto decreases. |
| Sequence fitness | NABench Gregory 2018 mRNA | position-aware ridge | mutation-position support | prospective | 1.1582 contiguous RMSE | 1.5723 lowest-overlap RMSE | -1.2536 | Contiguous holdout error concentrates farther from trained mutation positions. |
| Sequence fitness | NABench Martin 2018 MYC enhancer | position-aware ridge | mutation-centered window identity | prospective | 0.0763 contiguous RMSE | 0.0899 lowest-overlap RMSE | -0.0936 | Local sequence-context novelty explains a harder subset. |
| Sequence fitness | NABench Pitt 2010 ribozyme | position-aware ridge | position-fitness composite | post-hoc diagnostic | 0.0137 contiguous RMSE | 0.0165 lowest-overlap RMSE | -0.0214 | No tested prospective axis was sufficient; the supported curve uses labels. |
| Perturbation biology | PerturBench devel combination prediction | additive single-perturbation response baseline | component support | prospective | 0.0199 profile RMSE when both components are supported | 0.0978 profile RMSE when no components are supported | -0.0919 | Combination prediction is much worse when component perturbations are unseen as singles. |

Aggregated machine-readable rows:

`/ewsc/yektefai/spectra_paper_results_20260513/tables/cross_domain_benchmark_rows.json`

## PerturBench Capsule

The PerturBench repository includes a small development AnnData file:

`/ewsc/yektefai/spectra_depth_demos/repos/perturbench/src/perturbench/data/resources/devel.h5ad`

The new runner `spectrae.benchmark_runners.perturbench_devel_mini_audit` reads
this h5ad directly with `h5py`. It does not require `anndata` or `scanpy`.

Setup:

- cell type: `k562`
- train units: 18 single-gene perturbation response profiles
- eval units: 15 two-gene combination response profiles
- response dimensionality: 500 genes
- model: additive sum of available single-perturbation responses
- similarity: fraction of combination components observed as training singles

Profile-level result:

| Component support | Combination count | Mean profile RMSE | Median profile RMSE |
| ---: | ---: | ---: | ---: |
| 0.0 | 7 | 0.097776 | 0.104994 |
| 0.5 | 3 | 0.074035 | 0.037093 |
| 1.0 | 5 | 0.019950 | 0.019874 |

Artifacts:

`/ewsc/yektefai/spectra_paper_results_20260513/deterministic/perturbench_devel_component_support`

## DART-Eval Status

DART-Eval was not ignored. It was inspected and found to be data-blocked for a
local execution pass:

- local repo: `/ewsc/yektefai/spectra_depth_demos/repos/dart_eval`
- processed task files: Synapse project `syn59522070`
- genome references: Synapse-hosted, including `syn60581044`

The local clone contains code, not ready-to-audit prediction tables. A valid
DART-Eval SPECTRA capsule requires authenticated Synapse download and genome
reference setup. Until that is done, DART-Eval should be listed as planned or
data-blocked, not as an executed result.

## Paper Claim Supported by This Run

Supported:

> SPECTRA converts fixed model predictions plus explicit train-test similarity
> relationships into standardized generalization audits across molecules,
> sequence fitness, and perturbation biology.

Also supported:

> `/spectra` improves agent-executed audits by forcing iterative similarity
> hypotheses, leakage classification, failed-axis reporting, AUSPC, and
> reusable audit artifacts.

Not supported:

> Vanilla agents cannot discover generalization failures.

The BOOM and Martin/NABench agent trials show that strong vanilla agents can
find broad failures. The better claim is about completeness, reproducibility,
and scientific disclosure.

## Fresh Agent-Ablation Follow-Up

After this deterministic run, we executed a fresh blinded agent-ablation across
the three executable domains. Separate vanilla and `/spectra` agents were
started from scratch for molecules, sequence fitness, and perturbation biology.
Agents received only visible train data, held-out predictions, and minimal
metadata. They did not receive paper context, repository context, split labels,
precomputed train-test similarities, grading-only files, or deterministic
SPECTRA summaries.

| Domain | Vanilla | `/spectra` | Delta |
| --- | ---: | ---: | ---: |
| Molecules | 17 / 24 | 24 / 24 | +7 |
| Sequence fitness | 17 / 24 | 23 / 24 | +6 |
| Perturbation biology | 18 / 24 | 24 / 24 | +6 |

The ablation supports the audit-quality claim, not a vanilla-agent-failure
claim. Vanilla agents found the main failure mode in all three domains. The
`/spectra` agents produced more complete audit artifacts: pairwise similarity
files, overlap validation, performance-over-novelty curves, AUSPC or equivalent
summaries, leakage classification, failed-axis reporting, and audit cards.

A stricter follow-up removed all similarity/novelty language from the vanilla
prompt and asked only, "Evaluate whether the model generalizes." The strict
vanilla agents still found the main failures:

| Domain | Strict-naive vanilla | `/spectra` | Delta |
| --- | ---: | ---: | ---: |
| Molecules | 21 / 24 | 24 / 24 | +3 |
| Sequence fitness | 22 / 24 | 23 / 24 | +1 |
| Perturbation biology | 22 / 24 | 24 / 24 | +2 |

This weakens any claim that `/spectra` or the registries are required for agents
to discover schema-obvious generalization axes. The stronger, defensible claim
is that `/spectra` standardizes and records the audit.

Fresh ablation report:

`docs/cross_domain_agent_ablation_20260513.md`

Scoring artifacts:

`/ewsc/yektefai/spectra_agent_ablation_20260513/scoring`

Strict-naive scoring artifacts:

`/ewsc/yektefai/spectra_agent_ablation_20260513/scoring_strict_naive`

## Next Results Needed for a Stronger Submission

1. Run DART-Eval once Synapse data and genome references are available.
2. Replace the PerturBench development capsule with one full public task from
   the benchmark, if data size and dependencies are manageable.
3. Add one clinical/site/time capsule only if individual-level data and
   predictions can be accessed without ambiguity.
4. Repeat the vanilla versus `/spectra` agent benchmark on the cross-domain
   capsules, not only BOOM and Martin/NABench.
