# Paper Results Run

Date: 2026-05-12

Update: this BOOM-centered run has been extended by the 2026-05-13 cross-domain
run in `docs/paper_results_20260513_cross_domain.md`. The newer run adds two
prospective NABench sequence-fitness curves and a local PerturBench
perturbation-biology capsule.

This document records the experiments executed to populate the results section of
`/home/unix/yektefai/SPECTRA/main.pdf`. All generated experiment artifacts were
written under:

`/ewsc/yektefai/spectra_paper_results_20260512`

## Executed Experiments

1. Deterministic BOOM molecular audit.
2. Deterministic NABench sequence-fitness audits on three assays.
3. Controlled Martin/NABench vanilla versus `/spectra` agent trial.
4. Controlled BOOM four-condition ablation: vanilla, SPECTRA CLI only, CLI plus
   registries, and full `/spectra`.
5. Feasibility inspection for DART-Eval and PerturBench.

DART-Eval and PerturBench were not executed as results capsules in this run.
DART-Eval requires Synapse-hosted task files and genome references. PerturBench
requires Hugging Face or LaminDB datasets and prediction artifacts. The local
repos contain code and notebooks, but not ready-to-audit prediction tables.

## Deterministic Benchmark Results

| Domain | Task | Model | Axis | Near/random RMSE | High-novelty RMSE | AUSPC | Key finding |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Molecules | BOOM 10k density | RF Morgan fingerprints | max Morgan Tanimoto to train | 0.0443 ID | 0.2534 lowest-overlap OOD | -0.2410 | OOD RMSE was 0.2296 versus ID RMSE 0.0443; within OOD, RMSE increased as mean max Tanimoto dropped to 0.390. |
| Sequence fitness | NABench Gregory_2018_mRNA | position-aware ridge | fitness_support_similarity | 1.0664 random | 1.8921 lowest-overlap | -2.6175 | Contiguous RMSE was 1.1582; the strongest curve used post-hoc fitness support. |
| Sequence fitness | NABench Martin_2018_myc_enhancer | position-aware ridge | position_fitness_composite_similarity | 0.1086 random | 0.1038 lowest-overlap | -0.1759 | Contiguous RMSE was 0.0763; the selected explanatory curve used post-hoc fitness support. |
| Sequence fitness | NABench Pitt_2010_ribozyme | position-aware ridge | position_fitness_composite_similarity | 0.0135 random | 0.0165 lowest-overlap | -0.0214 | Contiguous RMSE was 0.0137; the selected explanatory curve used post-hoc fitness support. |

The BOOM result is the cleanest flagship result because its selected axis is
prospective and computed only from molecular structures. The NABench results
demonstrate the iterative audit loop, but the strongest selected axes are
post-hoc explanatory because they use observed fitness labels.

Flagship figure candidate:

`/ewsc/yektefai/spectra_paper_results_20260512/deterministic/boom_numeric_mini_audit/artifacts/spectral_curve.svg`

## Agent Trial: Martin/NABench

Input/output root:

`/ewsc/yektefai/spectra_agent_trials/iterative_martin_20260512`

Both agents found a near-constant sequence-fitness predictor with RMSE about
0.0763 and R2 about -0.008. The vanilla agent produced a competent static
analysis but did not produce an iterative audit, AUSPC, or standardized artifact
bundle. The `/spectra` agent generated pairwise similarities, per-axis audits,
failed-axis reporting, leakage classification, AUSPC, and a selected prospective
local-context axis.

| Condition | Score |
| --- | ---: |
| Vanilla | 17 / 24 |
| `/spectra` | 24 / 24 |

## Agent Trial: BOOM Molecular OOD

Input/output root:

`/ewsc/yektefai/spectra_paper_results_20260512/agent_trials/boom_molecular_ood`

All conditions received anonymous train/eval molecular files with no paper,
split labels, or precomputed train-test similarity.

| Condition | Score | Main behavior |
| --- | ---: | --- |
| Vanilla | 19 / 24 | Found exact-overlap, Morgan, scaffold, descriptor, and target-support axes; found monotonic Morgan and descriptor novelty degradation; did not compute AUSPC or produce standardized SPECTRA audit cards. |
| SPECTRA CLI only | 22 / 24 | Constructed Morgan novelty, ran deterministic SPECTRA audit, computed AUSPC, and produced audit artifacts. |
| CLI + registries | 23 / 24 | Used registry-backed Morgan Tanimoto and evaluated scaffold, physicochemical, and radius-sensitivity axes. |
| Full `/spectra` | 24 / 24 | Ran a structured hypothesis loop, selected `descriptor_neighborhood`, reported post-hoc/invalid axes, computed AUSPC, and produced per-axis artifacts. |

Key numeric BOOM agent-trial results:

- Aggregate eval RMSE: about 0.1929.
- Vanilla Morgan novelty curve: MAE increased from 0.0829 in the lowest-novelty quintile to 0.2218 in the highest-novelty quintile.
- CLI-only selected Morgan AUSPC: -0.2221.
- CLI+registries selected Morgan AUSPC: -0.2268.
- Full `/spectra` selected descriptor-neighborhood AUSPC: -0.2696; RMSE increased from 0.1929 overall to 0.3088 in the highest descriptor-novelty subset.

Interpretation: BOOM is an easy task for a strong vanilla agent because SMILES
make molecular similarity obvious. The stronger claim is not that vanilla fails.
The supported claim is that `/spectra` increases artifact standardization,
explicit hypothesis-loop behavior, registry-backed alternatives, metric
reporting, and prospective/post-hoc classification.

## Claim Support

Claim 1, spectral curves reveal novelty-dependent failure:

- Strongly supported by BOOM. The prospective chemical-overlap curve shows large
  error increases as train-test molecular similarity decreases.
- Partially supported by NABench. The iterative loop finds explanatory curves,
  but the strongest axes use labels and should be framed as post-hoc diagnostic
  findings, not prospective split-design axes.

Claim 2, `/spectra` improves agent-executed audits:

- Supported, but with a careful framing. Vanilla agents can find broad failures,
  especially on obvious molecular tasks. `/spectra` improves completeness,
  reproducibility, failed-axis reporting, leakage classification, AUSPC, and
  reusable artifact quality.

Claim 3, registries provide useful scientific priors:

- Partially supported by the BOOM ablation. The CLI-only condition can execute a
  valid audit once an axis is chosen. Registries and full `/spectra` improve the
  breadth and documentation of candidate axes and computation strategies.

## Paper-Ready Result Sentence

Across one molecular and one sequence-fitness agent benchmark, `/spectra`
improved audit completeness from 17/24 to 24/24 on the sequence task and from
19/24 to 24/24 on the molecular task. On the molecular task, even the vanilla
agent identified the main novelty-dependent failure, indicating that
`/spectra` should be framed as improving reproducibility, audit structure, and
scientific disclosure rather than as uniquely enabling discovery of the failure.

## Remaining Work Before Submission

The paper would be stronger with one additional fully executable non-molecular,
non-NABench capsule. The most likely next target is PerturBench, but it requires
downloading a processed dataset and obtaining or generating prediction tables.
DART-Eval is higher friction because the required files are Synapse-hosted and
depend on genome references.
