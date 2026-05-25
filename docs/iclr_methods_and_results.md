# Methods and Pilot Results

## Agent-Native Generalization Audit Protocol

We implemented `/spectra`, an agent-native workflow for converting scientific benchmark papers into executable generalization audits. The workflow was exposed through a lightweight Model Context Protocol (MCP) server and a local benchmark utility layer. The server provides a versioned generalizability-analysis procedure, benchmark capsules, audit-card templates, validation utilities, an area-under-spectral-performance-curve (AUSPC) utility, and report-rendering tools. The goal of the protocol is not to replace domain-specific benchmark code, but to standardize the steps by which an agent identifies the scientific unit of generalization, proposes meaningful novelty axes, constructs or plans property graphs, validates train-test overlap, and reports performance as a function of measured novelty.

Each benchmark capsule specifies: the target paper and code repository, the scientific unit of generalization, the original evaluation protocol, the paper's core generalization finding, recommended novelty axes, candidate property graphs, and expected audit artifacts. For the initial release, we created capsules for seven scientific AI benchmarks: BOOM, UMAP-based virtual-screening splits, DART-Eval, NABench, Systema, PerturBench, and GeSS. The benchmark utility supports safe asset fetching, shallow repository cloning, audit-card validation, prompt generation for with- and without-`/spectra` conditions, deterministic artifact scoring, AUSPC computation, and aggregation of experiment summaries. Benchmark assets were stored under `/ewsc/yektefai/spectra_assets` to avoid using transient `/tmp` storage.

## With- vs Without-`/spectra` Audit-Quality Pilot

We first evaluated whether `/spectra` improves the quality of agent-produced generalization audits. We selected BOOM as the initial target because it is directly concerned with molecular out-of-distribution prediction and provides a public paper and repository. Two agents were given the same task: read the BOOM paper and associated repository, reproduce one core generalization finding, and extend the analysis to assess whether model performance degrades under scientifically meaningful train-test novelty. The vanilla condition received only the paper/repository target and the task prompt. The `/spectra` condition additionally received the BOOM capsule, the `/spectra` audit protocol, recommended molecular novelty axes, required audit artifacts, and audit-card schema.

Both agents returned structured JSON audit cards with the same required fields: paper, scientific unit, models, original evaluation, discovered novelty axes, property graphs, spectral split protocol, performance-overlap curve, AUSPC, reproduced claim, additional finding, and citation. Outputs were scored on a seven-dimensional rubric: axis discovery, split construction, curve generation, metric reporting, finding recovery, new insight, and citation/report quality. Each dimension was scored from 0 to 2, where 0 indicates missing or invalid content, 1 indicates a partial or planned answer, and 2 indicates a complete and scientifically useful answer under the constrained no-retraining setting.

The vanilla agent scored 8/14, while the `/spectra` agent scored 11/14. The main improvements were in axis discovery, curve-generation readiness, and new insight. In particular, the `/spectra` condition produced a more explicit molecule-level audit plan with Morgan fingerprint Tanimoto similarity, Bemis-Murcko scaffold novelty, property-extremeness, embedding-distance axes, property-graph definitions, overlap-validation statistics, required artifacts, and a falsifiable expected outcome. This pilot supports the claim that `/spectra` improves the structure and completeness of agent-produced generalization audits. It does not, by itself, establish a computational performance result, because neither condition retrained models or generated new overlap-controlled splits.

## BOOM Numeric Mini-Audit

To move beyond audit-quality scoring, we ran a lightweight numeric mini-audit on BOOM's 10k density task. We used BOOM's official 10k split-generation logic to download the raw 10k density and heat-of-formation CSVs from `FLASK-LLNL/LLNL-10k-Dataset` and generate `10k_dft_data_with_ood_splits.csv`. The resulting density split contained 8,766 training molecules, 440 in-distribution test molecules, and 1,000 OOD test molecules. We used molecule identity as the scientific unit of generalization.

As a lightweight baseline, we trained a `RandomForestRegressor` on Morgan radius-2 fingerprints with 1,024 bits. The model used 120 trees, `max_features="sqrt"`, `n_jobs=-1`, and random seed 42. We trained once on the BOOM density training split and evaluated on the BOOM ID and OOD splits. To measure chemical novelty, we computed, for each test molecule, the maximum Morgan Tanimoto similarity to any molecule in the BOOM training set. We then constructed a post-hoc OOD novelty curve by evaluating nested OOD subsets defined by decreasing maximum train-test Tanimoto thresholds: <=1.0, <=0.8, <=0.7, <=0.6, and <=0.5.

For each subset, we reported sample count, mean maximum Tanimoto similarity, median maximum Tanimoto similarity, RMSE, and MAE. We also computed AUSPC over the OOD curve using novelty defined as `1 - normalized_mean_max_tanimoto` and score defined as negative RMSE, so larger values correspond to better performance under increasing novelty. The runner wrote five artifacts: `audit_card.json`, `split_stats.csv`, `performance_by_overlap.csv`, `spectral_curve.svg`, and `report.md`.

## Numeric Results

The random-forest baseline achieved ID RMSE 0.0443 and ID MAE 0.0338 at mean maximum train-test Tanimoto similarity 0.5595. On the full BOOM OOD density split, RMSE increased to 0.2296 and MAE increased to 0.2085 at mean maximum Tanimoto similarity 0.4726, corresponding to a 5.19x OOD/ID RMSE ratio.

Within the OOD split, performance degraded as chemical overlap decreased. The full OOD set had RMSE 0.2296. Restricting to OOD molecules with maximum Tanimoto <=0.8 yielded RMSE 0.2317; <=0.7 yielded RMSE 0.2354; <=0.6 yielded RMSE 0.2434; and <=0.5 yielded RMSE 0.2534. Across these thresholds, mean maximum train-test Tanimoto decreased from 0.4726 to 0.3900, while RMSE increased by 0.0238. The computed AUSPC was -0.24096 using negative RMSE area over normalized chemical novelty.

These results provide an initial measured example of the `/spectra` audit idea: BOOM's property-tail OOD split is substantially harder than the ID split, and within the OOD split, error further increases as measured chemical train-test overlap decreases. This supports the audit hypothesis that benchmark difficulty can be made more informative by reporting performance as a function of measured novelty rather than only by named split categories.

## Hard Blind Model-Generalizability Pilot

We then ran a stricter blind model-evaluation pilot. Agents were not shown the BOOM paper, repository, split labels, precomputed errors, or precomputed train-test Tanimoto similarities. The agent-visible bundle contained only a trained molecular random-forest model, training molecules and targets, and a held-out table with `sample_id`, `smiles`, `target`, `y_true`, and `y_pred`. The task was to evaluate whether the model generalized. The vanilla condition received no `/spectra` language. The `/spectra` condition received the same artifacts but was asked to identify novelty axes, construct or approximate property graphs, validate overlap, compute prediction errors from raw predictions, and report performance as a function of measured novelty.

Both agents discovered the core generalization failure. Each computed overall held-out MAE 0.1551, RMSE 0.1929, and R2 0.1548 from raw predictions, and each identified prediction-range compression: the training target range was approximately 1.2001 to 1.5647, while held-out labels extended to 1.9631 and predictions only reached 1.5258. Both agents also constructed approximate SMILES-based novelty analyses and found that error increased for less training-like molecules. The primary preregistered rubric therefore saturated, with both agents scoring 16/16.

The `/spectra` condition nevertheless produced a more complete and reusable audit. It defined exact-identity, formula-key, rough-skeleton, approximate-similarity, descriptor-distance, descriptor-OOD-count, and target-shift axes; built an all-pairs train-eval similarity graph over 12,623,040 pairs; validated monotonic overlap reduction as the Jaccard threshold tightened from 0.2 to 0.8; and reported performance curves by similarity bin, descriptor distance, descriptor-OOD count, and target support. A secondary post-hoc audit-depth comparison scored vanilla 6/10 and `/spectra` 10/10. This result supports the narrower claim that `/spectra` improves audit completeness, not the stronger claim that a capable vanilla agent cannot find the failure.

## Limitations

This mini-audit is intentionally limited. It is a post-hoc OOD test-subset analysis, not a full SPECTRA resplitting experiment. The training set was fixed, and only the OOD evaluation subset changed. We evaluated one lightweight baseline model on one BOOM property task. Therefore, the result should be interpreted as evidence that `/spectra` can expose additional performance degradation along a measured novelty axis, not as evidence that the full BOOM benchmark has been reproduced or that SPECTRA improves the trained model. The next required experiment is to generate overlap-controlled train/test splits, retrain or consistently evaluate models across those splits, validate that cross-split overlap decreases monotonically, and repeat the with- versus without-`/spectra` comparison across additional capsules such as DART-Eval, NABench, and Systema.

## 2026-05-13 Cross-Domain Update

The BOOM-only result set was extended with executable sequence-fitness and
perturbation-biology capsules. In NABench, a revised runner added a prospective
local mutation-window identity axis and leakage-aware axis selection. Gregory
2018 mRNA selected mutation-position support, with contiguous RMSE 1.1582 and
lowest-overlap RMSE 1.5723. Martin 2018 MYC enhancer selected local
mutation-window identity, with contiguous RMSE 0.0763 and lowest-overlap RMSE
0.0899. Pitt 2010 ribozyme did not produce a supported prospective curve in the
tested axes; its strongest curve used a post-hoc position-fitness composite and
should be treated as diagnostic only.

PerturBench was also made executable using the repository's bundled
`devel.h5ad`. The capsule evaluates K562 two-gene combination perturbations
using an additive single-perturbation response baseline. Component-support
similarity is prospective because it only uses perturbation component names and
training availability. Mean profile RMSE was 0.0978 with no component support,
0.0740 with one supported component, and 0.0199 with both components supported.

DART-Eval was inspected but not executed because the required task files and
genome references are Synapse-hosted. The supported empirical claim is now
cross-domain but bounded: SPECTRA can convert fixed model predictions and
explicit train-test similarity relationships into standardized audits in
molecules, sequence fitness, and perturbation biology. It does not yet establish
coverage of regulatory DNA because DART-Eval remains data-blocked.
