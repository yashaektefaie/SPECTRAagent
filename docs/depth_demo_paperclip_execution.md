# Depth Demo Paperclip Execution Report

Date: 2026-05-12

This report records the first depth-demo pass after building the SPECTRA
registries and MCP interface. The goal was to stop adding registry breadth and
instead test whether selected benchmark capsules can become polished executable
demos.

## Paperclip Mining

Paperclip was available through:

```bash
/ewsc/yektefai/envs/envs/boltz/bin/paperclip
```

Searches were run for four launch areas:

- molecular OOD generalization,
- regulatory DNA and nucleotide fitness,
- single-cell perturbation response,
- clinical/site/time generalization.

The most useful Paperclip result sets were:

- `s_b3502834`: molecular OOD, scaffold, random, cluster, and BOOM-like papers.
- `s_7f98e379`: DART-Eval, NABench, and related regulatory DNA/sequence papers.
- `s_734b06c2`: PerturBench and perturbation-response benchmark papers.
- `s_d6cf0b79`: clinical prediction generalizability and external validation papers.

The strongest four executable depth-demo candidates were:

| Capsule | Domain | Paperclip evidence | Immediate status |
| --- | --- | --- | --- |
| BOOM | Molecules | Paperclip found BOOM as a direct molecular OOD benchmark with open GitHub code. | Runnable now. |
| NABench | Nucleotide fitness | Paperclip found random vs contiguous CV as an explicit extrapolation setup, with public DMS CSVs in the repo. | Runnable now. |
| DART-Eval | Regulatory DNA | Paperclip found chromosome/task split details and DART's public benchmark repo. | Repo fetched, data requires Synapse download. |
| PerturBench | Perturbation biology | Paperclip found covariate transfer, combo prediction, and split definitions. | Repo fetched, processed h5ad data is on HuggingFace and may be large. |

Clinical/site/time papers were useful conceptually, but not selected for the
first execution pass because immediate individual-level data access is less
clear.

## Assets Fetched

The benchmark fetcher downloaded papers and shallow-cloned repos into:

```bash
/ewsc/yektefai/spectra_depth_demos
```

Command:

```bash
/ewsc/yektefai/envs/envs/boltz/bin/python -m spectrae.spectra_benchmarks fetch \
  --paper-ids boom,dart_eval,nabench,perturbench \
  --output-dir /ewsc/yektefai/spectra_depth_demos \
  --include-repos \
  --include-data
```

Fetched repos:

- `/ewsc/yektefai/spectra_depth_demos/repos/boom`
- `/ewsc/yektefai/spectra_depth_demos/repos/dart_eval`
- `/ewsc/yektefai/spectra_depth_demos/repos/nabench`
- `/ewsc/yektefai/spectra_depth_demos/repos/perturbench`

## Executed Demo 1: BOOM Molecules

Command:

```bash
PYTHONPATH=/ewsc/yektefai/spectra_depth_demos/repos/boom \
/ewsc/yektefai/envs/envs/boltz/bin/python \
  -m spectrae.benchmark_runners.boom_numeric_mini_audit \
  --output-dir /ewsc/yektefai/spectra_depth_demos/boom_numeric_mini_audit \
  --n-estimators 120
```

Setup:

- Dataset: BOOM 10k density.
- Model: `RandomForestRegressor`.
- Representation: Morgan radius-2 fingerprints, 1024 bits.
- Similarity definition: maximum Morgan Tanimoto similarity to the training set.
- Operating point: BOOM property-tail OOD split.
- SPECTRA curve: OOD performance over decreasing train-set chemical overlap.

Results:

| Subset | n | Mean max Tanimoto | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| ID test | 440 | 0.5595 | 0.0443 | 0.0338 |
| OOD all | 1000 | 0.4726 | 0.2296 | 0.2085 |
| OOD max Tanimoto <= 0.8 | 978 | 0.4632 | 0.2317 | 0.2110 |
| OOD max Tanimoto <= 0.7 | 931 | 0.4489 | 0.2354 | 0.2149 |
| OOD max Tanimoto <= 0.6 | 823 | 0.4230 | 0.2434 | 0.2231 |
| OOD max Tanimoto <= 0.5 | 653 | 0.3900 | 0.2534 | 0.2330 |

AUSPC:

- `-0.24096112142647133`

Artifacts:

- `/ewsc/yektefai/spectra_depth_demos/boom_numeric_mini_audit/artifacts/audit_card.json`
- `/ewsc/yektefai/spectra_depth_demos/boom_numeric_mini_audit/artifacts/performance_by_overlap.csv`
- `/ewsc/yektefai/spectra_depth_demos/boom_numeric_mini_audit/artifacts/spectral_curve.svg`
- `/ewsc/yektefai/spectra_depth_demos/boom_numeric_mini_audit/artifacts/report.md`

Interpretation:

BOOM is the strongest current depth demo. It shows the desired pattern: ID is
much easier than OOD, and within OOD, lower chemical overlap corresponds to
higher RMSE. This is a good launch-anchor demo for the molecular setting.

## Executed Demo 2: NABench Sequence Fitness

A new runner was added:

- `spectrae/benchmark_runners/nabench_sequence_mini_audit.py`

The runner:

1. loads a public NABench DMS CSV,
2. creates a contiguous mutational-region holdout,
3. trains a position-aware ridge baseline,
4. emits `train.csv`, `eval_predictions.csv`, and `pairwise_similarity.csv`,
5. treats similarity definitions as hypotheses,
6. runs one SPECTRA pairwise audit per candidate axis,
7. scores whether each curve supports monotonic or localized degradation,
8. selects the strongest explanatory axis and records the iteration trace.

The current candidate axes are:

- mutation-position support,
- sequence identity,
- mutation-depth support,
- post-hoc fitness support,
- post-hoc position-fitness composite support.

The post-hoc fitness axes use evaluation labels. They are appropriate for
explaining an observed failure mode after labels are available, but not for
prospective split design.

Primary command:

```bash
/ewsc/yektefai/envs/envs/boltz/bin/python \
  -m spectrae.benchmark_runners.nabench_sequence_mini_audit \
  --repo-dir /ewsc/yektefai/spectra_depth_demos/repos/nabench \
  --output-dir /ewsc/yektefai/spectra_depth_demos/nabench_sequence_mini_audit \
  --dataset Martin_2018_myc_enhancer.csv
```

Additional assays were run with the same runner:

- `Gregory_2018_mRNA.csv`
- `Pitt_2010_ribozyme.csv`

### First-Pass NABench Results

| Assay | Random RMSE | Contiguous RMSE | AUSPC | Main curve result |
| --- | ---: | ---: | ---: | --- |
| Martin 2018 MYC enhancer | 0.108639 | 0.076314 | -0.069685 | No monotonic degradation; lowest-overlap subset had lower RMSE than all eval. |
| Gregory 2018 mRNA | 1.066440 | 1.158248 | -1.253552 | Contiguous holdout was worse than random; max RMSE appeared at an intermediate/low-support subset. |
| Pitt 2010 ribozyme | 0.013479 | 0.013725 | -0.012269 | Nearly flat; no strong monotonic novelty signal. |

Artifacts:

- `/ewsc/yektefai/spectra_depth_demos/nabench_sequence_mini_audit/summary.md`
- `/ewsc/yektefai/spectra_depth_demos/nabench_sequence_mini_audit/spectra_audit/audit_card.json`
- `/ewsc/yektefai/spectra_depth_demos/nabench_sequence_mini_audit_Gregory_2018_mRNA/summary.md`
- `/ewsc/yektefai/spectra_depth_demos/nabench_sequence_mini_audit_Pitt_2010_ribozyme/summary.md`

Interpretation:

The NABench mini-audit is useful, but not because it gives a simple marketing
win. It shows why SPECTRA is useful as an audit framework: it can reveal whether
a named split actually corresponds to performance decay over the proposed
scientific distance axis.

In these three lightweight runs, mutation-position support was not universally
monotonic. Gregory showed a directionally expected contiguous-vs-random gap and
a localized failure region. Martin and Pitt did not show a clean degradation
curve. This suggests the next sequence demo should either use more NABench
assays, a stronger sequence model, or a richer similarity definition combining
mutation position, mutational identity, and assay family.

### Iterative NABench Results

After implementing the similarity-hypothesis loop, the same three assays were
rerun under `/ewsc/yektefai/spectra_depth_demos/nabench_sequence_mini_audit_iterative*`.

| Assay | Initial position-axis status | Selected axis | Selected-axis status | Selected AUSPC | Main change |
| --- | --- | --- | --- | ---: | --- |
| Martin 2018 MYC enhancer | `not_explanatory` | `position_fitness_composite_similarity` | `monotonic_supported` | -0.175900 | Position alone failed; the composite post-hoc axis found a harder low-support subset. |
| Gregory 2018 mRNA | `monotonic_supported` | `fitness_support_similarity` | `monotonic_supported` | -2.617461 | Position already worked, but target-support explained a larger failure gradient. |
| Pitt 2010 ribozyme | `not_explanatory` | `position_fitness_composite_similarity` | `monotonic_supported` | -0.021391 | Position alone was flat/downward; the post-hoc composite found a small but monotonic degradation. |

The behavior now matches the intended SPECTRA loop:

1. Start with a plausible scientific similarity definition.
2. Run the spectral curve.
3. If the curve is non-explanatory, record that as a finding.
4. Use the finding to try the next similarity hypothesis.
5. Distinguish prospective axes from post-hoc explanatory axes.

This changes the interpretation of non-monotonic curves. They are no longer
"failed demos." They are audit evidence that a proposed notion of similarity is
not sufficient for explaining the model's observed generalization behavior.

## DART-Eval Feasibility

DART-Eval was fetched, but not executed in this pass.

Reason:

- The repo is available at `/ewsc/yektefai/spectra_depth_demos/repos/dart_eval`.
- The processed benchmark data is stored in Synapse project `syn59522070`.
- The README states that each task directory contains `data.h5`; these files
  must be downloaded from Synapse before a runnable SPECTRA demo can be built.

Best SPECTRA demo target:

- Start with the chromosome holdout already described in the paper.
- Use scientific unit `regulatory DNA sequence`.
- Use similarity definitions: sequence identity, k-mer cosine/Jaccard, GC
  content control, motif profile similarity, and chromosome holdout as the
  operating point.

## PerturBench Feasibility

PerturBench was fetched, but not executed in this pass.

Reason:

- The repo is available at `/ewsc/yektefai/spectra_depth_demos/repos/perturbench`.
- Processed datasets are available as gzipped h5ad files on HuggingFace.
- Running the full benchmark requires installing PerturBench and downloading
  potentially large single-cell assets.

Best SPECTRA demo target:

- Start with Srivatsan20 or Frangieh21 covariate transfer.
- Use scientific unit `perturbation-covariate response`.
- Use similarity definitions: perturbation identity or embedding distance,
  cell-type/covariate distance, pathway/ontology distance, and combinatorial
  component overlap.

## What This Means

This pass supports the updated framing:

> Agents equipped with `/spectra` more reliably transform static benchmark
> papers into executable scientific generalization audits.

The strongest evidence so far is BOOM. The NABench pilot adds cross-domain
evidence that the SPECTRA workflow is runnable outside molecules, but it also
shows that the first chosen sequence axis is not always sufficient.

That is not a failure of the framework. It is exactly what a generalization
audit should do: distinguish a real distance-dependent failure from a split that
is difficult for other reasons, or not difficult at all.

## Next Execution Steps

1. Promote BOOM as the polished molecular launch demo.
2. Expand NABench from three single-assay pilots to a small assay panel and test
   richer sequence similarities.
3. Download DART-Eval Synapse `data.h5` files and run a chromosome/k-mer/motif
   SPECTRA audit.
4. Install PerturBench in a separate environment and download one small h5ad
   dataset for a covariate-transfer SPECTRA audit.
5. After the executable demos are stable, run the actual vanilla-agent vs
   `/spectra`-agent benchmark using these capsules and score audit cards.

## 2026-05-13 Follow-Up

The next pass resolved two issues from this report.

For NABench, the runner now includes a prospective
`mutation_centered_window_identity_similarity` axis and ranks leakage-free
supported axes ahead of post-hoc label-based axes. In the rerun, Gregory 2018
mRNA selected mutation-position support, Martin 2018 MYC enhancer selected
local mutation-window identity, and Pitt 2010 ribozyme remained diagnostic-only
because the strongest supported curve used observed fitness labels.

For PerturBench, the full benchmark assets remain external, but the repository
bundles a small `devel.h5ad`. A new local capsule uses that file to audit
K562 combination perturbation prediction. The prospective axis is component
support, defined as the fraction of combination components observed as
single-perturbation profiles in training. Mean profile RMSE was 0.0978 when no
components were supported, 0.0740 when one of two was supported, and 0.0199
when both were supported.

DART-Eval remains data-blocked because the processed task files and genome
references are Synapse-hosted.

See `docs/paper_results_20260513_cross_domain.md`.
