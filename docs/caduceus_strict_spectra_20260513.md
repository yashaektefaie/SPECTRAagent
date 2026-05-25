# Strict Caduceus SPECTRA Result

Date: 2026-05-13

## Setup

We built a strict Caduceus bundle under:

`/ewsc/yektefai/spectra_caduceus_strict_20260513/agent_inputs/caduceus_strict_visible`

The bundle includes the Caduceus paper, repo, Caduceus-PS checkpoint, and the
original downstream evaluation datasets named by the Caduceus repository:

- 8 GenomicBenchmarks tasks.
- 18 Nucleotide Transformer downstream tasks.
- Long-range eQTL metadata from the original Caduceus eQTL evaluation source.

The bundle intentionally excludes DART-Eval papers, conclusions, and result
tables.

## SPECTRA Agent Result

The SPECTRA agent wrote outputs to:

`/ewsc/yektefai/spectra_caduceus_strict_20260513/agents/caduceus_strict_spectra`

Main artifacts:

- `report.md`
- `report.json`
- `artifacts/task_overlap_summary.csv`
- `artifacts/spectral_curve_rows.csv`
- `artifacts/strict_audit_full.json`

The agent audited 26 original sequence-classification tasks with 1,155,782
training sequences and 238,656 test sequences. It did not use DART-Eval.

## Findings

The strict original Caduceus evaluation data supports DART-like concerns at the
data-split level:

- Several original test sets contain exact train duplicates.
- Many tasks have high local train support under 15-mer presence.
- The 15-mer train-support curve was monotonic for 26/26 tasks.

Largest exact train-test duplicate rates:

- `genomic_benchmarks/human_enhancers_ensembl`: 11,774 duplicate test rows, 38.02%.
- `nucleotide_transformer/enhancers`: 100 duplicate test rows, 25.00%.
- `nucleotide_transformer/enhancers_types`: 100 duplicate test rows, 25.00%.
- `nucleotide_transformer/splice_sites_all`: 221 duplicate test rows, 7.37%.
- `genomic_benchmarks/demo_human_or_worm`: 647 duplicate test rows, 2.59%.

Highest mean test 15-mer train support:

- `nucleotide_transformer/H3K4me3`: 0.7105.
- `nucleotide_transformer/H3K36me3`: 0.6843.
- `nucleotide_transformer/H4ac`: 0.6780.
- `nucleotide_transformer/H3K14ac`: 0.6752.
- `nucleotide_transformer/H3K4me1`: 0.6387.

## Interpretation

This is evidence that parts of the original Caduceus evaluation are not strict
novelty tests. It is a DART-like concern because it questions whether standard
task scores reflect regulatory generalization rather than local sequence support
or duplicate exposure.

It is not yet a direct Caduceus model-failure claim. The strict audit did not
train fresh downstream Caduceus heads/probes on SPECTRA-controlled splits and
did not compute Caduceus performance-overlap curves.

This run exposed a protocol issue: the agent treated a data-level screen as a
reasonable stopping point. The SPECTRA protocol has since been updated so this
kind of result is explicitly pre-benchmark screening. A suspicious overlap axis
must trigger the smallest feasible behavioral follow-up, or the report must name
a concrete blocker.

The next model-level experiment is to fine-tune or fit frozen-embedding probes
for Caduceus on SPECTRA-controlled splits and report performance as a function
of measured overlap.

## Exploration-Protocol Rerun

After the SPECTRA procedure was updated from a checklist-style audit to an
active exploration protocol, we reran the strict Caduceus task with the same
input bundle. The rerun wrote outputs to:

`/ewsc/yektefai/spectra_caduceus_strict_20260513/agents/caduceus_strict_spectra_exploration_rerun`

The rerun changed the agent behavior in the intended direction. Instead of
stopping after the split-overlap screen, the agent produced:

- `question_trace.json`
- `next_experiment.json`
- `blockers.json`
- `performance_by_overlap.csv`
- `split_stats.csv`
- `retraining_manifest.csv`
- `similarity_hypothesis_scores.json`

The artifact comparison is saved at:

`/ewsc/yektefai/spectra_caduceus_strict_20260513/agents/caduceus_strict_exploration_comparison.json`

The old run had no behavioral-artifact set and no exploration-artifact set. The
rerun had both, with 16 performance-overlap rows and 16 controlled split rows.

### Rerun Behavior

The updated agent framed the audit as three linked questions:

1. Do the original Caduceus train/test splits contain sequence-level overlap
   that could support DART-like generalization concerns?
2. If overlap exists, does performance change when train/test k-mer overlap is
   reduced?
3. Can this be claimed as a direct Caduceus checkpoint benchmark result?

It screened exact sequence identity, reverse-complement canonical identity, and
6-mer MinHash nearest-neighbor overlap across all 26 original tasks. It then ran
the smallest feasible behavioral follow-up available in the current environment:
fresh k-mer logistic baselines on controlled decontamination splits for four
representative tasks:

- `nucleotide_transformer/promoter_tata`
- `nucleotide_transformer/enhancers`
- `nucleotide_transformer/H3`
- `genomic_benchmarks/human_enhancers_cohn`

This is not a direct Caduceus checkpoint result. It is a lightweight behavioral
follow-up showing whether the discovered overlap axes explain model behavior for
fresh downstream baselines.

### Rerun Findings

The strongest behavioral signal was on `nucleotide_transformer/enhancers`.
Removing highly similar training examples reduced the k-mer baseline MCC from
0.4557 on the original split to 0.3654 after removing train examples with
estimated test similarity >= 0.95 or >= 0.85. The agent marked this axis as
monotonic.

The other three follow-ups were weak or non-explanatory:

- `promoter_tata`: F1 changed from 0.9152 to 0.9017-0.9157 across retained
  splits.
- `H3`: MCC slightly increased as high-overlap examples were removed.
- `human_enhancers_cohn`: accuracy did not track the k-mer overlap axis.

This matters for the protocol: the agent did not treat every discovered
similarity axis as a confirmed explanation. It tested whether reducing overlap
changed behavior, accepted one axis as explanatory for one task, and treated the
others as weak or non-explanatory.

### Remaining Blocker

The exploration rerun treated direct Caduceus execution as blocked because the
default SPECTRA environment did not have `torch` or `transformers`:

- `/ewsc/yektefai/envs/envs/spectra-full-ablation/bin/python` does not have
  `torch`.
- The same environment does not have `transformers`.

That was too weak as a terminal blocker. The agent should have inspected the
Caduceus repo, dependency file, local model directory, and available micromamba
environments before falling back to a surrogate baseline.

## Direct Caduceus Recovery Attempt

We tested that recovery path manually. The existing micromamba environment:

`/ewsc/yektefai/envs/envs/pgt`

has `torch`, `transformers`, and `mamba_ssm`. With:

`PYTHONPATH=/ewsc/yektefai/spectra_caduceus_strict_20260513/repos/caduceus`

the local Caduceus-PS checkpoint loaded from:

`/ewsc/yektefai/spectra_caduceus_strict_20260513/model/caduceus_ps`

A smoke test produced a valid Caduceus hidden state on GPU, so direct Caduceus
probing was not actually blocked.

We then ran a direct target-model follow-up on
`nucleotide_transformer/enhancers`:

`experiments/caduceus_strict/run_caduceus_direct_probe.py`

The script extracts frozen Caduceus mean-pooled embeddings and fits a fresh
logistic probe for each SPECTRA-controlled train-retention split.

Artifacts are saved under:

`/ewsc/yektefai/spectra_caduceus_strict_20260513/agents/caduceus_direct_probe_recovery`

Full-task result:

| split | retained train | removed train | MCC |
| --- | ---: | ---: | ---: |
| original | 14968 | 0 | 0.4602 |
| remove train sim >= 0.95 | 14861 | 107 | 0.3924 |
| remove train sim >= 0.85 | 14861 | 107 | 0.3924 |
| remove train sim >= 0.75 | 14860 | 108 | 0.4040 |

This is the first direct Caduceus behavioral evidence in the strict run. It
agrees with the k-mer baseline direction on the `enhancers` task: removing a
small number of high-overlap training examples substantially reduces performance
for a fresh downstream classifier.

The result is still a frozen-embedding probe, not full Caduceus fine-tuning. The
next stronger experiment is to run fresh Caduceus fine-tuning heads per split,
or repeat the frozen-embedding probe across the other suspicious tasks.
