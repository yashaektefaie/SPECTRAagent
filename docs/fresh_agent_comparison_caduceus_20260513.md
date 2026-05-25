# Fresh Agent Comparison: Caduceus Generalization

Date: 2026-05-13

## Setup

We ran two fresh agents on the same strict Caduceus bundle:

`/ewsc/yektefai/spectra_caduceus_strict_20260513/agent_inputs/caduceus_strict_visible`

The vanilla agent was asked only to assess whether Caduceus-PS generalizes.

The SPECTRA agent was asked the same task, but with the `/spectra` exploration
protocol and explicit instructions to test similarity hypotheses, recover from
ordinary environment blockers, and avoid treating surrogate baselines as target
model evidence.

Output directories:

- Vanilla: `/ewsc/yektefai/spectra_caduceus_strict_20260513/agents/vanilla_generalization_fresh_20260513`
- SPECTRA: `/ewsc/yektefai/spectra_caduceus_strict_20260513/agents/spectra_generalization_fresh_20260513`
- Comparison JSON: `/ewsc/yektefai/spectra_caduceus_strict_20260513/agents/fresh_agent_comparison_20260513.json`

## Artifact-Level Difference

The vanilla agent produced useful but non-standardized artifacts:

- `all_tasks_exact_overlap.csv`
- `sequence_audit_core/kmer_probe_controlled_splits.csv`
- `source_name_controlled_probe.csv`
- Caduceus dependency-install logs
- `final_report.md`

The SPECTRA agent produced the expected protocol artifact set:

- `audit_card.json`
- `spectral_properties.json`
- `mode_decision.json`
- `question_trace.json`
- `split_stats.csv`
- `performance_by_overlap.csv`
- `retraining_manifest.csv`
- `similarity_hypothesis_scores.json`
- `next_experiment.json`
- `blockers.json`
- `report.md`

The comparison script found:

- Vanilla had no top-level `performance_by_overlap.csv` or `split_stats.csv`.
- SPECTRA had 54 performance-overlap rows and 27 split-stat rows.
- Vanilla did not obtain direct Caduceus controlled-split metrics.
- SPECTRA did obtain direct Caduceus frozen-embedding probe metrics.

## Environment Recovery

This was the clearest behavioral difference.

The vanilla agent tried to create a local venv and install Caduceus runtime
dependencies, but concluded direct Caduceus evaluation was blocked by missing
`mamba_ssm`.

The SPECTRA agent inspected available environments and found that:

- base Python lacked scientific packages,
- `fleming` had `torch` and `transformers` but lacked `mamba_ssm`,
- `pgt` could load and run the local Caduceus-PS checkpoint.

The SPECTRA run therefore recovered the target model and ran Caduceus-PS frozen
embeddings with fresh logistic heads per controlled split.

## Vanilla Findings

The vanilla agent identified real split-quality concerns.

Largest exact or reverse-complement train/test overlaps:

| task | test rows | exact/RC overlap | fraction |
| --- | ---: | ---: | ---: |
| `human_enhancers_ensembl` | 30970 | 11784 | 0.380 |
| `enhancers` | 400 | 100 | 0.250 |
| `enhancers_types` | 400 | 100 | 0.250 |
| `splice_sites_all` | 3000 | 221 | 0.0737 |
| `demo_human_or_worm` | 25000 | 650 | 0.026 |
| `promoter_tata` | 621 | 6 | 0.0097 |

The vanilla k-mer/source-name probes found surrogate-model degradation on some
axes:

- `enhancers_types`: k-mer probe MCC dropped from 0.367 to 0.273 after exact/RC
  decontamination.
- `splice_sites_acceptors`: source-name filtered F1 dropped from 0.718 to 0.571.
- `splice_sites_donors`: source-name filtered F1 dropped from 0.719 to 0.624.

However, the vanilla agent concluded direct Caduceus metrics remained blocked.
That conclusion was too pessimistic because the `pgt` environment could run
Caduceus.

## SPECTRA Findings

The SPECTRA agent ran benchmark-mode frozen Caduceus probes on three tasks:

- `nucleotide_transformer:enhancers`
- `nucleotide_transformer:promoter_tata`
- `genomic_benchmarks:dummy_mouse_enhancers_ensembl`

It tested three prospective axes:

- `central_6mer_jaccard`
- `global_6mer_jaccard`
- `length_gc_profile`

For each selected split, it fit:

- a fresh logistic head on frozen Caduceus-PS mean-pooled embeddings,
- a fresh k-mer SGD probe as a control.

Curve score summary:

- Caduceus curves: 9/9 `not_explanatory`.
- k-mer controls: 1 `monotonic_supported`, 1 `weak_supported`, 7
  `not_explanatory`.

Strongest Caduceus high-to-low degradations:

| task/axis | high score | low score | degradation | status |
| --- | ---: | ---: | ---: | --- |
| dummy mouse / length-GC | 0.838 | 0.817 | 0.0207 | not explanatory |
| dummy mouse / global 6-mer | 0.797 | 0.778 | 0.0189 | not explanatory |
| promoter TATA / global 6-mer | 0.953 | 0.945 | 0.0078 | not explanatory |

Supported control signals:

| task/axis | model | high score | low score | status |
| --- | --- | ---: | ---: | --- |
| promoter TATA / global 6-mer | k-mer control | 0.918 | 0.833 | monotonic supported |
| dummy mouse / global 6-mer | k-mer control | 0.777 | 0.718 | weak supported |

## Interpretation

The result does not support the simple claim that `/spectra` makes the agent
discover a Caduceus failure that vanilla misses.

The stronger and more defensible claim is:

`/spectra` changed the agent from a general audit into a structured
target-model investigation. It recovered the Caduceus environment, produced
standard SPECTRA artifacts, ran controlled target-model probes, separated
surrogate-model failures from Caduceus behavior, and treated non-explanatory
axes as real findings.

In this run, vanilla found data and surrogate-model concerns, but did not reach
direct Caduceus evidence. SPECTRA reached direct Caduceus evidence and concluded
that the tested novelty axes did not explain Caduceus frozen-probe failure.

## Paper-Relevant Claim

This comparison supports a process claim, not a failure claim:

SPECTRA-equipped agents more reliably turn a broad generalization question into
a documented target-model generalization audit with explicit similarity axes,
controlled splits, fresh downstream heads, blocker recovery, negative findings,
and next-experiment decisions.

It does not yet support:

SPECTRA-equipped agents discover that Caduceus fails to generalize on these
tasks.

## Next Experiment

The highest-value next step is to run the same vanilla-vs-SPECTRA comparison on
a task where the expected failure is less obvious but direct target-model
execution is straightforward. For Caduceus specifically, the next stronger test
is full downstream fine-tuning or adapter/head tuning over the
`promoter_tata/global_6mer_jaccard` splits, where the k-mer control degraded but
the frozen Caduceus probe did not.
