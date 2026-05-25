# SPECTRA v0.4 Agent Rerun on Caduceus Strict Bundle

Date: 2026-05-14

## Purpose

Rerun only the SPECTRA-side agent on the same strict Caduceus visible bundle
after updating the protocol so non-explanatory axes force continued exploration
or scope escalation rather than stopping.

Vanilla was not rerun. The existing vanilla output remains:

`/ewsc/yektefai/spectra_caduceus_strict_20260513/agents/vanilla_generalization_fresh_20260513`

Fresh SPECTRA v0.4 output:

`/ewsc/yektefai/spectra_caduceus_strict_20260513/agents/spectra_generalization_v04_20260514`

## Run Setup

The fresh SPECTRA agent was given the strict visible Caduceus bundle, the local
Caduceus-PS checkpoint, the Caduceus repository, and the current SPECTRA v0.4
protocol. It was told not to inspect or rely on prior vanilla or SPECTRA agent
output directories.

It ran in benchmark mode: frozen Caduceus-PS embeddings with a fresh logistic
probe trained independently for each controlled split point.

## Behavior Change

The prior SPECTRA run tested three representative tasks over central 6-mer,
global 6-mer, and length/GC axes. All nine target-model Caduceus curves were
`not_explanatory`; only k-mer controls showed supported degradation. The run
then stopped with next-experiment suggestions.

The SPECTRA v0.4 rerun did not stop after non-explanatory axes. It tested across
four axis classes:

- surface/content similarity
- target-model representation similarity
- scientific-mechanism similarity
- metadata/context/provenance similarity

It also wrote an explicit `axis_search_budget.json` / scope file describing what
additional data, model access, and compute would be needed if the visible scope
failed.

## Main Results

The v0.4 run tested three tasks:

- `nucleotide_transformer__enhancers`
- `nucleotide_transformer__promoter_tata`
- `nucleotide_transformer__H3`

It found two supported but not yet replicated target-model degradation axes on
the enhancer task:

| task | axis | class | high score | low score | delta |
|---|---|---|---:|---:|---:|
| `nucleotide_transformer__enhancers` | `surface_kmer5` | surface/content | MCC 0.5318 | MCC 0.3874 | 0.1445 |
| `nucleotide_transformer__enhancers` | `caduceus_representation` | target-model representation | MCC 0.6131 | MCC 0.3140 | 0.2991 |

Non-explanatory axes:

- `enhancers / scientific_mechanism_content`
- all three `promoter_tata` axes
- `H3 / metadata_genomic_coordinate`
- `H3 / caduceus_representation`

Because the supported axes were not replicated across another task, this is not
a completed positive SPECTRA finding. The correct result is: SPECTRA v0.4 found
candidate target-model degradation on the enhancer task and escalated to
replication rather than claiming broad Caduceus failure or broad Caduceus
generalization.

## Interpretation

This is a better behavior pattern than the previous run:

1. It did not stop after the first set of failed axes.
2. It searched across multiple axis classes, including Caduceus representation
   similarity.
3. It found a stronger candidate target-model behavioral axis than the previous
   run.
4. It did not overclaim. It marked the finding as unreplicated and requested
   larger scope.

The next experiment should replicate the enhancer representation-axis result
with larger/uncapped train sets, additional tasks, and ideally fresh Caduceus
heads/adapters per split.

## Key Artifacts

- `report.md`
- `audit_card.json`
- `axis_search_budget.json`
- `similarity_hypothesis_scores.json`
- `performance_by_overlap.csv`
- `split_stats.csv`
- `retraining_manifest.csv`
- `next_experiment.json`
