# SPECTRA Memory Registry

Date: 2026-05-21

## Purpose

The SPECTRA memory registry stores reusable lessons from prior SPECTRA runs:
datasets that were useful, split candidates that passed or failed validity
checks, runnable environment notes, bounded findings, and dead ends that future
agents should not rediscover from scratch.

This is separate from the similarity-definition registry and the dataset
catalog. Similarity definitions answer, "What notion of similarity could be
meaningful?" Dataset catalog entries answer, "What portable resource can a
fresh agent fetch?" Memory entries answer, "What has SPECTRA already learned for
models or datasets like this?"

## Current Seed Entries

| Entry | Use |
| --- | --- |
| `caduceus_external_perturbational_memory` | Reuse the DC-TAP and CRISPR companion design for Caduceus-like regulatory DNA foundation-model audits. |
| `caduceus_strict_sequence_memory` | Reuse strict Caduceus bundle lessons: exact/RC overlap screens, k-mer support screens, environment recovery, and direct frozen-probe caveats. |
| `boom_numeric_mini_audit_memory` | Reuse the BOOM 10k molecular OOD pilot and Morgan Tanimoto curve as a molecule-domain smoke test. |
| `cross_domain_agent_ablation_memory` | Reuse the strict-naive vanilla control, audit-quality rubric, and warning that easy capsules do not prove unique SPECTRA discovery. |

## CLI

```sh
python -m spectrae.cli memory list
python -m spectrae.cli memory search --query "Caduceus regulatory DNA generalization"
python -m spectrae.cli memory suggest --model-description "Caduceus-like DNA foundation model" --domain regulatory_dna
python -m spectrae.cli memory render caduceus_external_perturbational_memory
python -m spectrae.cli memory validate
```

## Agent Policy

Future `/spectra` agents should query memory before proposing new resources or
axes. A matching memory is a prior, not a final answer. The agent should reuse
catalog IDs and design lessons only after checking that the current model,
dataset, labels, and claim boundary match.

The most important rule is negative reuse: if memory says a path was leaky,
control-dominated, only a proxy, or not target-model evidence, the next run must
not present it as a new discovery.

## Result Count

There are at least three result-bearing SPECTRA runs worth preserving:

1. BOOM molecular numeric mini-audit.
2. Strict Caduceus original-task sequence audit and direct frozen-probe recovery.
3. Fresh Caduceus external perturbational audit with matched DC-TAP and CRISPR companions.

The cross-domain vanilla-vs-SPECTRA ablation is also preserved, but it is best
treated as process evidence about audit quality rather than a reusable dataset
finding.
