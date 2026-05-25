# Cross-Domain Agent Ablation

Date: 2026-05-13

This document records the fresh blinded agent-ablation experiment requested for
the paper. Unlike the deterministic benchmark runners, this experiment launched
separate agents from scratch for each executable domain.

Artifacts:

- run root: `/ewsc/yektefai/spectra_agent_ablation_20260513`
- scoring summary: `/ewsc/yektefai/spectra_agent_ablation_20260513/scoring/summary.md`
- scoring JSON: `/ewsc/yektefai/spectra_agent_ablation_20260513/scoring/scores.json`
- rubric: `experiments/cross_domain_agent_ablation/rubric.json`

## Design

We created blinded bundles for three executable domains:

1. molecules,
2. sequence fitness,
3. perturbation biology.

Each bundle exposed only:

- visible training data,
- held-out predictions with `y_true` and `y_pred`,
- minimal task metadata.

The bundles hid benchmark paper context, repository context, split labels,
grading files, deterministic SPECTRA summaries, and precomputed train-test
similarities. DART-Eval was excluded because the required task files and genome
references are Synapse-hosted and unavailable locally.

For each domain, we launched two fresh agents:

- vanilla: broad instruction to evaluate model generalization;
- `/spectra`: same visible data, plus the SPECTRA audit loop and permission to
  use local SPECTRA docs/package/registries.

All agents wrote only to their assigned output directories under:

`/ewsc/yektefai/spectra_agent_ablation_20260513/agent_outputs`

## Scoring

Reports were scored on the fixed 12-dimension rubric:

1. scientific unit,
2. aggregate error,
3. axis discovery,
4. pairwise similarity,
5. overlap validation,
6. performance curve,
7. AUSPC or equivalent summary,
8. failed-axis reporting,
9. adaptive iteration,
10. leakage classification,
11. domain interpretation,
12. artifact quality.

Each dimension was scored 0, 1, or 2.

| Domain | Vanilla | `/spectra` | Delta |
| --- | ---: | ---: | ---: |
| Molecules | 17 / 24 | 24 / 24 | +7 |
| Sequence fitness | 17 / 24 | 23 / 24 | +6 |
| Perturbation biology | 18 / 24 | 24 / 24 | +6 |

## Domain Outcomes

### Molecules

The vanilla agent found the main failure. It computed aggregate MAE 0.1551,
RMSE 0.1929, R2 0.1548, target shift, SMILES q-gram novelty, and nearest-train
feature-distance novelty. It reported that MAE increased from 0.0914 in the
least-novel quantile to 0.2270 in the most-novel quantile.

The `/spectra` agent produced a more complete audit. It evaluated multiple
candidate axes, selected `composition_weighted_jaccard`, validated decreasing
overlap, computed spectral summaries, classified post-hoc target support as
leaky, and marked Morgan fingerprint Tanimoto and Bemis-Murcko scaffold axes as
not evaluable because RDKit was unavailable. On the selected axis, MAE increased
from 0.0493 in the lowest-novelty bin to 0.3054 in the highest-novelty bin.

Interpretation: vanilla discovered the molecular generalization problem;
`/spectra` improved reproducibility, axis disclosure, leakage handling, and
artifact completeness.

### Sequence Fitness

The vanilla agent also found the main sequence failure. It identified that all
359 held-out predictions were identical, computed RMSE 0.0763 and R2 -0.0081,
and inferred that evaluation mutations occupied a contiguous held-out position
block from 241 to 360 with zero training-position overlap.

The `/spectra` agent evaluated five prospective sequence axes and wrote
pairwise similarity files, performance curves, AUSPC values, and an audit card.
Its selected direct axis was nearest train mutation-position similarity
`exp(-distance/30bp)`. Overlap validation passed, but the spectral curve was
non-explanatory: error did not monotonically worsen with novelty. The agent
correctly reported full-sequence Hamming identity and substitution-class
similarity as not evaluable because they collapsed to one similarity level.

Interpretation: this is a useful negative SPECTRA result. `/spectra` did not
force a monotonic curve. It documented that the dominant failure was global
constant prediction rather than a clean novelty-specific degradation under the
tested prospective axes.

### Perturbation Biology

The vanilla agent found the component-support failure. It computed aggregate
RMSE 0.08594, MAE 0.00944, R2 0.233, and showed that predictions exactly
matched a visible-component sum baseline. It found that RMSE worsened from
0.0201 when both perturbation components were supported to 0.1070 when no
components were supported.

The `/spectra` agent formalized this as a spectral audit. It selected
`component_support_fraction`, validated decreasing overlap, computed AUSPC
-0.09876, reported performance-overlap curves, marked gene/cell support as
not evaluable, and classified response-profile similarity as post-hoc/leaky for
split design.

Interpretation: vanilla discovered the same main biological failure, while
`/spectra` provided the reusable audit machinery and leakage-aware reporting.

## Claim Supported

Supported:

> `/spectra` makes agent-executed scientific generalization audits more complete,
> reproducible, leakage-aware, and artifact-rich across molecules, sequence
> fitness, and perturbation biology.

Not supported:

> Vanilla agents cannot discover the broad generalization failure.

In all three executable domains, vanilla agents found the main failure mode.
The value of `/spectra` in this experiment is standardization and audit quality,
not unique discovery.

## Strict-Naive Vanilla Follow-Up

We then reran the vanilla side with a stricter prompt. The strict-naive prompt
did not mention being tested, SPECTRA, novelty, similarity, performance by
group, curves, or audit fields. It only said:

> Evaluate whether the model generalizes.

The agent still received the same visible train data, held-out predictions, and
metadata, and was still forbidden from using hidden files, grading files, paper
context, repository context, prior outputs, or external benchmark context.

Strict-naive scoring used the same 24-point rubric and compared those outputs to
the existing `/spectra` outputs:

| Domain | Strict-naive vanilla | `/spectra` | Delta |
| --- | ---: | ---: | ---: |
| Molecules | 21 / 24 | 24 / 24 | +3 |
| Sequence fitness | 22 / 24 | 23 / 24 | +1 |
| Perturbation biology | 22 / 24 | 24 / 24 | +2 |

This is the most important result for the paper framing. Even without any
similarity or novelty language in the prompt, vanilla agents still inferred the
obvious schema-exposed axes:

- molecules: SMILES string overlap and target-support shift;
- sequence fitness: mutation-position holdout and constant predictions;
- perturbation biology: component support in combination names.

Therefore, the current cross-domain capsules do not demonstrate that registries
or `/spectra` are necessary for discovery. They demonstrate that `/spectra`
adds a standardized audit contract: AUSPC, audit cards, consistent pairwise
artifacts, explicit leakage labels, and failed-axis accounting.

Strict-naive scoring artifacts:

`/ewsc/yektefai/spectra_agent_ablation_20260513/scoring_strict_naive`

## Paper-Ready Sentence

In strict-naive blinded trials across molecular property prediction, nucleotide
fitness prediction, and perturbation-response modeling, vanilla agents
identified the dominant generalization failures without being prompted to search
for similarity or novelty axes. `/spectra` still improved audit completeness
under a fixed 24-point rubric, but the gains were modest: 24 versus 21 for
molecules, 23 versus 22 for sequence fitness, and 24 versus 22 for perturbation
biology. The observed value of `/spectra` in these capsules is therefore
standardization, leakage disclosure, AUSPC/audit-card generation, and reusable
artifact structure rather than unique discovery of the failure mode.
