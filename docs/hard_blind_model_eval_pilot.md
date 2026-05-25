# Hard Blind Model-Generalizability Pilot

Date: 2026-05-11

## Goal

This pilot tested the stricter version of the blind molecular-model task:

> Can an agent discover a BOOM-like molecular generalization failure without
> seeing the BOOM paper, split labels, precomputed errors, or precomputed
> train-test similarity?

Agents received only the trained model, training data, raw held-out labels, and
raw held-out predictions. The visible prediction file had only:

- `sample_id`
- `smiles`
- `target`
- `y_true`
- `y_pred`

The bundle was stored under
`/ewsc/yektefai/spectra_assets/hard_blind_molecular_model_eval`.

## Conditions

Vanilla:

- Asked to evaluate whether the model generalizes.
- Received no `/spectra` terminology.
- Was not given precomputed errors, similarity scores, split labels, BOOM
  paper context, or benchmark repository context.

`/spectra`:

- Received the same visible artifacts.
- Was asked to apply `/spectra`: identify the scientific unit, construct
  novelty axes, build or approximate property graphs, validate overlap, compute
  prediction errors from `y_true` and `y_pred`, and report performance as a
  function of measured novelty.

## Primary Result

Both agents discovered the core generalization failure.

| Condition | Primary Score |
| --- | ---: |
| Vanilla | 16 / 16 |
| `/spectra` | 16 / 16 |

The primary rubric saturated because the vanilla agent did more than expected.
It computed aggregate errors, compared train and held-out target support,
constructed approximate SMILES-neighborhood novelty, and reported that error
increased as held-out molecules became less similar to training molecules.

## Key Quantitative Findings

Both agents computed overall held-out metrics from raw predictions:

- MAE: 0.155118
- RMSE: 0.192916
- R2: 0.154806
- Mean prediction bias: -0.114425

Both agents also identified target-range failure. The training target range was
approximately 1.200122 to 1.564744, while held-out targets ranged from about
1.051251 to 1.963114. The model's predictions were compressed into roughly
1.215299 to 1.525844.

The vanilla agent found that held-out molecules above the training target
maximum had MAE 0.228160 and bias -0.228160. It also reported an approximate
nearest-training SMILES-similarity curve, where MAE rose from about 0.080 at
high similarity to about 0.219 in lower-similarity bins.

The `/spectra` agent produced a more explicit audit. It built approximate
novelty axes for exact identity, formula keys, rough skeleton keys, SMILES
Jaccard similarity, descriptor distance, descriptor-OOD count, and target shift.
It also computed an all-pairs train-eval similarity graph over 12,623,040 pairs
and showed edge density decreased monotonically as the Jaccard threshold
tightened from 0.2 to 0.8.

## Audit-Depth Comparison

The primary rubric was too coarse to separate the conditions. A post-hoc
audit-depth comparison gives a clearer picture:

| Dimension | Vanilla | `/spectra` |
| --- | ---: | ---: |
| Property graph specificity | 1 / 2 | 2 / 2 |
| Overlap validation | 1 / 2 | 2 / 2 |
| Multi-axis novelty | 2 / 2 | 2 / 2 |
| Performance-curve richness | 1 / 2 | 2 / 2 |
| Reusable audit structure | 1 / 2 | 2 / 2 |
| Total | 6 / 10 | 10 / 10 |

This secondary comparison should not be treated as the main benchmark score,
because it was added after seeing that the primary rubric saturated. It is still
useful for diagnosing what `/spectra` changed: not discovery of the failure
itself, but the completeness and reproducibility of the audit.

## Interpretation

The hard blind task supports a narrower, more defensible claim:

> `/spectra` helps agents turn raw model-evaluation artifacts into structured,
> reusable generalization audits.

It does not support the stronger claim:

> Vanilla agents cannot discover this molecular generalization failure.

A capable vanilla agent found the main result from the model, data, and raw
predictions alone. The evidence for `/spectra` is therefore about audit quality:
explicit novelty axes, property graphs, overlap validation, multiple
performance-vs-novelty curves, and a reusable report structure.

## Next Design Change

The next benchmark should reduce target-support leakage and increase task
replication:

1. Ask agents to design a prospective generalization audit from only the model
   and training data before revealing held-out labels.
2. Reveal predictions and labels only after the audit plan is fixed.
3. Run multiple anonymous model/data tasks, not one molecular task.
4. Score with a rubric that includes overlap validation and property-graph
   specificity from the beginning.
