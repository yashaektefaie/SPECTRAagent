# SPECTRA Value Ablation Results

Date: 2026-05-11

## Setup

We ran the three-condition ablation on the hard-blind molecular model bundle.
All agents received the same visible artifacts:

- trained random-forest molecular model,
- training molecules and targets,
- held-out `y_true` and `y_pred`,
- metadata.

No condition was given split labels, precomputed errors, precomputed Tanimoto
similarities, BOOM paper/repository context, or external benchmark context.

The three prompts were:

1. broad generalizability,
2. explicit distance-from-train,
3. full `/spectra` protocol.

## Scores

| Condition | Baseline | Spectral Audit | Total |
| --- | ---: | ---: | ---: |
| Broad generalizability | 6 / 6 | 13 / 16 | 19 / 22 |
| Distance from train | 6 / 6 | 13 / 16 | 19 / 22 |
| `/spectra` protocol | 6 / 6 | 16 / 16 | 22 / 22 |

The basic dimensions saturated. Every condition identified the scientific unit,
computed aggregate error, and found an evidence-backed failure mode.

## What Happened

The broad prompt was stronger than expected. It loaded the model, reproduced
the predictions, computed Morgan Tanimoto nearest-neighbor similarity, measured
Murcko scaffold overlap, checked descriptors, and reported error by similarity
bin. This means the broad condition was already capable of finding the main
spectral generalization failure in this molecular task.

The explicit distance prompt produced the cleanest single-axis control result.
It defined Morgan Tanimoto distance from train, validated distance quintiles,
and reported a monotonic performance degradation:

- closest quintile MAE: 0.0805,
- farthest quintile MAE: 0.2260,
- closest quintile RMSE: 0.1055,
- farthest quintile RMSE: 0.2582.

This shows that simply saying "measure performance as a function of distance
from train" is enough to recover the core curve here.

The `/spectra` prompt produced the most complete audit. It reported:

- Morgan fingerprint similarity,
- scaffold and generic-skeleton novelty,
- physicochemical descriptor distance,
- functional-group profile distance,
- target-support shift,
- explicit property graphs for each axis,
- threshold-level overlap validation,
- performance curves over multiple novelty axes,
- a caveat that scaffold novelty alone was not sufficient in this split.

## Interpretation

This run does not support the claim that `/spectra` is needed to make agents
discover the core generalization failure. Broad and distance-prompted agents
both found it.

It does support a narrower claim:

> `/spectra` improves protocol completeness beyond a direct distance-from-train
> prompt in this molecular run.

The key difference is not discovery of the concept. The difference is audit
discipline: multiple axes, graph definitions, overlap validation, invalid-axis
handling, and reusable report structure.

## Implication

For this molecular task, `/spectra` is not justified as a magic discovery tool.
It is justified, if at all, as a standardization layer.

The next experiment should be harder and more diagnostic:

1. Run multiple anonymous tasks, not one molecule benchmark.
2. Add domains where the right distance is less obvious than Morgan Tanimoto.
3. Require prospective audit design before revealing `y_true`.
4. Score whether the distance prompt and `/spectra` both choose valid domain
   axes before seeing labels.
5. Treat `/spectra` as valuable only if it beats the distance prompt on
   validation and artifact quality across tasks.
