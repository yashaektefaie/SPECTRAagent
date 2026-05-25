# BOOM Pilot: With vs Without `/spectra`

Date: 2026-05-07

## Setup

Target:

- BOOM: Benchmarking Out-Of-distribution Molecular Property Predictions of Machine Learning Models
- Paper: https://openreview.net/forum?id=QoBxQrvFRd
- arXiv: https://arxiv.org/abs/2505.01912
- Repo: https://github.com/FLASK-LLNL/BOOM

Fetched local assets:

- `/ewsc/yektefai/spectra_assets/boom_pilot/papers/boom.pdf`
- `/ewsc/yektefai/spectra_assets/boom_pilot/repos/boom`

Prompt used for both conditions:

> Read this paper and associated repo. Reproduce one core generalization finding. Then extend the analysis to evaluate whether model performance degrades under scientifically meaningful train-test novelty.

Conditions:

- Vanilla: no `/spectra` capsule or protocol guidance.
- `/spectra`: received the BOOM capsule, novelty axes, required artifacts, and audit protocol.

This was a constrained no-retraining pilot. The agents produced audit cards from paper/repo inspection and source/table evidence. They did not download full BOOM datasets, train models, generate SPECTRA splits, or compute measured spectral performance curves.

## Rubric

Each dimension was scored 0-2:

- `axis_discovery`
- `split_construction`
- `curve_generation`
- `metric_reporting`
- `finding_recovery`
- `new_insight`
- `citation_report_quality`

## Scores

| Condition | Axis | Split | Curve | Metrics | Finding | Insight | Report | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla | 1 | 1 | 0 | 1 | 2 | 1 | 2 | 8 / 14 |
| `/spectra` | 2 | 1 | 1 | 1 | 2 | 2 | 2 | 11 / 14 |

## Result

The `/spectra` condition outperformed vanilla on audit design quality. It produced a more explicit molecule-level generalization audit with:

- chemically meaningful novelty axes,
- property-graph definitions,
- overlap-validation statistics,
- required audit artifacts,
- an explicit AUSPC plan,
- a falsifiable expected outcome.

The vanilla condition was stronger than expected: it recovered BOOM's core qualitative finding and proposed several structural novelty axes. However, it was less explicit about the reusable audit protocol and did not generate the same curve/AUSPC-ready structure.

## Interpretation

This is moderate pilot evidence for the benchmark hypothesis:

> `/spectra`-equipped agents more reliably convert static benchmark papers into executable generalization audits.

It is not evidence that SPECTRA improves model generalizability, and it is not yet evidence from a completed computational reproduction. The current evidence is about agent audit quality under a constrained paper/repo-inspection task.

## Next Required Run

To turn this into strong evidence, the next benchmark run needs:

1. A runnable BOOM task with processed molecule/property data.
2. RDKit-based Morgan fingerprint and scaffold graph construction.
3. Measured cross-split overlap across spectral parameters.
4. At least one lightweight baseline model evaluated on each split.
5. Numeric performance-over-overlap curves and AUSPC.
6. The same with-vs-without agent comparison repeated across BOOM, DART-Eval, NABench, and Systema.
