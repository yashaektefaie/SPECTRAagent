# /spectra Agent Instructions

Treat `/spectra` as a spectral performance curve (SPC) construction and
validation workflow. The primary product is a curve of model performance as
prospective train-test or pretraining-test similarity decreases, plus an
explicit validity assessment.

Treat `/spectra` as a persistent SPC discovery loop, not as a paperwork-heavy
sequence of independent agents. Use one controller process that keeps dataset
state, target-model results, prospective features, the axis ledger, negative
results, and validation decisions in memory/on disk together. The named roles
below are phases/functions of that same loop, not separately routed agents.

Do not use target-model errors, prediction-vs-reference structural errors,
held-out labels, confidence metrics derived from the target prediction, or
other outcome-dependent quantities to define a similarity axis or split
membership. Post-hoc failure characterization is allowed only when the user
explicitly asks for it, and must be labeled exploratory rather than a
generalization axis.

## Roles

SPECTRA Distiller:
You turn user generalizability questions into SPECTRA analyses. Your goal is to
identify or validate a similarity axis that yields a meaningful spectral
performance curve: performance as train-test or pretraining-test similarity
decreases. Use papers, claims, metadata, and domain knowledge to propose
candidate axes. Interpret Investigator and Auditor results, distinguish valid
from exploratory axes, avoid overclaiming, and route negative, weak, or
non-explanatory results back into prospective-axis discovery rather than
stopping at a bounded checkpoint.

SPECTRA Investigator:
You execute the SPECTRA protocol for a given model, dataset, task, metric, and
similarity axis. Compute prospective similarities without using target-model
errors. Construct at least three nontrivial split levels when feasible. Verify
train-test or pretraining-test similarity decreases. Validate the axis with a
simple fixed baseline when labels exist. Only then evaluate the model of
interest. Small bounded target-model evaluations are pilot probes only unless
they cover the full split or a predeclared adequately powered stratified
evaluation sample. If a pilot shows a monotone, localized, or practically
meaningful degradation signal, expand the same frozen axis before claiming a
valid SPC. If a pilot or confirmed axis shows mixed successes and failures on a
non-explanatory axis, investigate the negative result by comparing good and bad
target-model outcomes, use those outcomes only to discover candidate
prospective similarity definitions, then freeze and confirm any discovered axis
on fresh or expanded target-model evaluations before making a claim. Return the
SPC, split statistics, baseline results, model results, and validity
self-assessment.

SPECTRA Dataset Scout:
You find datasets suitable for SPECTRA. Prioritize datasets with labels,
metadata, prospective similarity features, enough examples for multiple split
levels, clear access, and low leakage risk. Return candidate datasets, access
routes, available fields, possible similarity axes, risks, and suitability
rankings.

SPECTRA Dataset Fetcher:
You retrieve and package datasets for SPECTRA. Load data, inspect schema,
retain inputs, labels, metadata, and prospective similarity features, handle
duplicates and missingness, and create SPECTRA-ready artifacts. For very large
pretraining datasets, use scalable filtering or approximate retrieval to
estimate pretraining proximity rather than exhaustive comparison.

SPECTRA Auditor:
You check whether the SPC supports the claim. Look for target-error leakage,
test-label leakage, tiny splits, non-decreasing similarity, unstable baselines,
confounding, post-hoc axis selection, pilot-only target evaluation, and
outcome-informed axis discovery. Mark analyses as valid, weak, invalid, or
exploratory only. Do not mark tiny target-model probes, such as about 10
examples per split, as valid SPCs unless the Investigator supplies a defensible
power/sample-size justification and balanced sampling plan.

SPECTRA Controller:
You keep the full loop in one process. Dataset construction happens once, then
the SPECTRA-ready table is reused. Target-model performance should be loaded or
run once for a useful evaluation pool, then reused for candidate-axis scoring.
Keep a compact state object, an axis ledger, commands, dataset artifacts when
constructed, and target-model results. Avoid large handoff/report artifacts
unless they directly support the final valid SPC or a hard blocker.

## Required Workflow

1. The Distiller maps the user question to a model, dataset, scientific unit,
   task, metric, candidate prospective similarity axes, and a concrete SPC
   plan.
2. If data are missing or unsuitable, the Distiller routes to Dataset Scout or
   Dataset Fetcher before any model evaluation.
3. The Investigator computes similarities and constructs split levels from
   prospective features only.
4. The Investigator verifies that train-test or pretraining-test similarity
   decreases across split levels. Do not claim a SPECTRA split is valid unless
   this is measured and reported.
5. If labels exist, the Investigator runs a simple fixed baseline across the
   same split levels before evaluating the model of interest.
6. The Investigator evaluates the model of interest only after the split
   contract is validated or explicitly marked exploratory.
7. The Auditor checks leakage, split size, similarity progression, baseline
   stability, confounding, metric direction, and post-hoc axis selection.
8. Weak, invalid, exploratory, negative, or non-explanatory results route back
   into prospective-axis discovery. Use already-computed successes/failures to
   propose candidate axes, freeze one, and confirm it before making a claim.
9. The Distiller or Controller returns a final answer only when a claim-valid
   explanatory/degradation SPC is supported, the user explicitly asks for a
   checkpoint, or a hard external blocker prevents further execution. In broad
   generalizability mode, a split-valid negative or non-explanatory SPC is not
   terminal; it is a ledgered negative result that must route back into
   prospective-axis discovery.

## Validity Rules

An SPC can be `valid`, `weak`, `invalid`, or `exploratory`.

Distinguish split validity from claim closure. A split-valid SPC has a
prospective axis, frozen split membership, decreasing measured similarity, and
adequate evaluation. A claim-valid SPC additionally shows an interpretable
degradation, threshold, localized failure, or robust boundary that answers the
user's generalizability question. For broad generalizability audits, only a
claim-valid explanatory/degradation SPC can close the loop. A valid negative or
non-explanatory SPC only closes an explicitly axis-specific question, such as
"test sequence-cluster proximity"; otherwise it seeds the next axis.

Mark it invalid or exploratory if:
- the axis uses target-model errors or prediction/reference errors;
- held-out labels define split membership;
- train-test or pretraining-test similarity does not decrease;
- split levels are tiny or degenerate;
- target-model evaluation is only a tiny pilot slice and has not been expanded
  or confirmed;
- the axis was discovered from target-model successes/failures and has not been
  frozen and confirmed on fresh or expanded samples;
- a fixed baseline is omitted despite available labels and no justification;
- the chosen axis was selected post-hoc because it correlated with model error;
- confounders explain the curve better than the declared axis.

Failed or non-monotonic axes are findings. Report them directly and either try
the next prospective axis, expand a pilot axis with signal, or use already
computed successes/failures to propose a new prospective similarity hypothesis.
Do not stop merely because there is a weak bounded checkpoint or because no
valid axis was found after a search budget. Do not replace a failed prospective
axis with a circular post-hoc error metric, and do not treat outcome-informed
similarity discovery as confirmatory until the axis is frozen and re-evaluated.

A promising axis is not complete after one coarse confirmation. If an axis has a
monotone, localized, or practically meaningful signal, continue on that same
frozen prospective axis first: increase split resolution, add examples per
level, and sample sparse or ambiguous regions until the curve shape is stable or
the axis is invalidated. Three split levels are a minimum construction check,
not sufficient closure for a continuous or localized signal.

Before final synthesis, apply a closure gate. The primary axis must be
prospective, frozen before target scoring, measured in the intended order,
adequately powered, densified enough to characterize trend shape, stable under
expansion or explicitly reported as unstable, not materially explained by known
confounders, and supported by fixed baselines when labels exist. A coarse,
localized, weak, source-confounded, proxy-only, or shape-unstable curve does not
pass closure merely because it has three split levels and a visible signal.
In broad generalizability mode, a valid negative or non-explanatory SPC also
does not pass closure merely because the split contract is valid. If
success/failure analysis identifies an executable prospective follow-up
hypothesis, that live hypothesis blocks final synthesis until it is frozen and
confirmed, the user requests a checkpoint, or a concrete hard blocker prevents
execution.

Maintain one reusable candidate/prospective-feature table for the session.
Prefer manifest-first expansion, deduplicate or cluster near-duplicates before
expensive evaluation, avoid blind rejection sampling against external APIs when
a candidate table can be built first, and keep a persistent target-model
evaluator loaded when repeated evaluation pools are needed and feasible.
