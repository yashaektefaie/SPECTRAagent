# SPECTRA Value Ablation

Date: 2026-05-11

## Question

The hard-blind BOOM-style pilot showed that a capable vanilla agent can discover
the key molecular generalization failure from model/data/prediction artifacts
alone. That raises the central question:

> If `/spectra` only tells the agent to evaluate performance as a function of
> distance from the training set, is it more than a prompt?

The right answer is empirical. We should compare `/spectra` not only against a
broad prompt, but also against an explicit distance-from-train prompt.

## Three Conditions

1. **Broad generalizability**

   The agent is asked: "Evaluate whether the model generalizes."

   This tests the baseline behavior when generalizability is underspecified.

2. **Explicit distance-from-train**

   The agent is asked: "Evaluate how performance changes as held-out examples
   become more distant from the training set."

   This controls for the central SPECTRA idea being stated directly in ordinary
   language.

3. **Full `/spectra`**

   The agent is asked to execute the `/spectra` protocol: define scientific
   units, construct novelty axes, build or approximate property graphs, validate
   overlap, report performance-vs-novelty curves, and return a reusable audit
   report.

   This tests whether `/spectra` adds operational structure beyond the idea.

The prompts and rubric are in `experiments/spectra_value_ablation/`.
Results from the first molecular run are summarized in
`docs/spectra_value_ablation_results.md` and
`experiments/spectra_value_ablation/scores.json`.

## What Would Make `/spectra` Nontrivial?

`/spectra` is not nontrivial merely because it tells the agent to make a curve.
The distance-from-train prompt already does that. `/spectra` becomes
infrastructure if it reliably adds:

- domain-valid novelty axes,
- property or similarity graph construction,
- overlap validation,
- multiple scientific axes rather than one convenient proxy,
- rejection or caveating of invalid axes,
- standardized audit artifacts,
- cross-domain portability.

The key comparison is therefore condition 2 versus condition 3.

## Primary Scoring

Each condition should first be scored on basic competence:

- scientific unit,
- aggregate error,
- evidence-backed failure mode.

Those dimensions may saturate. The more important scoring block is spectral
audit quality:

- distance axis defined,
- domain-axis quality,
- property-graph specificity,
- overlap validation,
- performance curve,
- multi-axis audit,
- invalid-axis handling,
- artifact completeness.

Each dimension is scored 0-2. The full machine-readable rubric is
`experiments/spectra_value_ablation/rubric.json`.

## Interpretation

If broad generalizability loses to both distance-from-train and `/spectra`, that
only shows that broad prompting is underspecified.

If distance-from-train matches `/spectra`, then `/spectra` is mostly packaging
for this task.

If `/spectra` beats distance-from-train on overlap validation, property-graph
specificity, multi-axis audit quality, invalid-axis handling, and reusable
artifacts, then `/spectra` is doing useful work as an executable audit standard.

## Recommended Next Run

Do not stop with the BOOM-style molecular task. Run the same three-condition
ablation on several anonymous tasks:

- molecules,
- regulatory DNA,
- nucleotide fitness,
- perturbation response.

The strongest claim would be:

> Across domains, `/spectra` does not merely remind agents to measure distance
> from train. It makes them execute a validated, domain-aware spectral audit
> with reusable artifacts more reliably than either broad generalizability
> prompts or explicit distance-from-train prompts.
