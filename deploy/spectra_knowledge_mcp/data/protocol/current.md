# SPECTRA Generalizability Protocol

This server exposes SPECTRA guidance and prior findings as read-only knowledge.
It does not run audits, launch agents, call models, fetch datasets, or mutate
artifacts.

## Purpose

Treat SPECTRA as a spectral performance curve construction and validation
workflow. The primary product is model performance as prospective train-test or
pretraining-test similarity decreases, plus an explicit validity assessment.

For broad prompts such as "assess the generalizability of this model", first
extract the model paper's stated and implicit generalization claims and use
those claims to seed the hypothesis ledger. If the scope is beyond the paper,
use the paper as model context but search for or construct public/local datasets
that test important external generalization axes.

## Controller Loop

Use one persistent controller process for an audit. Dataset state,
target-model results, prospective features, the axis ledger, negative results,
and validation decisions should stay in one shared state. Distiller,
Investigator, Dataset Scout, Dataset Fetcher, and Auditor are phases of the
same loop, not separately routed agents unless a host client explicitly needs
that decomposition.

The loop is:

1. Distill the user question into model, dataset, scientific unit, task, metric,
   candidate prospective axes, and a concrete SPC plan.
2. Scout or fetch data if the available data cannot test the live hypothesis.
3. Construct similarity levels from prospective features only.
4. Verify measured train-test or pretraining-test similarity decreases.
5. Run a simple fixed baseline when labels exist.
6. Evaluate the target model only after the split contract is validated or
   explicitly marked exploratory.
7. Audit leakage, split size, similarity progression, baseline stability,
   confounding, metric direction, and post-hoc axis selection.
8. Route weak, invalid, exploratory, negative, or non-explanatory results back
   into prospective-axis discovery.
9. Stop only when a claim-valid explanatory/degradation SPC is supported, the
   user explicitly asks for a checkpoint, or a concrete hard blocker prevents
   confirmation.

## Roles

SPECTRA Distiller turns user generalizability questions into SPECTRA analyses.
It identifies or validates candidate similarity axes that could yield meaningful
SPCs, interprets results, distinguishes valid from exploratory axes, avoids
overclaiming, and routes negative or weak results back into discovery.

SPECTRA Investigator executes the protocol for a model, dataset, task, metric,
and similarity axis. It computes prospective similarities without target-model
errors, constructs at least three nontrivial split levels when feasible,
verifies decreasing similarity, runs fixed baselines when labels exist, then
evaluates the model of interest. Small bounded model runs are pilot probes, not
valid SPC evidence, unless they cover the full split or a predeclared adequately
powered stratified evaluation sample.

SPECTRA Dataset Scout finds datasets suitable for SPECTRA. It prioritizes
datasets with labels, metadata, prospective similarity features, enough examples
for multiple split levels, clear access, and low leakage risk.

SPECTRA Dataset Fetcher retrieves and packages datasets for SPECTRA. It retains
inputs, labels, metadata, and prospective features, handles duplicates and
missingness, and creates SPECTRA-ready artifacts. For very large pretraining
datasets, it should use bounded filtering or approximate retrieval rather than
exhaustive comparison. It must also record source provenance: data source URL
or repository, split/shard/filter details, unit of analysis, row or unit count
when known, local/cache path, download or access command, and license or access
constraints when relevant.

SPECTRA Auditor checks whether the SPC supports the claim. It looks for
target-error leakage, test-label leakage, tiny splits, non-decreasing
similarity, unstable baselines, confounding, post-hoc axis selection, pilot-only
target evaluation, and outcome-informed axis discovery. It marks analyses as
valid, weak, invalid, or exploratory only.

## Axis Rules

Do not use target-model errors, prediction-versus-reference errors, held-out
labels, target confidence derived from the evaluated model, or other
outcome-dependent quantities to define a similarity axis or split membership.
Post-hoc failure characterization can suggest hypotheses, but those hypotheses
are exploratory until frozen and confirmed.

Do not claim a SPECTRA split is valid unless measured cross-split overlap or
pretraining-test proximity decreases across the spectral parameters.

Failed or non-monotonic axes are findings. Report them, then try the next
scientifically plausible prospective axis or use already-computed
successes/failures to propose a new prospective hypothesis.

Select the strongest scientifically defensible, leakage-aware novelty axis, not
merely the most monotonic curve.

## Plausible Axis Confirmation

When SPECTRA finds a plausible axis from existing evaluated evidence, prior run
artifacts, pilots, or outcome-informed success/failure analysis, treat it as a
live hypothesis, not a claim-valid answer.

Required transition:

1. Record the source of the hypothesis as predeclared, pilot-derived,
   prior-artifact-derived, or outcome-informed exploratory.
2. Freeze the axis before any new target-model scoring: features, direction,
   thresholds or level rule, metric, unit, inclusion/exclusion criteria,
   deduplication rule, sample-size target, controls, and success criteria.
3. Construct or select a confirmation set that is new evidence relative to
   discovery whenever feasible.
4. Validate nontrivial levels, measured axis order, adequate counts, duplicate
   or cluster diagnostics, baseline availability, and confound coverage.
5. Score all predeclared levels in one target-model run when feasible.
6. If confirmation supports the degradation after controls, the axis may pass
   claim closure. If it fails, downgrade it, ledger the negative result, and
   continue discovery using combined successes and failures.

Do not stop when a plausible axis first appears in existing evidence.

## Runtime Policy

Use a cheap-first behavioral runtime policy. Before expensive fitting,
fine-tuning, all-pairs graph construction, large ANN indexing, or full-dataset
embedding, run the smallest leakage-aware behavioral slice that can answer the
next live hypothesis.

Start with schema/leakage checks, simple controls, cached or chunked
representations, and deterministic non-iterative probes such as nearest
centroid scores, mean-difference linear scores, kNN/prototype retrieval, or
closed-form small linear baselines. Escalate only when the cheap probe is
inconclusive or the stronger deployment claim requires it.

Before launching a heavy step, write the time/resource budget, success
criterion, timeout/fallback condition, and cheaper fallback. After one timeout
or runtime failure from a solver family, switch to the fallback or a smaller
slice instead of retrying the same slow method repeatedly.

Maintain one reusable candidate/prospective-feature table. Prefer
manifest-first expansion, deduplicate or cluster near-duplicates before
expensive evaluation, avoid blind rejection sampling against external APIs, and
keep a persistent target-model evaluator loaded when feasible.

## Provenance Contract

Every paper-facing or MCP-published SPECTRA finding must include normalized
provenance. A stored finding is incomplete unless an agent can answer:

1. Where did the model code come from?
2. Where did the model weights, checkpoint, or official precomputed scored
   outputs come from?
3. How were those weights or scored outputs downloaded, cached, or loaded?
4. Where did each dataset, split, shard, or candidate panel come from?
5. How was each dataset downloaded, filtered, deduplicated, or bounded?
6. Which metadata resources defined similarity axes, controls, labels, or
   claim boundaries?
7. Which local paths and MCP artifact ids support the record?
8. What exact gaps remain, such as missing checkpoint revisions or legacy
   cache hashes?

For hosted knowledge-server findings, add or update the normalized provenance
record and validate it before publishing. If fresh inference was not run, say
so directly and record the official scored-result repositories/files instead
of pretending model weights were evaluated locally. If a legacy run lacks exact
download hashes or cache revisions, mark the gap explicitly and route to
backfill before claiming full reproducibility.

If a paper-facing or reviewer-facing claim requires re-running the analysis
from per-target or per-example rows, the raw scored tables must be published as
normal HTTPS downloads with byte size, row count, and SHA-256 checksum. MCP
artifact preview tools may truncate large files and should not be treated as a
bulk transport mechanism. Hosted knowledge-server entries should expose such
files through the download manifest and direct agents to `list_spectra_downloads`
or `get_spectra_download`.

## Closure Gate

Before final synthesis, the primary axis must be prospective, frozen before
target scoring, measured in the intended order, adequately powered, densified
enough to characterize trend shape, stable under expansion or explicitly
reported as unstable, confirmed on new or explicitly adequate evidence if
discovered from existing results, not materially explained by known confounders,
supported by fixed baselines when labels exist, and accompanied by model,
weights/checkpoint or official-score, dataset, metadata, download, cache, and
known-gap provenance.

A coarse, localized, weak, source-confounded, proxy-only, shape-unstable, or
discovery-only curve does not pass closure merely because it has three split
levels and a visible signal.

In broad generalizability mode, a valid negative or non-explanatory SPC is not
terminal. It only closes an explicitly axis-specific question. Otherwise, ledger
it as a negative result and continue to the next prospective axis.
