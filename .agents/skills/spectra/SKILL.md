---
name: spectra
description: Run SPECTRA spectral performance curve construction and validation. Use when the user invokes "/spectra", asks whether a model generalizes, asks whether a model should be used for a dataset/task, or asks to compute a SPECTRA curve from predictions and prospective similarity.
---

# SPECTRA

Treat `/spectra ...` as an instruction to build and validate a spectral
performance curve (SPC), not merely to explain the framework.

## Preferred Local Commands

For human-friendly model/dataset questions:

```bash
spectra ask "<question>" \
  --paper /path/to/model_or_reference_paper.pdf \
  --model /path/to/model_or_checkpoint \
  --dataset /path/to/raw_dataset_or_directory
```

For core SPECTRA curve mode from predictions plus an existing prospective axis:

```bash
spectra audit \
  --eval eval_predictions.csv \
  --out spectra_audit \
  --target-col y_true \
  --pred-col y_pred \
  --axis-col max_train_similarity \
  --axis-type similarity
```

For MCP-backed agents, register the local server:

```bash
spectra-mcp serve --transport stdio
```

Run diagnostics first when setup is uncertain:

```bash
spectra-doctor
```

## Behavior

- The primary output is model performance as prospective train-test or
  pretraining-test similarity decreases, plus a validity decision.
- Distinguish split validity from claim closure. A split-valid SPC has a
  prospective axis, frozen split membership, decreasing measured similarity, and
  adequate evaluation. A claim-valid SPC additionally shows an interpretable
  degradation, threshold, localized failure, or robust boundary that answers the
  user's generalizability question.
- Let SPECTRA infer domain, scientific unit, prospective similarity axes,
  controls, split strategy, and output location unless the user gives explicit
  constraints.
- Preserve user constraints in `--constraints`, especially isolation rules,
  compute limits, and prohibited prior artifacts.
- Use `SPECTRA_SCRATCH_ROOT` when set. Avoid `/tmp` for large caches, downloads,
  generated datasets, model outputs, or temporary files.
- Do not use prior SPECTRA run artifacts as evidence unless the user explicitly
  allows prior memory or those artifacts are part of the current session.
- Prefer `spectra ask` or `spectra agent run --execute` for autonomous audits.
  These prepare or launch one SPECTRA Controller session. Distiller, Scout,
  Fetcher, Investigator, Auditor, and synthesis are internal phases of that
  same Codex session, not separate Python-routed agents.
- Do not use target-model errors, prediction/reference errors, held-out labels,
  or target confidence derived from the evaluated model to define the similarity
  axis or split membership.
- Require measured decreasing similarity across split levels before trusting
  model metrics.
- Run a simple fixed baseline when labels exist, then evaluate the target model.
- Treat small bounded target-model runs as pilot probes, not valid SPC evidence,
  unless they cover the full split or a predeclared adequately powered
  stratified evaluation sample. About 10 examples per split is pilot-only.
- If a pilot shows monotone, localized, or practically meaningful degradation,
  expand the same frozen prospective axis before final synthesis.
- A promising axis is not complete after one coarse confirmation. Densify the
  same frozen SPC by increasing split resolution, adding examples per level, and
  sampling sparse or ambiguous regions until the trend shape is stable or the
  axis is invalidated.
- Maintain one reusable candidate/prospective-feature table for the session.
  Prefer manifest-first expansion, deduplicate or cluster near-duplicates before
  expensive evaluation, and avoid blind rejection sampling against external APIs
  when a candidate table can be built first.
- When target-model evaluation is needed repeatedly, keep a persistent evaluator
  or model process loaded when feasible and feed it queued evaluation pools
  instead of launching separate scripts that reload the model for each axis.
- If a prospective axis shows no pattern, record that negative result. You may
  use already-computed successes/failures to discover candidate prospective
  similarity definitions, but that discovery is exploratory until the axis is
  frozen and confirmed on fresh or expanded target-model evaluations.
- After any failed confirmation, invalidated axis, weak axis, negative SPC, or
  non-explanatory SPC, run all-scored-evidence axis discovery before synthesis:
  use every target-model-scored example accumulated in the session, including
  successes, failures, weak curves, and failed confirmation panels, to identify
  the best prospectively definable axis or feature composite that consistently
  explains degradation. Reject axes that reverse direction across meaningful
  evidence slices, depend on cherry-picked panels, require reference answers, or
  belong to a family already falsified by fresh confirmation. Do not require
  each historical panel to have been purpose-built for the new axis or to meet
  the final confirmation effect threshold; those panels are discovery evidence,
  not claim closure.
- When an axis is found from existing evaluated evidence, prior run artifacts,
  pilot slices, or outcome-informed success/failure mining, treat it as a live
  hypothesis, not a valid claim. First freeze the axis definition before any new
  target-model scoring: feature(s), direction, thresholds or level rule, metric,
  unit, inclusion/exclusion criteria, deduplication rule, sample-size target,
  controls, and success criteria. Then construct or select a confirmation panel
  that is new evidence relative to discovery whenever feasible, excluding prior
  target IDs, exact inputs, near-duplicates, or leakage-linked records unless the
  claim boundary explicitly allows within-pool expansion. Validate split order,
  counts, duplicates, baselines, and confound coverage before scoring. Only if
  the frozen axis still supports the degradation/boundary on that confirmation
  evidence may it pass claim closure. If it fails, downgrade it, ledger the
  negative confirmation, and continue prospective-axis discovery using the
  combined successes/failures.
- If the current evaluated table or unused candidate pool cannot validly confirm
  the best all-scored-evidence hypothesis, route to dataset construction or
  acquisition instead of stopping. Freeze the hypothesis, then search for, fetch,
  or construct data that spans the frozen axis with adequate levels and controls.
  If no current prospective feature explains the accumulated successes/failures,
  derive or acquire additional prospective features such as homology/family
  distance, MSA/search depth, fold/topology class, provenance, taxonomy,
  disorder/membrane annotations, or measured pretraining proximity, then repeat
  all-scored-evidence discovery.
- Do not stop because a weak bounded checkpoint is available or because no valid
  axis was found after a search budget. Weak, negative, non-monotonic,
  non-explanatory, coarse, localized, or shape-unstable axes should trigger
  outcome-informed prospective-axis discovery or densification of the same live
  axis, then frozen-axis confirmation.
- In broad generalizability mode, a valid negative or non-explanatory SPC is not
  terminal merely because the split contract is valid. It only closes an
  explicitly axis-specific question. Otherwise, ledger it as a negative result
  and continue to the next prospective axis.
- Before final synthesis, apply a closure gate: the primary axis must be
  prospective, frozen before target scoring, measured in the intended order,
  adequately powered, densified enough to characterize trend shape, stable under
  expansion or explicitly reported as unstable, confirmed on new or explicitly
  adequate evidence if discovered from existing results or outcome mining, not
  materially explained by known confounders, and supported by fixed baselines
  when labels exist. A coarse, localized, weak, source-confounded, proxy-only,
  shape-unstable, or discovery-only curve does not pass closure merely because
  it has three split levels and a visible signal.
- If all-scored-evidence success/failure analysis identifies a prospective
  follow-up hypothesis, that live hypothesis blocks final synthesis until it is
  frozen and confirmed. If the hypothesis is not confirmable in the current
  pool, construct or acquire data that can test it. Current-pool exhaustion is
  not a hard blocker.
- Treat the Auditor decision as required: `valid`, `weak`, `invalid`, or
  `exploratory`.

## Reporting Back

After a run finishes, report:

- output root,
- terminal role and status,
- final recommendation or finding,
- key evidence numbers,
- main report/checkpoint artifacts.

If a run fails, report the concrete failing command or role log and whether the
failure happened before or after SPECTRA produced usable evidence.
