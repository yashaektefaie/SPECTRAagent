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
  expansion or explicitly reported as unstable, not materially explained by known
  confounders, and supported by fixed baselines when labels exist. A coarse,
  localized, weak, source-confounded, proxy-only, or shape-unstable curve does
  not pass closure merely because it has three split levels and a visible signal.
- If success/failure analysis identifies an executable prospective follow-up
  hypothesis, that live hypothesis blocks final synthesis until it is frozen and
  confirmed, the user requests a checkpoint, or a concrete hard blocker prevents
  execution.
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
