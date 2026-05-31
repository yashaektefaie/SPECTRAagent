---
name: spectra
description: Run SPECTRA agent-native generalization, applicability-domain, and spectral-curve audits. Use when the user invokes "/spectra", asks whether a model generalizes, asks whether a model should be used for a dataset/task, or asks to compute a SPECTRA curve from predictions and similarity.
---

# SPECTRA

Treat `/spectra ...` as an instruction to use the installed SPECTRA tools, not
merely to explain the framework.

## Preferred Commands

Run a human-friendly model/dataset audit:

```bash
spectra ask "<question>" \
  --paper /path/to/model_or_reference_paper.pdf \
  --model /path/to/model_or_checkpoint \
  --dataset /path/to/raw_dataset_or_directory
```

For direct autonomous agent runs, use one SPECTRA Controller session:

```bash
spectra agent run ... --execute
```

Run a core SPECTRA curve from predictions and a measured axis:

```bash
spectra audit \
  --eval eval_predictions.csv \
  --out spectra_audit \
  --target-col y_true \
  --pred-col y_pred \
  --axis-col max_train_similarity \
  --axis-type similarity
```

Run diagnostics when setup is uncertain:

```bash
spectra-doctor
```

## Behavior

- Let SPECTRA infer domain, scientific unit, similarity hypotheses, controls,
  split strategy, and output location unless the user gives explicit
  constraints.
- Preserve user constraints in `--constraints`, especially isolation rules,
  compute limits, and prohibited prior artifacts.
- Use `SPECTRA_SCRATCH_ROOT` when set. Avoid `/tmp` for large caches, downloads,
  generated datasets, model outputs, or temporary files.
- Do not use prior SPECTRA run artifacts as evidence unless the user explicitly
  allows prior memory or those artifacts are part of the current session.
- Do not stop because a weak bounded checkpoint is available or because no valid
  axis was found after a search budget. Negative, weak, non-monotonic,
  non-explanatory, coarse, localized, or shape-unstable axes should trigger
  outcome-informed prospective-axis discovery or densification of the same live
  axis, then frozen-axis confirmation.
- Distinguish split validity from claim closure. In broad generalizability mode,
  a valid negative or non-explanatory SPC is not terminal merely because the
  split contract is valid. It only closes an explicitly axis-specific question.
  Otherwise, ledger it as a negative result and continue to the next prospective
  axis.
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

## Reporting Back

After a run finishes, report the output root, terminal role/status, final
recommendation or finding, key evidence numbers, and main artifacts.

If a run fails, report the concrete failing command or role log and whether the
failure happened before or after SPECTRA produced usable evidence.
