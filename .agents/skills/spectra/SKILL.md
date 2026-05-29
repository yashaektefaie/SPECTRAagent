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
- Let SPECTRA infer domain, scientific unit, prospective similarity axes,
  controls, split strategy, and output location unless the user gives explicit
  constraints.
- Preserve user constraints in `--constraints`, especially isolation rules,
  compute limits, and prohibited prior artifacts.
- Use `SPECTRA_SCRATCH_ROOT` when set. Avoid `/tmp` for large caches, downloads,
  generated datasets, model outputs, or temporary files.
- Do not use prior SPECTRA run artifacts as evidence unless the user explicitly
  allows prior memory or those artifacts are part of the current session.
- Do not pass `--max-rounds` unless the user asks for a bounded debug run.
- Do not use target-model errors, prediction/reference errors, held-out labels,
  or target confidence derived from the evaluated model to define the similarity
  axis or split membership.
- Require measured decreasing similarity across split levels before trusting
  model metrics.
- Run a simple fixed baseline when labels exist, then evaluate the target model.
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
