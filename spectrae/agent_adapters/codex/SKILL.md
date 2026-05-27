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
- Do not pass `--max-rounds` unless the user asks for a bounded debug run.

## Reporting Back

After a run finishes, report the output root, terminal role/status, final
recommendation or finding, key evidence numbers, and main artifacts.

If a run fails, report the concrete failing command or role log and whether the
failure happened before or after SPECTRA produced usable evidence.
