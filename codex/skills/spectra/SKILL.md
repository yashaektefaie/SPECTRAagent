---
name: spectra
description: Run SPECTRA agent-native generalization and applicability-domain audits. Use when the user invokes "/spectra", asks to use SPECTRA, asks to construct SPECTRA splits, or asks Codex to assess whether a model generalizes.
---

# SPECTRA

Treat `/spectra ...` as an instruction to run the installed SPECTRA workflow, not merely to explain the framework.

## Runner

Use the installed CLI:

```bash
spectra ask "<question>" \
  --paper /path/to/model_or_reference_paper.pdf \
  --model /path/to/model_or_checkpoint \
  --dataset /path/to/raw_dataset_or_directory \
  --scratch-root /ewsc/$USER/spectra_runs
```

Equivalent bundled helper:

```bash
codex/skills/spectra/scripts/spectra-ask "<question>" \
  --paper /path/to/paper.pdf --model /path/to/model --dataset /path/to/data
```

If the user provides only prose for the model or dataset, use `--model-description` and `--dataset-description` instead of inventing paths. If a required artifact is absent and cannot be discovered locally, ask for the smallest missing item.

Do not write SPECTRA caches, downloads, generated datasets, model outputs, or temporary files under `/tmp`; use the session root or `/ewsc/$USER/spectra_runs`.

## Mode Routing

Prefer `spectra ask` for human-friendly `/spectra` requests. The CLI routes automatically:

- Split-only requests such as "construct SPECTRA splits" or "generate SPECTRA splits" use focused split construction and set `uses_general_audit_loop=false`.
- Model/paper/checkpoint generalization questions use the autonomous audit loop.
- Use `--ask-mode splits` or `--ask-mode audit` only when the user explicitly wants to override the automatic route.

For split construction, SPECTRA must:

1. Choose and record a similarity definition, preferring `spectra similarity-definitions suggest`.
2. Choose and record a pairwise computation strategy, preferring `spectra similarity-computation suggest`.
3. Compute pairwise similarities or a thresholded property graph.
4. Construct prospective split candidates without using target-model errors.
5. Sweep at least three spectral parameters when feasible.
6. Verify train-test similarity decreases across split levels.
7. If labels are available, train/test a fixed simple baseline and report whether performance degrades as train-test similarity decreases.

For molecular CSV datasets with `SMILES` and binary `Y`, the native focused constructor uses RDKit canonical SMILES, Morgan fingerprints, Tanimoto similarity, thresholded connected components, and logistic-regression baseline validation.

## Reporting Back

After the run finishes, report:

- output root,
- terminal role/status or split-constructor status,
- whether the general audit loop was used,
- final recommendation or finding,
- key evidence numbers,
- main report/checkpoint artifacts.

If the run fails, report the concrete failing command or role log and whether the failure happened before or after SPECTRA produced usable evidence.
