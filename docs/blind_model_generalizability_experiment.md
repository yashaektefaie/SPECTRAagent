# Blind Model-Generalizability Experiment

## Question

The stronger hypothesis is not that agents can reproduce BOOM after reading the
BOOM paper. The stronger hypothesis is:

> Given only a trained scientific model and evaluation artifacts, a `/spectra`
> agent is more likely than a vanilla agent to autonomously discover BOOM-like
> generalization failure modes.

## Blind Bundle

Agent-visible files:

- `/ewsc/yektefai/spectra_assets/blind_molecular_model_eval/agent_visible/train.csv`
- `/ewsc/yektefai/spectra_assets/blind_molecular_model_eval/agent_visible/eval_predictions.csv`
- `/ewsc/yektefai/spectra_assets/blind_molecular_model_eval/agent_visible/metadata.json`

Grading-only file:

- `/ewsc/yektefai/spectra_assets/blind_molecular_model_eval/grading_only/hidden_eval_labels.csv`

The agent-visible files do not mention BOOM and do not expose ID/OOD labels.
They provide training molecules, held-out predictions, errors, and maximum
train-set Morgan Tanimoto similarity.

## Conditions

Vanilla:

- Asked to evaluate whether the model generalizes.
- Receives no `/spectra` terminology.

`/spectra`:

- Asked to evaluate whether the model generalizes using `/spectra`.
- Must identify scientific unit, novelty axes, property graphs, overlap
  validation, and performance-vs-novelty behavior.

Prompt files:

- `experiments/blind_model_eval/prompt_vanilla.md`
- `experiments/blind_model_eval/prompt_spectra.md`

## Expected Discovery

The target discovery is BOOM-like:

1. The model has much larger errors on a subset of held-out molecules.
2. Error is associated with distributional novelty, especially target-value
   extremeness and/or low train-test molecular similarity.
3. A useful audit reports performance as a function of measured novelty rather
   than only an aggregate held-out metric.

This setup no longer asks the agent to reproduce BOOM from the BOOM paper. It
asks the agent to infer BOOM-like generalization failure from model artifacts.
