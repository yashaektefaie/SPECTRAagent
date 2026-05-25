# Hard Blind Model-Generalizability Experiment

## Goal

This version removes the scaffolding that made the first blind task too easy.
Agents receive only:

- the trained model,
- training data,
- raw held-out labels and predictions.

They do not receive:

- ID/OOD labels,
- precomputed absolute or squared errors,
- precomputed train-test Tanimoto similarity,
- benchmark paper context.

## Agent-Visible Bundle

- `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/model.pkl`
- `/ewsc/yektefai/spectra_assets/hard_blind_molecular_model_eval/agent_visible/train.csv`
- `/ewsc/yektefai/spectra_assets/hard_blind_molecular_model_eval/agent_visible/eval_predictions.csv`
- `/ewsc/yektefai/spectra_assets/hard_blind_molecular_model_eval/agent_visible/metadata.json`

The prediction file has only:

- `sample_id`
- `smiles`
- `target`
- `y_true`
- `y_pred`

## Hypothesis

The vanilla agent may compute aggregate error and target-distribution shift. The
`/spectra` agent should be more likely to construct or request novelty metrics
from SMILES and report performance as a function of novelty.

## Rubric

Each dimension is scored 0-2:

- scientific unit,
- aggregate error,
- target distribution shift,
- constructs error metrics,
- constructs similarity or novelty,
- performance vs novelty,
- failure mode,
- next experiment.

This is the benchmark that better tests:

> Does `/spectra` cause an agent to discover and operationalize BOOM-like
> generalization failure without being handed BOOM's paper or precomputed
> novelty metrics?

## Completed Pilot

The completed pilot is summarized in
`docs/hard_blind_model_eval_pilot.md`, with machine-readable scores in
`experiments/hard_blind_model_eval/scores.json`. The primary rubric saturated:
both vanilla and `/spectra` agents scored 16/16. The useful difference was audit
depth: `/spectra` produced explicit property graphs, overlap validation, and
multiple performance-vs-novelty curves, while the vanilla agent still found the
main failure mode.
