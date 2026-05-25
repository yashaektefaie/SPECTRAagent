# BOOM Trained-Model Generalizability Evaluation Design

## Motivation

The first with-vs-without pilot gave both agents the BOOM paper and asked them
to reproduce one core BOOM finding. That tested whether `/spectra` helps agents
turn a known benchmark paper into a structured generalization audit.

A cleaner next test is to give agents a trained model and evaluation artifacts,
without foregrounding the BOOM paper result, and ask:

> Does this model generalize?

This better tests whether `/spectra` changes the evaluation behavior itself.

## Artifact Bundle

The current trained-model bundle is:

- Model: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/model.pkl`
- Predictions: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/model_predictions.csv`
- Split stats: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/split_stats.csv`
- Performance summary: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/performance_by_overlap.csv`
- Existing audit card: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/audit_card.json`

The model is a random forest trained on Morgan fingerprints for the BOOM 10k
density task. The prediction table contains split labels, true values,
predictions, errors, and maximum Morgan Tanimoto similarity to the training set.

## Experimental Conditions

Vanilla condition:

- Receives the model/prediction artifacts.
- Is asked to evaluate whether the model generalizes.
- Does not receive `/spectra` terminology or checklist.

`/spectra` condition:

- Receives the same model/prediction artifacts.
- Is asked to evaluate generalizability using `/spectra`.
- Must identify novelty axes, validate overlap, and report performance as a
  function of measured overlap.

Prompt files:

- `experiments/boom_model_eval/prompt_vanilla.md`
- `experiments/boom_model_eval/prompt_spectra.md`

## Expected Difference

The vanilla agent may report ID vs OOD error and conclude that OOD performance
is worse. The `/spectra` agent should additionally use the provided
`max_train_tanimoto` field to show that OOD error increases as chemical overlap
decreases.

## Scoring

Recommended 0-2 rubric:

- Identifies molecule as the scientific unit.
- Reports ID vs OOD error.
- Uses measured train-test Tanimoto overlap.
- Reports performance as a function of overlap.
- Identifies the model's failure mode.
- States limitations clearly.
- Proposes the next valid computational experiment.

This experiment tests the claim:

> `/spectra` causes agents to evaluate trained scientific models in terms of
> measured novelty, not only named dataset splits.
