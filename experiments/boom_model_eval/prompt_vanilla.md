# Vanilla Model-Generalizability Evaluation Prompt

You are given a trained molecular property model and its evaluation artifacts.
Your task is to evaluate whether the model generalizes.

Available artifacts:

- Model: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/model.pkl`
- Predictions: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/model_predictions.csv`
- Split metadata: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/split_stats.csv`
- Performance summary: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/performance_by_overlap.csv`

The model predicts one molecular property from Morgan fingerprints. The
prediction table contains molecule SMILES, split labels, true values, predicted
values, absolute error, squared error, and each test molecule's maximum Morgan
Tanimoto similarity to the training set.

Write a concise audit report answering:

1. Does the model generalize from the training distribution to the test data?
2. Which evidence supports your conclusion?
3. What additional analysis would you run next?

Return a JSON audit card with the fields:

- `scientific_unit`
- `model`
- `evaluation_summary`
- `generalization_conclusion`
- `failure_modes`
- `recommended_next_steps`
- `limitations`
