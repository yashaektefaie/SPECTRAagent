# `/spectra` Model-Generalizability Evaluation Prompt

You are given a trained molecular property model and its evaluation artifacts.
Your task is to evaluate whether the model generalizes using the `/spectra`
generalization-audit protocol.

Available artifacts:

- Model: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/model.pkl`
- Predictions: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/model_predictions.csv`
- Split metadata: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/split_stats.csv`
- Performance summary: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/performance_by_overlap.csv`

The model predicts one molecular property from Morgan fingerprints. The
prediction table contains molecule SMILES, split labels, true values, predicted
values, absolute error, squared error, and each test molecule's maximum Morgan
Tanimoto similarity to the training set.

Use `/spectra`:

1. Identify the scientific unit of generalization.
2. Identify the relevant novelty axis or property graph.
3. Validate whether train-test overlap changes across evaluated subsets.
4. Report model performance as a function of measured overlap.
5. Compute or interpret a spectral performance summary.
6. State whether the model generalizes and where it fails.

Return a JSON audit card with the fields:

- `scientific_unit`
- `model`
- `novelty_axis`
- `property_graph`
- `overlap_validation`
- `performance_vs_overlap`
- `spectral_summary`
- `generalization_conclusion`
- `failure_modes`
- `recommended_next_steps`
- `limitations`
