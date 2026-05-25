# Hard Blind Vanilla Model-Generalizability Prompt

You are given artifacts for an anonymous trained molecular property model. You
are not given any benchmark paper.

Agent-visible files:

- Model: `/ewsc/yektefai/spectra_assets/boom_numeric_pilot/artifacts/model.pkl`
- Training data: `/ewsc/yektefai/spectra_assets/hard_blind_molecular_model_eval/agent_visible/train.csv`
- Held-out predictions: `/ewsc/yektefai/spectra_assets/hard_blind_molecular_model_eval/agent_visible/eval_predictions.csv`
- Metadata: `/ewsc/yektefai/spectra_assets/hard_blind_molecular_model_eval/agent_visible/metadata.json`

The prediction file contains only raw held-out labels and predictions. It does
not include precomputed errors, ID/OOD labels, train-test similarity, or novelty
metrics.

Task:

Evaluate whether the trained model generalizes. Use the model, data, and
predictions to find evidence for or against generalization. Do not use hidden
labels, grading-only files, BOOM paper/repo context, or external paper context.

Return a JSON report with:

- `scientific_unit`
- `model_summary`
- `analysis_performed`
- `generalization_evidence`
- `generalization_conclusion`
- `failure_modes`
- `recommended_next_steps`
- `limitations`
