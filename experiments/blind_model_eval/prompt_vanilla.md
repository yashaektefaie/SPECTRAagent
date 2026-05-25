# Blind Vanilla Model-Generalizability Prompt

You are given artifacts for an anonymous trained molecular property model. You
are not given any benchmark paper.

Agent-visible files:

- Training data: `/ewsc/yektefai/spectra_assets/blind_molecular_model_eval/agent_visible/train.csv`
- Held-out predictions: `/ewsc/yektefai/spectra_assets/blind_molecular_model_eval/agent_visible/eval_predictions.csv`
- Metadata: `/ewsc/yektefai/spectra_assets/blind_molecular_model_eval/agent_visible/metadata.json`

Task:

Evaluate whether the trained model generalizes. Use the artifacts to find
evidence for or against generalization. Do not use any hidden labels or any
external paper context.

Return a JSON report with:

- `scientific_unit`
- `model_summary`
- `analysis_performed`
- `generalization_evidence`
- `generalization_conclusion`
- `failure_modes`
- `recommended_next_steps`
- `limitations`
