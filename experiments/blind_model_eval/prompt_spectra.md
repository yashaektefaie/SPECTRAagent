# Blind `/spectra` Model-Generalizability Prompt

You are given artifacts for an anonymous trained molecular property model. You
are not given any benchmark paper.

Agent-visible files:

- Training data: `/ewsc/yektefai/spectra_assets/blind_molecular_model_eval/agent_visible/train.csv`
- Held-out predictions: `/ewsc/yektefai/spectra_assets/blind_molecular_model_eval/agent_visible/eval_predictions.csv`
- Metadata: `/ewsc/yektefai/spectra_assets/blind_molecular_model_eval/agent_visible/metadata.json`

Task:

Evaluate whether the trained model generalizes using `/spectra`. Use the
artifacts to identify the scientific unit, define novelty axes, validate
train-test overlap, and report performance as a function of measured novelty.
Do not use any hidden labels or any external paper context.

Return a JSON report with:

- `scientific_unit`
- `model_summary`
- `novelty_axes`
- `property_graphs`
- `overlap_validation`
- `performance_vs_novelty`
- `spectral_summary`
- `generalization_conclusion`
- `failure_modes`
- `recommended_next_steps`
- `limitations`
