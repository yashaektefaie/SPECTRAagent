# Full `/spectra` Protocol Prompt

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

Execute a `/spectra` generalization audit. Use the model, data, and predictions
to:

1. Identify the scientific unit of generalization.
2. Propose multiple scientifically meaningful novelty axes.
3. Treat each novelty axis as a similarity hypothesis, not as a presumed answer.
4. Construct or approximate property graphs over the unit of generalization.
5. Validate that stricter novelty thresholds actually reduce train-test
   overlap.
6. Compute prediction-error metrics from `y_true` and `y_pred`.
7. Score each resulting curve as monotonic, localized, weak,
   non-explanatory, or not evaluable.
8. Use failed or weak axes to choose the next similarity definition or
   computation strategy.
9. Report performance as a function of measured novelty for supported axes.
10. Separate prospective axes from post-hoc explanatory axes that use labels,
    outcomes, predictions, or evaluation-set information.
11. Summarize the selected spectral failure mode, failed axes, and recommended
    next experiment.

For molecules, prefer chemically meaningful axes where available, such as
fingerprint similarity, scaffold or skeleton novelty, descriptor distance,
functional-group/property extremeness, and target-support shift. If the
required chemistry libraries are unavailable, use transparent approximations
from visible SMILES strings and state the limitation.

Do not use hidden labels, grading-only files, BOOM paper/repo context, or
external paper context.

Return a JSON report with:

- `scientific_unit`
- `model_summary`
- `novelty_axes`
- `property_graphs`
- `overlap_validation`
- `similarity_hypothesis_trace`
- `performance_vs_novelty`
- `selected_similarity_axis`
- `failed_or_weak_axes`
- `leakage_or_posthoc_caveats`
- `spectral_summary`
- `generalization_conclusion`
- `failure_modes`
- `recommended_next_steps`
- `limitations`
