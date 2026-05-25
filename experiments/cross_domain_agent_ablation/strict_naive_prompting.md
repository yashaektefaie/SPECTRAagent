# Strict-Naive Vanilla Prompting

This prompt is used for the stricter vanilla condition. It intentionally avoids
mentioning SPECTRA, novelty, similarity, spectral curves, audit cards,
agent-ablation, or being tested.

## Strict-Naive Vanilla Condition

You are given files for a trained scientific prediction model.

Files:

- training data: `{train_path}`
- held-out predictions: `{eval_path}`
- metadata: `{metadata_path}`

Task:

Evaluate whether the model generalizes.

Use only the files listed above. Do not use hidden files, grading files, paper
context, repository context, prior experiment outputs, or external benchmark
context. Write your analysis and any artifacts under:

`{output_dir}`

Return your final answer as JSON at:

`{output_dir}/report.json`
