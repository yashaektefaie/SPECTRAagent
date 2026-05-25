# Iterative Agent Behavior Rerun

Date: 2026-05-12

This rerun tests the current `/spectra` behavior after integrating the
similarity-hypothesis loop into the MCP/procedure layer. The key behavioral
change is that an agent should no longer assume its first similarity definition
is correct. It should try an axis, validate overlap, score the resulting curve,
record non-explanatory or non-evaluable axes, and use those results to choose
the next similarity definition or computation strategy.

## Paper Capsules

The current benchmark prompt generator covers the launch capsules we identified:

- BOOM: molecules and chemical novelty.
- NABench: nucleotide fitness with random versus contiguous mutational-region
  splits.
- DART-Eval: regulatory DNA foundation-model evaluation.
- PerturBench: perturbation-response benchmark extension.
- Systema, UMAP virtual screening, and GeSS remain available as additional
  capsules.

For each `/spectra` condition, the generated prompt now instructs the agent to
treat each spectral property as a similarity hypothesis and run an iterative,
leakage-aware search. The vanilla condition is still instructed to use normal
scientific reproduction workflow without the SPECTRA checklist.

## What Was Rerun

I reran the executable checks under `/ewsc/yektefai/spectra_depth_demos`:

- BOOM numeric mini-audit:
  `/ewsc/yektefai/spectra_depth_demos/boom_numeric_mini_audit_agent_behavior_rerun`
- NABench sequence mini-audits:
  `/ewsc/yektefai/spectra_depth_demos/nabench_sequence_mini_audit_agent_behavior_rerun_Martin_2018_myc_enhancer`
  `/ewsc/yektefai/spectra_depth_demos/nabench_sequence_mini_audit_agent_behavior_rerun_Gregory_2018_mRNA`
  `/ewsc/yektefai/spectra_depth_demos/nabench_sequence_mini_audit_agent_behavior_rerun_Pitt_2010_ribozyme`

This was a deterministic rerun of the package-backed prompts and audit runners,
not a fresh multi-agent LLM trial.

## BOOM Result

The BOOM rerun reproduced the previous molecule-style behavior. A lightweight
random-forest baseline over Morgan fingerprints had much worse density-property
error on the BOOM OOD split than on the ID split, and the OOD error increased as
maximum train-set Tanimoto similarity decreased.

- ID RMSE: `0.044252`
- OOD all RMSE: `0.229631`
- OOD max Tanimoto <= 0.5 RMSE: `0.253409`
- AUSPC, negative RMSE area: `-0.240961`

This remains a clean demonstration that a measured train-test similarity axis
can turn a split-level OOD finding into a spectral performance curve.

## NABench Before Versus After

The earlier NABench run used one position-support similarity axis. It often did
not produce the expected degradation curve.

| Dataset | Old selected axis | Old all-eval RMSE | Old lowest-overlap RMSE | Old AUSPC |
| --- | --- | ---: | ---: | ---: |
| Martin_2018_myc_enhancer.csv | none | 0.076314 | 0.033897 | -0.069685 |
| Gregory_2018_mRNA.csv | none | 1.158248 | 0.701033 | -1.253552 |
| Pitt_2010_ribozyme.csv | none | 0.013725 | 0.011403 | -0.012269 |

After the iterative MCP behavior was integrated, the NABench runner evaluated
multiple similarity hypotheses and selected the strongest supported axis.

| Dataset | Selected axis | Curve status | All-eval RMSE | Lowest-overlap RMSE | AUSPC |
| --- | --- | --- | ---: | ---: | ---: |
| Martin_2018_myc_enhancer.csv | position_fitness_composite_similarity | monotonic_supported | 0.076314 | 0.103771 | -0.175900 |
| Gregory_2018_mRNA.csv | fitness_support_similarity | monotonic_supported | 1.158248 | 1.892088 | -2.617461 |
| Pitt_2010_ribozyme.csv | position_fitness_composite_similarity | monotonic_supported | 0.013725 | 0.016514 | -0.021391 |

The iteration traces show the intended behavior:

- Martin: mutation-position support was marked `not_explanatory`; sequence
  identity, mutation depth, and fitness support were not evaluable; the
  position-fitness composite axis was selected.
- Gregory: mutation-position support was already monotonic and retained as a
  secondary axis; fitness support produced the strongest selected curve.
- Pitt: mutation-position support was marked `not_explanatory`; fitness support
  and the composite axis were monotonic, with the composite selected.

## Interpretation

The agent behavior we want is now represented in the runnable system: a weak or
non-monotonic first axis is itself a finding, and it triggers another similarity
hypothesis rather than ending the audit.

The important caveat is that the strongest selected NABench axes use fitness
labels, either directly or through a position-fitness composite. Those are
post-hoc explanatory axes. They are useful for explaining where the current
model failed, but they are not valid prospective split-design axes before labels
are available. The audit reports this caveat in the generated summaries.

This supports a more precise SPECTRA claim:

> `/spectra` helps agents transform broad generalization questions into an
> iterative search over scientifically meaningful similarity hypotheses, while
> recording failed axes and leakage caveats.

It does not yet prove that SPECTRA-equipped LLM agents outperform vanilla LLM
agents on these capsules. The next benchmark should run the updated vanilla
versus `/spectra` prompts and score them with
`experiments/iterative_agent_behavior/rubric.json`.

## 2026-05-13 Prospective-Axis Rerun

A follow-up rerun tightened the sequence evidence by adding a local
mutation-window identity axis and preferring leakage-free supported axes over
post-hoc label-based axes.

| Dataset | Selected axis | Leakage risk | Curve status | All-eval RMSE | Lowest-overlap RMSE | AUSPC |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Gregory_2018_mRNA.csv | mutation_position_support_similarity | none | monotonic_supported | 1.158248 | 1.572344 | -1.253552 |
| Martin_2018_myc_enhancer.csv | mutation_centered_window_identity_similarity | none | monotonic_supported | 0.076314 | 0.089943 | -0.093598 |
| Pitt_2010_ribozyme.csv | position_fitness_composite_similarity | post_hoc_uses_eval_labels | monotonic_supported | 0.013725 | 0.016514 | -0.021391 |

The interpretation changed in an important way: NABench is no longer only a
post-hoc explanatory demo. Two of the three tested assays now have selected
prospective axes. Pitt remains a useful diagnostic failure case because the
tested leakage-free axes did not explain the held-out error pattern.
