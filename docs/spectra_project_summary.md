# SPECTRA Project Summary

Date: 2026-05-13

This document summarizes the current SPECTRA build, the framing decisions we
settled on, the experiments run so far, and the next engineering steps.

## Purpose

SPECTRA is an executable standard for scientific generalization auditing. The
core idea is to evaluate model performance as a function of scientifically
meaningful train-test similarity or distance, rather than relying on a single
random, scaffold, chromosome, site, or metadata split.

The clean architecture is:

> SPECTRA is the deterministic audit engine. `/spectra` is the agent-native
> interface to that engine.

The package should work without agents. An agent adds value by inspecting local
files, selecting a defensible scientific similarity definition, choosing a
scalable computation strategy, running the audit, and writing a standardized
report.

## Current Flow

SPECTRA now has two evaluation modes.

1. Comprehensive audit mode: estimate performance over the full train-similarity
   or novelty curve.
2. Targeted operating-point mode: evaluate a named deployment condition, such as
   scaffold holdout, chromosome holdout, leave-site-out validation, or contiguous
   mutational-region split, then report where that point lands on the broader
   SPECTRA curve.

The expected agent or human workflow is:

1. Identify the scientific unit of generalization.
2. Choose or define a similarity notion for that unit.
3. Choose a computation strategy for producing train-eval similarities at the
   available scale.
4. Emit either a long-form `pairwise_similarity.csv` or an evaluation table with
   a precomputed spectral axis.
5. Run the deterministic SPECTRA audit.
6. Validate that stricter thresholds reduce measured train-test overlap.
7. Report performance curves, AUSPC or equivalent summaries, limitations, and a
   reusable audit card.

This keeps molecule-specific logic out of the framework core. Morgan Tanimoto is
one example similarity definition, not the SPECTRA abstraction.

## Implemented Components

The repo now contains:

- A package-backed audit CLI in `spectrae/cli.py` and `spectrae/audit.py`.
- A deployable MCP server in `spectrae/scientific_skill_mcp.py`.
- A similarity definition registry with 54 seed-mined entries.
- A similarity computation strategy registry with 38 strategies.
- A targeted operating-point registry with 28 methods.
- Benchmark capsule utilities in `spectrae/spectra_benchmarks.py`.
- Report templates and procedure docs under `spectrae/report_templates/` and
  `spectrae/procedures/`.
- Tests for the audit engine, benchmark capsules, and all three registries.

## CLI Surface

The central CLI entry point is:

```bash
spectra audit
```

The generic audit accepts predictions plus a measured axis:

```bash
spectra audit \
  --eval eval_predictions.csv \
  --out artifacts/spectra_audit \
  --target-col y_true \
  --pred-col y_pred \
  --axis-col max_train_similarity \
  --axis-type similarity \
  --axis-name agent_defined_similarity \
  --scientific-unit sample
```

The preferred framework-level workflow accepts a train-eval similarity graph:

```bash
spectra audit \
  --mode pairwise \
  --eval eval_predictions.csv \
  --pairwise-similarity pairwise_similarity.csv \
  --out artifacts/spectra_audit \
  --eval-id-col sample_id \
  --similarity-eval-id-col sample_id \
  --similarity-train-id-col train_id \
  --similarity-col similarity \
  --target-col y_true \
  --pred-col y_pred
```

The CLI writes standardized artifacts:

- `audit_card.json`
- `split_stats.csv`
- `performance_by_axis.csv`
- `eval_with_axis.csv`
- `spectral_curve.svg`
- `report.md`

See `docs/spectra_cli.md`.

## MCP Surface

The MCP server exposes SPECTRA as an agent-native skill:

```bash
python -m spectrae.scientific_skill_mcp \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 8000
```

Docker support exists through `Dockerfile.mcp`:

```bash
docker build -f Dockerfile.mcp -t spectrae-scientific-skills .
docker run --rm -p 8000:8000 spectrae-scientific-skills
```

The server exposes resources for the generalizability procedure, benchmark
capsules, similarity definitions, similarity computation strategies, and
operating-point methods. It also exposes tools for listing, suggesting,
validating, and rendering those resources, plus `run_spectra_audit`.

For paper/question-driven use, the high-level entrypoint is
`start_spectra_audit_session`. A user supplies a generalization question, a
model paper/reference, and any visible model/dataset context. The tool returns
the role graph, spawn plan, routing policy, artifact tree, and terminal gate for
an Investigator-Distiller-Dataset Scout/Dataset Constructor loop. Clients with
subagent support can spawn those roles directly; single-agent MCP clients can
execute the same roles sequentially.

The autonomous runner surface is now split into two safe operations:
`prepare_spectra_audit_session` creates the session directory, initial
Investigator prompt, work order, and state files without launching roles;
`run_spectra_audit_session` can execute the loop through a configurable
host-agent command template when `execute_roles=true`. The CLI equivalent is
`spectra agent prepare` and `spectra agent run`.

Session scope is explicit. `paper_claim_audit` audits the paper's own claims;
`beyond_paper_discovery` treats the paper as context and directs the agent to
search for or construct external/public tests of generalization. The latter is
the intended mode for reproducing open-ended ENCODE-style discovery behavior.

See `docs/scientific_skill_mcp.md`.

## Registries

The similarity definition registry answers:

> What should count as similarity for this scientific unit?

It currently includes 54 seed-mined entries covering molecules, drug-target
pairs, regulatory DNA, nucleotide and protein variants, RNA, perturbations,
ontology annotations, materials, geospatial samples, imaging, clinical records,
graphs, time series, spectra, multi-omics, patient similarity, text, and other
AI4Science settings.

See `docs/similarity_definition_registry.md`.

The similarity computation registry answers:

> How should an agent compute enough train-eval similarity edges without doing
> unnecessary all-pairs work?

It currently includes 38 strategies, including exact chunked all-pairs, LSH,
HNSW, FAISS-style indexing, filtered ANN, NN-Descent, MinHash, sequence
seed-and-extend, molecular Tanimoto indexing, sparse cosine search, DTW pruning,
metric trees, graph/kernel approximations, Wasserstein approximations, learned
candidate filters, privacy-preserving search, and distributed ANN.

See `docs/similarity_computation_registry.md`.

The operating-point registry answers:

> If the user wants one known deployment condition rather than the full curve,
> which split or validation operator should be used?

It currently includes 28 methods, including random IID, group/domain/time/site
holdouts, spatial and spatiotemporal splits, scaffold and molecular cluster
splits, chromosome holdout, cross-cell-type regulatory holdout, contiguous
mutational-region split, sequence homology splits, perturbation holdouts,
scanner/site holdouts, materials cluster splits, and graph distribution-shift
splits.

See `docs/operating_point_registry.md`.

## Experiments Completed

### BOOM Numeric Mini-Audit

We ran a measured BOOM-style density audit on a 10k split using a
`RandomForestRegressor` over Morgan fingerprints. The novelty axis was maximum
Morgan Tanimoto similarity from each test molecule to the training set.

Key result:

- ID test: RMSE 0.0443, MAE 0.0338.
- OOD all: RMSE 0.2296, MAE 0.2085.
- OOD with max Tanimoto <= 0.5: RMSE 0.2534, MAE 0.2330.
- AUSPC using negative RMSE area: `-0.24096112142647133`.

Interpretation: the result is directionally consistent with the SPECTRA
hypothesis. Lower train-test chemical overlap corresponded to worse model
performance for this lightweight baseline.

See `docs/boom_numeric_mini_audit.md`.

### Value Ablation

We ran a hard-blind molecular ablation with three conditions:

1. broad generalizability prompt,
2. explicit distance-from-train prompt,
3. full `/spectra` protocol.

Scores:

- Broad generalizability: 19 / 22.
- Distance from train: 19 / 22.
- `/spectra` protocol: 22 / 22.

Interpretation: this experiment does not support the claim that vanilla agents
cannot find the core molecular generalization failure. Broad and
distance-prompted agents both found it. It does support the narrower claim that
`/spectra` improves protocol completeness: multiple axes, property graph
definitions, overlap validation, invalid-axis handling, and reusable report
structure.

See `docs/spectra_value_ablation_results.md`.

### Depth Demo Execution

Paperclip was used to mine and triage launch-depth demos across molecules,
regulatory DNA, nucleotide fitness, perturbation biology, and clinical
generalization. Papers and repos for BOOM, DART-Eval, NABench, and PerturBench
were fetched under `/ewsc/yektefai/spectra_depth_demos`.

Two executable demos were run:

- BOOM molecular mini-audit: strong monotonic chemical novelty curve.
- NABench sequence mini-audit: runnable cross-domain audit with an iterative
  similarity-hypothesis loop. The first mutation-position axis was not always
  explanatory, so the runner tried follow-up axes and recorded which similarity
  definitions better explained the observed failures.

See `docs/depth_demo_paperclip_execution.md`.

### Iterative Agent Behavior Rerun

After integrating the iterative similarity-hypothesis instructions into the
MCP/procedure layer, the executable rerun showed the intended behavior on the
NABench sequence demos. The audit now records failed or non-evaluable axes,
continues to follow-up similarity definitions, selects the strongest supported
curve, and reports post-hoc label-use caveats when a selected axis is
explanatory rather than prospective.

See `docs/iterative_agent_behavior_rerun.md`.

### Vanilla Versus `/spectra` Agent Trial

We ran a controlled agent trial on an anonymous Martin 2018 sequence-fitness
task. Both agents found the same aggregate failure: the model behaves like a
near-constant predictor with held-out RMSE around `0.0763` and R2 around
`-0.008`. The vanilla agent produced a competent static analysis and reported
that most intuitive novelty axes were not convincingly evaluable. The `/spectra`
agent produced a full iterative audit: pairwise similarity graphs, per-axis
SPECTRA artifacts, failed-axis reporting, leakage/post-hoc classification,
AUSPC, and a selected prospective local-context axis.

Rubric score:

- Vanilla: `17 / 24`.
- `/spectra`: `24 / 24`.

This supports the narrower claim that `/spectra` improves protocol completeness
and iterative audit behavior. It does not support the stronger claim that
vanilla agents cannot find the broad aggregate generalization failure.

See `docs/iterative_agent_trial_results.md`.

### Paper Results Run

We executed the experiment set needed to populate the current paper draft's
results placeholders:

- deterministic BOOM molecular audit,
- deterministic NABench sequence-fitness audits on three assays,
- Martin/NABench vanilla versus `/spectra` agent comparison,
- BOOM four-condition ablation: vanilla, SPECTRA CLI only, CLI plus registries,
  and full `/spectra`,
- DART-Eval and PerturBench feasibility inspection.

The clean flagship empirical result is BOOM: ID RMSE was `0.0443`, full OOD RMSE
was `0.2296`, and the lowest-overlap OOD subset reached RMSE `0.2534` as mean
max train Tanimoto fell to `0.390`. The agent results support a careful claim:
`/spectra` improves audit completeness, artifact quality, metric reporting, and
prospective/post-hoc disclosure, but strong vanilla agents can still find obvious
generalization failures on molecule tasks.

See `docs/paper_results_20260512.md`.

### Cross-Domain Results Run

The BOOM-centered result set was extended with sequence-fitness and
perturbation-biology capsules under:

`/ewsc/yektefai/spectra_paper_results_20260513`

NABench was rerun with a broader sequence axis set and leakage-aware axis
selection. Two of three tested assays now have selected prospective axes:

- Gregory 2018 mRNA: mutation-position support; contiguous RMSE `1.1582`,
  lowest-overlap RMSE `1.5723`, AUSPC `-1.2536`.
- Martin 2018 MYC enhancer: mutation-centered window identity; contiguous RMSE
  `0.0763`, lowest-overlap RMSE `0.0899`, AUSPC `-0.0936`.

Pitt 2010 ribozyme remains diagnostic-only: the strongest supported curve used a
post-hoc position-fitness composite axis that depends on evaluation labels.

PerturBench was made executable using the repository's bundled `devel.h5ad`.
The local capsule evaluates K562 two-gene combination perturbations with an
additive single-perturbation response baseline. Component-support similarity
was prospective, and mean profile RMSE decreased from `0.0978` with no component
support to `0.0199` when both components were supported.

DART-Eval was inspected but not executed because the processed task files and
genome references are Synapse-hosted.

This run supports a bounded cross-domain claim: SPECTRA can convert fixed model
predictions and explicit train-test similarity relationships into standardized
audits across molecules, sequence fitness, and perturbation biology.

See `docs/paper_results_20260513_cross_domain.md`.

### Fresh Cross-Domain Agent Ablation

We then ran the actual with-versus-without experiment across the executable
domains. Six fresh agents were launched: vanilla and `/spectra` for molecules,
sequence fitness, and perturbation biology. Each agent received a blinded bundle
with visible train data, held-out predictions, and minimal metadata. Agents were
not given paper context, benchmark repo context, split labels, precomputed
train-test similarities, grading files, or deterministic SPECTRA summaries.

Scores on the fixed 24-point rubric:

- molecules: vanilla `17 / 24`, `/spectra` `24 / 24`.
- sequence fitness: vanilla `17 / 24`, `/spectra` `23 / 24`.
- perturbation biology: vanilla `18 / 24`, `/spectra` `24 / 24`.

The interpretation is narrow and important. Vanilla agents found the main
generalization failure in all three domains. `/spectra` improved completeness:
pairwise similarity artifacts, overlap validation, AUSPC or equivalent spectral
summaries, failed-axis reporting, leakage classification, and audit-card
quality.

A stricter rerun removed all similarity/novelty language from the vanilla
prompt. It only asked agents to evaluate whether the model generalizes. Even
then, strict-naive vanilla agents scored close to `/spectra` because the natural
axes were visible in the schemas:

- molecules: strict-naive `21 / 24`, `/spectra` `24 / 24`.
- sequence fitness: strict-naive `22 / 24`, `/spectra` `23 / 24`.
- perturbation biology: strict-naive `22 / 24`, `/spectra` `24 / 24`.

This weakens the discovery claim. The current evidence positions `/spectra` as
a standardization and reproducibility layer, not as something required for
capable agents to infer obvious similarity axes.

See `docs/cross_domain_agent_ablation_20260513.md`.

## Framing Decisions

We should not claim:

> Without SPECTRA, agents cannot discover generalization failures.

The current evidence does not support that. A defensible claim is:

> SPECTRA turns broad generalizability questions into executable, validated,
> domain-aware distance audits.

Another defensible claim is:

> `/spectra` reduces setup friction and improves artifact completeness by
> giving agents a standard audit engine, literature-backed similarity
> definitions, scalable similarity-computation strategies, and reusable
> reporting contracts.

Existing split protocols should not be framed as obsolete. The better framing
is that SPECTRA locates them on a broader novelty spectrum. If the goal is
discovery, use the comprehensive curve. If the goal is a known deployment
condition, use the targeted operating point and measure where it falls on the
curve.

## Deployment Status

The project is deployable as an MCP service. The entry point accepts
`--transport streamable-http`, `--host`, and `--port`, and `Dockerfile.mcp`
builds a server image.

Before public release, the deployment still needs:

- a clean Docker build from a fresh checkout,
- an explicit hosted deployment target and URL,
- authentication or access restrictions if exposed publicly,
- rate limits and request-size limits,
- a tagged release or commit boundary,
- a short README path for connecting from Claude, Cursor, or other MCP clients.

## Key Files

- `spectra.md`: core SPECTRA protocol.
- `AGENTS.md`: agent instructions.
- `spectrae/audit.py`: deterministic audit logic.
- `spectrae/cli.py`: CLI entry point.
- `spectrae/scientific_skill_mcp.py`: MCP server.
- `spectrae/similarity_registry.py`: similarity definition registry code.
- `spectrae/similarity_computation_registry.py`: computation strategy registry
  code.
- `spectrae/operating_point_registry.py`: targeted operating-point registry
  code.
- `docs/spectra_cli.md`: CLI documentation.
- `docs/scientific_skill_mcp.md`: MCP documentation.
- `docs/similarity_definition_registry.md`: similarity definition registry
  documentation.
- `docs/similarity_computation_registry.md`: computation strategy registry
  documentation.
- `docs/operating_point_registry.md`: operating-point registry documentation.
- `docs/spectra_framing_and_evidence_plan.md`: framing and evidence plan.
- `docs/boom_numeric_mini_audit.md`: BOOM numeric pilot.
- `docs/spectra_value_ablation_results.md`: hard-blind ablation results.
- `docs/depth_demo_paperclip_execution.md`: Paperclip-mined depth-demo
  execution report.

## Next Steps

1. Harden deployment: clean Docker build, hosted URL, auth, rate limits, and MCP
   client setup docs.
2. Add a public quickstart that starts from train, eval, predictions, and a
   generated `pairwise_similarity.csv`.
3. Run DART-Eval once Synapse data and genome references are available.
4. Replace the PerturBench development capsule with one full public PerturBench
   task if data size and dependencies are manageable.
5. Make the agent benchmark require prospective audit design before revealing
   labels, so `/spectra` is tested on valid axis choice rather than post-hoc
   curve fitting.
6. Promote high-confidence registry entries from `seed_mined` to
   `human_reviewed` after expert review.
7. Package the MCP server with a release tag and a simple hosted endpoint.
