# SPECTRA Scientific Skills MCP Server

This server exposes the generalizability analysis procedure as an MCP skill.

## Local Install

```sh
pip install -e ".[mcp]"
```

For an installed package, the portable server command is:

```sh
spectra-mcp serve --transport stdio
```

Run diagnostics with:

```sh
spectra-doctor
```

## Run Over STDIO

```sh
spectra-mcp serve --transport stdio
```

When running directly from a checkout without installing the full SPECTRA
package, use:

```sh
python -m spectrae.scientific_skill_mcp
```

## Run Over Streamable HTTP

```sh
spectra-mcp serve \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 8000
```

## Docker

```sh
docker build -f Dockerfile.mcp -t spectrae-scientific-skills .
docker run --rm -p 8000:8000 spectrae-scientific-skills
```

## Exposed Capabilities

Resources:

- `procedure://generalizability_analysis/0.5.1`
- `procedure://generalizability_analysis/examples`
- `benchmark://spectra/capsules`
- `similarity-registry://spectra/definitions`
- `similarity-computation://spectra/strategies`
- `operating-points://spectra/methods`

Prompt:

- `generalizability_analysis_prompt`

Tools:

- `list_procedures`
- `get_procedure`
- `get_procedure_examples`
- `start_spectra_audit_session`
- `prepare_spectra_audit_session`
- `run_spectra_audit_session`
- `start_generalizability_analysis`
- `start_spectra_investigator`
- `select_spectra_execution_mode`
- `decide_next_spectra_experiment`
- `recommend_spectral_property`
- `plan_iterative_similarity_search`
- `plan_similarity_computation`
- `score_similarity_hypothesis_curve`
- `update_hypothesis_ledger`
- `choose_discriminating_experiment`
- `plan_hypothesis_driven_dataset_acquisition`
- `distill_spectra_hypotheses`
- `synthesize_spectra_generalizability_finding`
- `review_investigator_checkpoint`
- `reflect_on_replication_evidence`
- `translate_model_space_axis_to_domain_hypotheses`
- `assess_explanatory_depth`
- `enforce_mechanism_debt_gate`
- `plan_public_resource_acquisition`
- `plan_hypothesis_test_dataset_construction`
- `prepare_dataset_scout_request`
- `distill_dataset_scout_output`
- `prepare_dataset_constructor_request`
- `distill_dataset_constructor_output`
- `validate_split_stats`
- `review_generalizability_report`
- `list_benchmark_capsules`
- `get_benchmark_capsule`
- `get_benchmark_download_plan`
- `create_audit_card_template`
- `compute_auspc`
- `validate_spectra_audit_card`
- `score_spectra_agent_audit`
- `render_spectra_audit_report`
- `list_similarity_definitions`
- `get_similarity_definition`
- `suggest_similarity_definitions`
- `get_similarity_example_script`
- `render_similarity_definition`
- `validate_similarity_registry`
- `list_similarity_computation_strategies`
- `get_similarity_computation_strategy`
- `suggest_similarity_computation_strategies`
- `get_similarity_computation_example_script`
- `render_similarity_computation_strategy`
- `validate_similarity_computation_registry`
- `list_operating_point_methods`
- `get_operating_point_method`
- `suggest_operating_point_methods`
- `render_operating_point_method`
- `validate_operating_point_registry`
- `run_spectra_audit`

## Recommended Agent Flow

When a user gives `/spectra` a scientific question and a model paper, start
with `start_spectra_audit_session`. This returns the single-controller prompt,
artifact tree, quality gates, and internal phase contract for a SPECTRA audit
session. Distiller, Investigator, Dataset Scout, Dataset Fetcher, Auditor, and
synthesis are phases inside the controller session, not separately routed
agents.

For the autonomous product path, use `prepare_spectra_audit_session` to create a
session directory and one controller prompt, or `run_spectra_audit_session` to
let a configurable host-agent command launch that controller. `run_spectra_audit_session`
is safe by default: it only prepares the session unless `execute_controller=true` and
`agent_command_template` is supplied. The command template can use
`{prompt_path}`, `{write_scope}`, `{role}`, and `{round}`; `role` is always
`controller`.

Example invocation:

```json
{
  "question": "Use /spectra to assess the generalizability of this model.",
  "model_paper": "/path/to/model_paper.pdf",
  "model_description": "Frozen Caduceus embeddings with a downstream probe",
  "dataset_description": "ENCODE cCRE 1024 bp sequence windows",
  "domain": "regulatory DNA",
  "client_capabilities": ["subagents", "filesystem", "network"],
  "output_root": "spectra_audit_session"
}
```

For broad questions like this, the session launcher marks the run as
`broad_model_generalizability_audit`. The controller first extracts the model
paper's stated and implicit generalization claims, then lets experiments and
internal critique narrow the live hypotheses. Users do not need to specify
similarity metrics, novelty axes, or separate roles.

When the desired behavior is discovery beyond the paper, set
`audit_scope="beyond_paper_discovery"` or make the question explicit:

```json
{
  "question": "Use /spectra to assess Caduceus generalizability beyond the paper's reported evaluations; construct or acquire public regulatory genomics datasets if needed.",
  "model_paper": "/path/to/caduceus_paper.pdf",
  "model_description": "Caduceus DNA sequence representations",
  "domain": "regulatory DNA",
  "audit_scope": "beyond_paper_discovery"
}
```

In this scope, the paper seeds the hypothesis ledger but does not constrain the
audit to the paper's benchmarks. The Investigator should search for public/local
datasets or construct a dataset when needed to test a scientifically important
external generalization axis.

The MCP server does not ask the host application to spawn workers or run a role
router. It supplies one controller prompt and the gates that controller should
use while it keeps thinking and iterating.

CLI equivalents:

```sh
spectra agent prepare \
  --question "Use /spectra to assess the generalizability of this model." \
  --paper ./paper.pdf \
  --model-description ./model \
  --dataset-description ./data \
  --domain "regulatory DNA" \
  --audit-scope beyond-paper-discovery \
  --out ./spectra_session

spectra agent run \
  --question "Use /spectra to assess the generalizability of this model." \
  --paper ./paper.pdf \
  --model-description ./model \
  --dataset-description ./data \
  --out ./spectra_session \
  --agent-command-template 'codex exec --prompt-file {prompt_path}' \
  --execute
```

Low-level flow inside each session:

1. Call `get_procedure`.
2. Call `get_procedure_examples`.
3. Call `start_generalizability_analysis`.
4. Call `start_spectra_investigator`. The hypothesis ledger, not the axis checklist, is the control state.
5. Call `select_spectra_execution_mode`. Benchmark mode is required when raw labels and a trainable model, model code, or defensible baseline are available.
5. If the target model cannot import or run, inspect the repository dependencies, setup scripts, model cards, and local artifacts; attempt a reasonable environment or code-path repair before falling back to a surrogate baseline.
6. Call `plan_iterative_similarity_search` with the dataset schema and task.
7. Declare an initial axis-search scope before stopping criteria are considered: candidate axis classes, approximate compute budget, minimum target-model behavioral tests, and the additional data/features/tasks/model access/compute needed if the current scope fails.
8. Declare a task-coverage plan. Enumerate all available tasks/datasets, run an all-task data screen when possible, rank tasks by suspiciousness, scientific diversity, source family, feasibility, sample size, and prior/internal signals, then run all feasible tasks or document blockers for omitted tasks.
9. Inspect candidate definitions with `get_similarity_definition` and, when useful, `get_similarity_example_script`.
10. For each candidate axis, call `suggest_similarity_computation_strategies` with the chosen definition, data shape, scale, and available inputs.
11. If the user wants a specific deployment condition rather than the full curve, call `suggest_operating_point_methods`.
12. If a data-level screen finds exact duplicates, high train-test support, or suspicious split geometry, label it pre-benchmark screening and choose a behavioral follow-up.
13. In benchmark mode, generate similarity-controlled train/test splits and train a fresh model for each split using the same training recipe.
14. In audit fallback mode, generate `pairwise_similarity.csv`, a precomputed spectral axis, or fixed-prediction evaluation subsets only after documenting why retraining is impossible.
15. Call `plan_similarity_computation`.
16. Run `run_spectra_audit` or an equivalent benchmark-mode evaluator.
17. Call `score_similarity_hypothesis_curve` on the generated performance curve.
18. Call `update_hypothesis_ledger`, then `choose_discriminating_experiment`. Do not choose the next axis because it is available; choose the experiment that would most change the live hypotheses.
19. If the current rows cannot falsify the sharpest live hypothesis, call `plan_hypothesis_driven_dataset_acquisition` and acquire or construct a fit-for-purpose public/local dataset that can distinguish the live explanations.
20. Call `decide_next_spectra_experiment` after each axis result. For supported axes, pass the current explanatory-depth classification and mechanism-debt status; replicated support cannot be selected while depth is unknown or mechanism debt is unresolved.
19. If the axis is `not_explanatory` or `not_evaluable`, record that finding and try the next candidate from an untested axis class while the declared search scope remains.
20. If tested tasks are negative or mixed and untested feasible tasks remain, expand task coverage before reporting a negative or exhausted search.
21. If the axis is `localized_supported`, refine the axis or try a composite similarity.
22. If the axis is `monotonic_supported` or `weak_supported`, immediately replicate it on another task, dataset, seed, model setting, or stronger operating point when feasible; do not stop with a proposal when replication inputs are available.
23. After replication, call `reflect_on_replication_evidence` to compare supported and non-supported tasks and derive residual or composite axes.
24. If replication support is mixed, test at least one executable residual axis, such as model-representation support after matching sequence similarity, motif-family support, task-context/provenance interactions, or class-conditional curves.
25. If a model-space or embedding-space axis is supported, call `translate_model_space_axis_to_domain_hypotheses`, annotate high- and low-support model-space regions, and test at least one biological/domain hypothesis when feasible.
26. For every supported proxy or domain-interpretable axis, call `assess_explanatory_depth`. If the axis is surface-level or broad-proxy evidence, continue to curated annotations, mechanism-level features, context interactions, or mediation/residual tests when inputs allow.
27. For every supported proxy, broad domain proxy, model-space pointer, or mechanism-like axis without controls, call `enforce_mechanism_debt_gate`. A proxy-only report is a checkpoint only. Local mechanism/mediation tests are an intermediate tier; if they remain proxy-level, continue to public-resource, constructed-dataset, residual-axis, broader-task, or launched continuation work. A queued-only manifest is not sufficient.
28. If local inputs contain sequences, structures, images, text, tables, metadata, or feature matrices, run input-derived mechanism or mediation tests before treating missing external annotations as blockers. For sequence tasks, missing coordinates do not excuse skipping motif/PWM, motif grammar, CpG/low-complexity/repeat-like, matched residual, or class-conditional tests.
29. If those resources are absent locally, call `plan_public_resource_acquisition`. If stable mapping identifiers are missing, inspect upstream dataset repositories, loaders, download scripts, raw archives, manifests, README files, dataset cards, and supplementary files for recoverable provenance before expensive matching/alignment or blocker reporting. Then search/download/query public resources when feasible, map them to local units, validate the mapping, and run the next mechanism or mediation test before declaring a blocker.
30. If the current benchmark still cannot test the mechanism hypothesis after local-derived mechanism tests, provenance recovery, and resource acquisition, call `prepare_dataset_scout_request` when the best replacement dataset is not obvious. The Dataset Scout compares public/local candidates and preserves the inconsistency ledger. Then call `distill_dataset_scout_output` to decide whether to continue scouting or promote a candidate to Dataset Constructor.
31. When a candidate has been selected, call `plan_hypothesis_test_dataset_construction` and `prepare_dataset_constructor_request`. A Dataset Constructor should construct a defensible dataset from public/local resources, validate provenance and mappings, audit leakage, control confounders, and return SPECTRA-ready split candidates.
31. Call `validate_split_stats` or inspect the generated audit-card validations.
32. Call `review_generalizability_report`.
33. Fill and score a SPECTRA checkpoint report that includes selected axes, failed axes, task coverage, omitted-task continuation jobs, replication evidence, replication reflection, residual-axis results, model-space/domain translation, explanatory-depth assessment, mechanism-debt status, public-resource acquisition attempts, constructed hypothesis-test dataset evidence, mechanism-axis or mediation results, execution mode, retraining evidence, axis-search scope, next-experiment decisions, and the next executable continuation.
34. When the loop appears to have produced a meaningful bounded insight, call `synthesize_spectra_generalizability_finding` with the model paper context, all SPECTRA findings, the Investigator trace, and any vanilla-agent comparison. The final Distiller must first check for a stronger unresolved alternative axis. If one exists, it should route back to the Investigator with that axis as the new primary hypothesis. Otherwise, the final output should be a paper-ready finding with interpretation, evidence boundary, evidence-to-claim mapping, and overclaim guardrails.

The core agent behavior is exploratory. SPECTRA should not be run as a single
metric check or static checklist. A non-monotonic, non-explanatory, or data-only
overlap result is itself a finding that should guide the next behavioral
experiment, similarity definition, or computation method.

In investigator mode, every result must update a hypothesis ledger. The agent
should report what became more likely, what became less likely, what surprised
it, and which experiment would distinguish the remaining explanations. A report
that only lists axes, mixed results, and mechanism debt is a failed checkpoint
unless it also contains competing hypotheses, falsifiable predictions, a belief
update, and a launched or completed discriminating experiment.

When a run has many proxy-level or mixed findings, call
`distill_spectra_hypotheses`. The Distiller is read-only: it summarizes the
curves, ledger, and belief updates into a ranked scientific story, identifies
the strongest unresolved hypothesis, and returns a concrete handoff experiment
for the Investigator. It should narrow the question, not run another generic
axis checklist.

When the loop appears to have resolved a bounded scientific question, call
`synthesize_spectra_generalizability_finding`. This final Distiller stage first
checks whether a larger unresolved alternative axis is present. A small
attenuation result should not become the terminal story when another plausible
axis produced a much larger degradation and has not been pursued as the primary
hypothesis. If no stronger unresolved axis remains, the tool reads the model
paper and all accumulated SPECTRA artifacts, then writes the manuscript-facing
interpretation. It should explain what the model paper established, what
SPECTRA added, which signals survived or attenuated under controls, what the
result means scientifically, and what cannot be claimed.

If the Distiller concludes that the benchmark cannot answer the current
scientific question, call `prepare_dataset_scout_request` before construction
unless the replacement dataset is already justified. Dataset Scout should
compare multiple public/local candidates, explain what inconsistency each one
tests, and return a ranked candidate table. Then call
`distill_dataset_scout_output`; only promoted candidates should go to
`prepare_dataset_constructor_request`.

After the Dataset Constructor returns a package, call
`distill_dataset_constructor_output`. The Distiller should decide whether the
package has enough validation, leakage control, labels, and split candidates to
go back to the Investigator, or whether blocking construction gaps remain.

External dataset use is also hypothesis-driven. The agent should not download
another dataset just because the protocol has a resource-acquisition step. It
should pull or construct external data only when the current rows cannot test a
live hypothesis. The acquisition plan should name the hypothesis, required
fields, bounded first slice, mapping validation, leakage controls, and the
belief update that each possible outcome would imply.

Fixed-prediction binning is no longer sufficient for the primary SPECTRA claim
when retraining is feasible. It is an audit fallback for closed models, missing
labels, or unavailable training code, and the report must label it diagnostic.

Data-only overlap screens are also insufficient for model-generalization claims.
Exact duplicates, high train support, or suspicious split geometry should trigger
a targeted feasible behavioral experiment, such as retraining a downstream
probe/head/baseline on controlled splits. The agent should write checkpoint
reports until full feasible task coverage and a mechanism-level explanation with
controls exist. If the declared axis/data/compute scope is exhausted first, the
agent must launch a continuation with additional data, features, tasks, model
access, public resources, constructed datasets, compute, or a bounded fallback.
It must not claim that no degradation axis
exists or that the model is generally generalizable. Missing packages or import
failures are not terminal blockers until the agent has inspected the target
repository and attempted an allowed install, environment creation, or code-path
repair.

Task coverage is also part of the claim. The agent should not choose a few tasks
because they feel representative. It should run all feasible tasks, or produce a
ranked task-coverage plan with all-task screen results, selected tasks, omitted
tasks, and concrete blockers. A negative or inconclusive result from a limited
subset is preliminary and must trigger expansion to remaining feasible tasks
through a launched continuation job or bounded fallback.

The default exploration target is not to confirm a preselected similarity
measure. The agent should actively search for a defensible axis where
target-model performance changes. Non-explanatory axes are useful negative
findings, but they are not stop conditions while untested surface/content,
representation, metadata/context/provenance, or scientific-mechanism axes remain
within the declared data and compute budget.

Replicated support is also not a final stopping point when the evidence is
heterogeneous. If a sequence-support axis works for enhancer tasks but not splice
or promoter tasks, the agent should ask why, contrast the supported and failed
tasks, and use that reflection to test residual axes rather than simply reporting
the first replicated metric.

Embedding-space distance is not a biological explanation by itself. When a
model-representation axis is supported, the agent should treat it as a pointer to
what the model may have learned or failed to learn. It should translate the
model-space signal into domain hypotheses, such as motif-family enrichment,
regulatory-element class, chromatin/assay context, GC/CpG or low-complexity
composition, species/provenance, or class-conditional sequence patterns.

Proxy-level domain findings are not final explanations either. K-mer support,
GC/CpG, length, entropy, low-complexity, crude regulatory words, task-family
labels, metadata/provenance labels, clusters, and generic embeddings can reveal
where behavior changes, but the agent should ask what mechanism they stand in
for. When current inputs allow, it should test curated annotations,
mechanism-level features, context-conditioned curves, or mediation/residual
experiments that distinguish the mechanism from simpler proxies.

Supported proxy axes create mechanism debt. The agent cannot satisfy this by
writing that the result is proxy-level, running one local proxy-like test, or
listing mechanism tests as future work. Local input-derived mechanism tests are
the first tier, not a closure condition. If they remain proxy-level, the agent
must continue to public-resource acquisition, constructed-dataset tests,
residual axes, broader task coverage, or a launched continuation job/fallback.
For sequence tasks, sequence strings alone can support motif/PWM scans, motif
grammar, CpG/low-complexity/repeat-like features, GC/length/k-mer matched
residual curves, class-conditional curves, and model-space residuals after
matching surface support.

Missing local resources should trigger acquisition, not immediate stopping. If a
mechanism hypothesis needs public data, the agent should search official or
canonical sources, check license/version/size, download or query a useful
resource, map it to the current samples, validate that mapping, and then rerun
the relevant SPECTRA test. A resource blocker should name the actual failure,
such as missing coordinates/IDs, license or credential barriers, unavailable
assembly/version, excessive download size, network failure, or failed mapping
validation.

Missing IDs in processed local files are not enough to stop. The agent should
inspect upstream dataset provenance first: original benchmark repositories,
dataset loader code, download scripts, raw archives, manifests, dataset cards,
and paper supplements. For molecules, this means recovering SMILES/InChIKey,
compound IDs, assay IDs, or target IDs before querying chemical/target
resources. For perturbation data, this means recovering gene/guide/drug/cell
metadata before joining pathway or perturbation resources. For clinical or
imaging data, this means recovering code/site/time/patient/device/DICOM fields
before joining terminology, cohort, or acquisition resources. Only after
provenance recovery fails should it attempt expensive matching or report a
mapping-identifier blocker.

If the current benchmark still cannot answer the mechanism question, the agent
must construct a hypothesis-test dataset from public/local resources when that
is feasible, or launch a bounded construction fallback with a concrete blocker
for the full construction. This dataset must have explicit scientific units, non-circular
labels, resource provenance, mapping validation, prospective split features,
leakage controls, confounder controls, and fresh per-split training or probing
when benchmark mode is feasible. Its results should be reported as extension
evidence for the mechanism hypothesis, not as a replacement for the original
benchmark.

Local benchmark-mode proxy-audit feasibility does not waive mechanism debt. If
the original rows cannot support a mechanism test and public resources cannot be
mapped, the agent must construct a public/local hypothesis-test dataset when
that can test the mechanism directly, or launch a bounded construction fallback.

The similarity registry is intentionally broad rather than molecule-specific.
The current seed set covers molecules, drug-target pairs, regulatory DNA,
nucleotide and protein variants, RNA structures, single-cell perturbations,
ontology annotations, materials, geospatial samples, medical images, clinical
contexts, graphs, time series, experimental batch/protocol shifts,
domain-level distribution distances, protein structures, biological sequence
homology, EHR concept relatedness, image/radiomics histograms, microbiome
beta-diversity, phylogenetic tree distances, mass-spectrometry spectra,
biomedical text embeddings, hyperspectral spectral signatures, point clouds,
SOAP atomistic kernels, graph kernels, patient similarity, and general
AI4Science shift factors.

The computation-strategy registry is the scale layer. It covers exact chunked
all-pairs computation, angular/random-projection LSH, dense-vector ANN,
FAISS-style compressed indexes, filtered ANN, p-stable Lp LSH, NN-Descent
neighbor graphs, SSD/disk-backed vector indexes, GPU-accelerated joins, hybrid
multimodal/multi-metric indexes, MIPS transforms, multi-vector late-interaction
search, MinHash/LSH, weighted MinHash, binary Hamming multi-index hashing,
streaming similarity joins,
filter-and-verify similarity joins, edit-distance string joins, structured
edit-distance search for graphs/trees/JSON/ASTs, sparse inverted indexes,
sequence seed-and-extend, genomic sketching, molecular fingerprint Tanimoto
indexing, mass-spectrometry sparse cosine acceleration, DTW lower-bound
pruning, matrix-profile and symbolic time-series indexes, blocking/filtering,
metric trees, trajectory indexing, Sinkhorn/Wasserstein approximations,
ontology/taxonomic semantic joins, graph-kernel sketches, kernel feature-map
approximations, learned candidate filters, privacy-preserving/federated
similarity search, pivot/permutation metric indexes, generic prefilter-rerank
pipelines, and distributed partitioned similarity search.

The operating-point registry is the targeted-evaluation layer. It covers random
IID baselines, group holdouts, leave-one-domain-out evaluation, temporal forward
splits, leave-site-out external validation, targeted intended-use validation,
spatial and spatiotemporal blocked validation, molecular scaffold/cluster/UMAP
splits, molecular property-extreme splits, bioactivity step-forward and
leave-assay-out splits, drug-target cold-start splits, chromosome holdouts,
regulatory cross-cell-type/assay holdouts, contiguous mutational-region splits,
sequence homology and protein-family holdouts, cross-species/taxon holdouts,
perturbation and perturbation-context holdouts, systematic-variation
confounder holdouts, imaging scanner/site holdouts, materials leave-cluster
splits, graph distribution-shift splits, and RNA structurally dissimilar splits.

## Benchmark Capsule Utility

List capsules:

```sh
python spectrae/spectra_benchmarks.py list
```

Fetch paper PDFs/pages for all capsules:

```sh
python spectrae/spectra_benchmarks.py fetch \
  --output-dir "$SPECTRA_ASSET_DIR"
```

Preview a safe fetch plan including shallow repo clones:

```sh
python spectrae/spectra_benchmarks.py fetch --include-repos --dry-run
```

Set `SPECTRA_ASSET_DIR` to a durable local scratch or project directory before
fetching benchmark assets. Avoid `/tmp` for benchmark assets.

## BOOM Numeric Mini-Audit

Run the first measured BOOM density audit:

```sh
PYTHONPATH="$SPECTRA_ASSET_DIR/boom_pilot/repos/boom" \
python -m spectrae.benchmark_runners.boom_numeric_mini_audit \
  --output-dir "$SPECTRA_ASSET_DIR/boom_numeric_pilot" \
  --split-file "$SPECTRA_ASSET_DIR/boom_numeric_pilot/10k_dft_data_with_ood_splits.csv" \
  --n-estimators 120
```

The runner writes `audit_card.json`, `split_stats.csv`,
`performance_by_overlap.csv`, `spectral_curve.svg`, and `report.md`.
