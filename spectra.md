# /spectra Protocol

`/spectra` is an agent-native protocol for exploring scientific model
generalization. It uses similarity definitions, computation strategies, and
operating-point registries to decide what to test next.

## Core Claim

One split is not enough. A useful scientific evaluation should measure model
performance over scientifically meaningful train-test novelty, with fresh model
training whenever the data and model family make that possible.

SPECTRA is not a fixed metric or checklist. It is an active investigation loop:
ask a generalization question, construct a similarity hypothesis, test whether
that hypothesis changes model behavior, then use the result to choose the next
question. The default search goal is to find a scientifically defensible axis on
which target-model performance changes; failed axes are negative findings that
drive the next trial. Failure to find such an axis in the current run is not
evidence that no axis exists or that the model is generally generalizable; it
means the agent must expand the search scope or report what additional data,
features, tasks, model access, or compute are needed.
When the missing ingredient is a public reference, annotation, ontology, or
dataset, the agent should try to acquire, map, and validate that resource before
reporting a blocker.
When public resources cannot be joined because local rows lack stable IDs, the
agent should inspect the upstream dataset source for hidden provenance before
attempting expensive matching/alignment or stopping.
When the current benchmark still cannot test the resulting mechanism hypothesis,
the agent should construct a defensible hypothesis-test dataset from
public/local resources and run fresh split-based SPECTRA experiments on that
dataset.

Dataset discovery should start from the portable SPECTRA dataset catalog before
general web search. The catalog records access mode, authentication, expected
fields, recommended axes, leakage risks, and scale guidance. A catalog entry is
a candidate resource, not an instruction to download the full dataset. For
multi-GB, credentialed, or much larger resources, the first SPECTRA action is a
manifest, bounded slice, public metadata plan, precomputed representation, or
user-authorized data path.

## Explanation Depth

SPECTRA separates three levels of evidence:

- Surface or model-space support: performance changes with generic overlap,
  local support, embedding distance, representation distance, fingerprints,
  sequence words, feature-space distance, or another measurement that mostly
  says the test example is unlike the training examples.
- Domain-proxy support: performance changes with a scientifically named but
  still broad proxy such as scaffold, motif count, assay label, batch, site,
  cell type, tissue, cluster, image acquisition setting, material family, or
  patient subgroup.
- Mechanism or deployment support: the proxy has been connected to a tested
  scientific, experimental, annotation, population, environment, or
  data-construction mechanism using controls, mediation, curated/public
  resources, prospective splits, or a constructed hypothesis-test dataset.

Surface and domain-proxy curves are useful because they reveal where to look
next. In investigation or paper-facing mode they are not terminal findings by
themselves. A supported proxy creates mechanism debt: the Investigator must ask
what the proxy means, what alternatives it confounds, what experiment would
distinguish those alternatives, and what data or resources are needed. The
Distiller should route back to the Investigator, Dataset Scout, or Dataset
Constructor until the debt is resolved, infeasibility is demonstrated with
recovery attempts, or the user explicitly requested only a screening run.

## Session Launcher

When a user supplies a scientific question and a model paper/reference,
`/spectra` should start with `start_spectra_audit_session`. The session launcher
returns a role graph, spawn plan, routing policy, artifact tree, and terminal
gate. In clients that support delegated workers, the host agent should spawn the
Investigator and Distiller roles and route Dataset Scout or Dataset Constructor
roles when needed. In clients without delegation, the same roles are executed
sequentially with the same artifacts and handoffs.

The MCP server does not itself force subagent creation; it provides the
orchestration contract that the host agent follows.

For the fully autonomous path, `/spectra` exposes a session runner:

- `prepare_spectra_audit_session`: creates the session directory, role prompts,
  initial Investigator work order, routing contract, and artifact tree without
  executing any agent.
- `run_spectra_audit_session`: runs the loop through a configurable host-agent
  command template. It is safe by default and does not execute roles unless
  `execute_roles=true` and an `agent_command_template` are provided.

The CLI mirrors this:

```text
spectra agent prepare --question ... --paper ... --model-description ... --out ...
spectra agent run --question ... --paper ... --model-description ... --out ... \
  --agent-command-template 'codex exec --prompt-file {prompt_path}' --execute
```

The front-door request can be broad:

```text
Use /spectra to assess the generalizability of this model.
Paper: ./paper.pdf
Model: ./model
Dataset: ./data
```

For this mode, `/spectra` first extracts the model paper's claimed
generalization settings, then lets the Investigator and Distiller narrow the
question through evidence. The user should not need to name similarity metrics,
spectral axes, or the internal roles.

There are two paper modes:

- `paper_claim_audit`: the paper's claims and datasets are the primary audit
  target.
- `beyond_paper_discovery`: the paper is context, but not a boundary. The
  Investigator must look for scientifically important external settings,
  public/local datasets, or constructed datasets that can expose generalization
  axes the paper did not test.

Use `beyond_paper_discovery` when the goal is to reproduce open-ended discovery
behavior such as finding ENCODE-style regulatory axes rather than only auditing
the Caduceus paper's VEP benchmark.

## SPECTRA Investigator

The agent identity is investigator, not procedure walker. The central state is
the hypothesis ledger, not the list of axes already tested. Registries,
splitters, metrics, and audit cards are tools the investigator uses; they are
not the objective.

The investigator loop is:

1. Observe what model behavior changed, did not change, and what was surprising.
2. Interpret what the pattern implies about the model, data, task, source, and
   possible confounders.
3. Maintain competing hypotheses for why the model generalizes or fails.
4. Choose the next experiment because it can distinguish live hypotheses.
5. If the current rows cannot falsify the sharpest live hypothesis, acquire,
   download, or construct a fit-for-purpose public/local dataset that can test it.
6. Update beliefs after the result: strengthen, weaken, split, or kill
   hypotheses.
7. Continue from the sharpest unresolved question.

Every new similarity axis must answer: what hypothesis does this test, what
result would falsify or weaken it, and what alternative hypothesis would become
more likely? Do not run another axis merely because it is available in the
registry.

External data follows the same rule. Do not download or construct datasets as a
procedural expansion. Pull another dataset only when the hypothesis ledger says
the current rows cannot distinguish the live explanations. The acquisition plan
must state which hypothesis the dataset can falsify, what fields are required,
what controls are needed, and what bounded first slice can run immediately.

## SPECTRA Distiller

Long Investigator runs can accumulate many proxy-level or mixed findings. The
Distiller is a read-only synthesis step that takes the curves, hypothesis
ledger, and belief updates, then returns the emerging scientific story and a
handoff experiment. Its job is to narrow the next question: which hypothesis is
most interesting, which result pattern made it plausible, which alternatives
remain live, and what experiment would actually distinguish them?

The Distiller must not run another generic axis sweep. It should tell the
Investigator what not to overclaim, for example when k-mer support is broad but
not mechanistic, motif support is task-specific rather than global, or a
model-space signal still needs a domain-level explanation.

When the loop appears to have produced a meaningful bounded insight, the final
Distiller must first check whether the evidence contains a stronger unresolved
axis. An attenuated or negative finding on the original hypothesis is not
terminal if another scientifically plausible axis produced a much larger model
degradation and has not yet been tested as the primary hypothesis with matched
controls. In that case, the Distiller must route back to the Investigator with
that axis as the new live hypothesis.

Only after that stronger-axis check passes is the final Distiller step a
paper-ready synthesis rather than another routing decision. It must take the
model paper, the Investigator trace, Dataset Scout/Constructor artifacts,
controls, negative findings, and final SPECTRA results and write a paper-ready
generalizability finding. This synthesis should explain what the original model
paper showed, what SPECTRA asked that the paper did not resolve, which signals
survived or attenuated under controls, and exactly what claim boundary follows.
The output must include interpretation, not only raw metrics.

Final Distiller synthesis outputs:

- `paper_ready_spectra_finding.md`
- `claim_boundary.json`
- `model_paper_context.md`
- `evidence_to_claim_table.csv`
- `overclaim_guardrails.md`

## Dataset Scout and Constructor

When the Distiller concludes that the current benchmark cannot answer the
scientific question, SPECTRA should first use a Dataset Scout unless the
replacement dataset is already justified. The Scout searches local and public
resources, compares candidate datasets, records rejected candidates, and keeps
the inconsistency ledger alive.

The Scout should query the dataset catalog before general web search. It should
prefer catalog entries with compatible scientific units and required fields, but
must reject or defer entries whose access terms, mapping fields, leakage risks,
or scale make the proposed experiment unrealistic.

The Scout returns:

- `inconsistency_ledger.json`
- `dataset_candidate_table.csv`
- `dataset_candidate_table.json`
- `candidate_resource_search_log.md`
- `rejected_candidates.md`
- `recommended_constructor_handoff.json`
- `scout_report.md`

The Distiller reviews Scout output before construction. It should promote a
candidate only when the Scout compared enough alternatives and the top candidate
actually distinguishes the live inconsistency.

After promotion, SPECTRA hands the problem to a Dataset Constructor. This role
recovers source provenance, builds the public/local dataset needed to test the
hypothesis, validates labels and mappings, audits leakage and confounders, and
returns SPECTRA-ready split candidates to the Investigator.

The Dataset Constructor does not decide whether the model generalizes. It
returns a dataset package:

- `dataset_card.md`
- `construction_manifest.json`
- `label_semantics.json`
- `provenance_table.csv`
- `sequence_table.parquet` or an equivalent domain table
- `spectra_ready_schema.json`
- `split_candidates/`
- `mapping_validation.json`
- `leakage_audit.json`
- `confounder_audit.json`
- `recommended_spectra_run.json`

The Investigator then runs fresh split-based SPECTRA experiments on that
package and keeps the constructed-dataset claim separate from the original
benchmark claim.

Before the package goes back to the Investigator, the Distiller reviews it. The
Distiller checks whether required artifacts exist, mappings validate, cross-
source leakage is controlled, labels are sufficiently documented, and at least
one SPECTRA split candidate is executable. If only nonblocking caveats remain,
the Distiller should hand the package to the Investigator rather than keeping
construction open indefinitely.

Required investigator artifacts:

- `observations.md`
- `hypothesis_ledger.json`
- `competing_explanations.md`
- `why_this_next_experiment.md`
- `discriminating_experiment_plan.json`
- `falsifiable_predictions.json`
- `belief_update.md`
- `hypothesis_driven_acquisition_plan.json` when current rows cannot test the
  live mechanism hypothesis
- `external_dataset_decision_log.json`

A checkpoint that only says "mixed results" or "mechanism debt remains" is not
valid. It must explain what the mixed pattern implies and launch or complete the
experiment that best distinguishes the remaining explanations.

Supported proxy axes create mechanism debt. The agent cannot satisfy SPECTRA by
writing that an axis is proxy-level, running one local proxy-like test, and
listing deeper mechanism work as future work. Local input-derived mechanism or
mediation tests are the first execution tier, not a stopping condition. If those
tests do not produce a mechanism-level explanation with controls, the agent must
continue to source-provenance recovery, public-resource mapping, constructed
public/local hypothesis-test datasets, residual axes, broader task coverage, or
a launched continuation job. A queued-only manifest is not enough; if the larger
continuation cannot run in the current workspace, the agent must launch a
bounded fallback slice and record the concrete blocker for the larger job.

Task coverage must be comprehensive. The agent should run all feasible tasks,
datasets, seeds, or model settings for the chosen question. If all tasks cannot
be run, it must first enumerate the full task set, run an all-task data screen
when possible, rank tasks by split suspiciousness, scientific diversity, source
family, feasibility, sample size, and prior/internal signals, then justify every
omitted task with a concrete blocker and launch evidence for a continuation or
bounded fallback. A self-selected "representative" subset is preliminary only.

## Mode Hierarchy

SPECTRA has two execution modes.

1. Benchmark mode is the default. Use it whenever raw labeled data and a
   trainable model, model code, or defensible lightweight baseline are available.
   In benchmark mode, construct similarity-controlled train/test splits and
   train a fresh model for each split before evaluating held-out performance.
2. Audit mode is a fallback. Use it only when retraining is impossible because
   only fixed predictions, a closed model, or incomplete labels are available.
   Audit-mode curves are diagnostic fixed-model analyses. They can motivate a
   benchmark-mode experiment, but they are not sufficient evidence for the core
   SPECTRA generalization claim.

## Agent Procedure

1. Read the paper and associated repository.
2. Identify the scientific unit of generalization.
3. Decide the execution mode. If raw labels and a trainable model or baseline
   are available, benchmark mode is required.
4. If the target model cannot import or run, inspect the repository, dependency
   files, setup scripts, model cards, and local artifacts. Attempt a reasonable
   allowed environment or code-path repair before downgrading to a surrogate
   baseline.
5. Start investigator mode: write the initial uncertainty, live hypotheses, and
   what first experiment could separate them.
6. State the first scientific question about generalization.
7. Reproduce one core generalization finding from the paper when the paper is
   part of the task; otherwise establish the strongest ordinary baseline split.
8. Propose one or more spectral properties tied to deployment novelty.
9. Treat each property as a similarity hypothesis.
10. Classify each hypothesis as prospective, post-hoc explanatory, or invalid.
11. Declare an initial axis-search scope: candidate axis classes, approximate
   compute budget, minimum target-model behavioral tests, and what additional
   inputs would be needed if the current scope fails. This is not a stopping
   budget; exhaustion of the current scope triggers a concrete continuation.
12. Declare a task-coverage plan. Enumerate all available tasks/datasets, rank
   them using all-task screens and scientific diversity, then run all feasible
   tasks or document a concrete blocker for each omitted task. Do not stop after
   an arbitrary representative subset.
12. Construct or approximate the first spectral property graph.
13. If an axis is only screened at the data level, mark it as a hypothesis and
   choose a targeted feasible behavioral follow-up.
14. In benchmark mode, generate overlap-controlled train/test splits across at
   least three measured overlap levels.
15. In benchmark mode, train a fresh model for each split with the same training
   recipe and evaluate only the corresponding held-out test units.
16. In audit fallback mode, generate fixed-prediction evaluation subsets only
   after recording why retraining is impossible.
17. Validate that cross-split overlap decreases.
18. Evaluate model performance over the split spectrum.
19. Score the curve as monotonic, localized, weak, non-explanatory, or not evaluable.
20. Decide the next experiment. There is no terminal stop action inside the
   SPECTRA loop:
   - If an axis is suspicious but model behavior has not been tested, run a
     behavioral split/probe/head experiment.
   - If an axis is non-explanatory, record that finding and test the next axis
     from an untested class while the current search scope remains.
   - If an axis is localized, refine the axis or test a composite similarity.
   - If an axis is supported, immediately replicate it on another task, dataset,
     seed, model setting, or stronger operating point when those inputs are
     available. Do not stop with a proposal when replication can be executed in
     the current workspace.
   - After replicated support, classify explanatory depth and continue mechanism
     debt work before selecting the axis as a completed finding.
21. Compute AUSPC or an equivalent spectral summary for supported behavioral axes.
22. After a candidate degradation axis is found, build a replication evidence
   table showing where the same axis is supported, weak, localized, or
   non-explanatory across available tasks/datasets.
23. If tested tasks are negative or mixed and untested feasible tasks remain,
   expand task coverage before treating the run as negative or exhausted.
24. Reflect on the replication pattern. Compare tasks where the axis works
   against tasks where it fails, infer what the axis is really measuring, and
   use that contrast to derive residual or composite axes. For example, if both
   k-mer and model-representation support are partially replicated, test whether
   representation support still matters after matching sequence similarity, and
   test motif-family, task-context, provenance, or class-conditional axes.
25. If the supported axis is a model-space or embedding-space axis, translate it
   into domain hypotheses before treating it as a scientific explanation. Ask
   what the model appears to learn smoothly and what it does not learn smoothly
   from the data. Annotate high- and low-support embedding regions with
   biological or scientific features, then test at least one domain-interpretable
   axis when feasible.
26. Assess explanatory depth for every supported proxy or domain-interpretable
   axis. Do not stop at k-mers, GC/CpG, length, entropy, crude word counts,
   task labels, metadata labels, generic embeddings, or other shallow proxies
   when curated annotations, mechanism-level features, context interactions, or
   mediation/residual tests are feasible.
27. Enforce mechanism debt for every supported proxy, broad domain proxy,
   model-space pointer, or mechanism-like axis without controls. Call the
   mechanism-debt gate and execute a local-derived mechanism or mediation test
   when possible. In sequence tasks, missing coordinates do not excuse skipping
   motif/PWM scans, motif grammar, GC/length/k-mer matched residual curves,
   CpG/low-complexity/repeat-like controls, or class-conditional curves.
28. If the next mechanism-level test needs resources absent from the local
   bundle, search for public resources, check license/version/size, download or
   query a useful resource when feasible, map it to local scientific
   units, validate that mapping, and continue. If coordinates or stable
   IDs are missing, first inspect original dataset repositories, loaders,
   download scripts, manifests, dataset cards, raw archives, and paper
   supplements for recoverable provenance.
29. If the original benchmark still cannot test the mechanism hypothesis after
   local-derived mechanism tests, provenance recovery, and resource acquisition,
   construct a defensible dataset from public/local resources. Define the
   scientific unit, labels, provenance, mapping validation, leakage controls,
   and confounder controls, then run fresh benchmark-mode SPECTRA splits on that
   constructed dataset. Report this as hypothesis-test evidence, not as a
   replacement for the original benchmark.
30. Write checkpoint reports, not terminal reports, until the audit has full
   feasible task coverage and a mechanism-level explanation with controls.
   Checkpoints must include the next executable experiment and launch it when it
   can run in the current workspace. If the full experiment cannot be launched,
   launch a bounded fallback slice and record why the full job is blocked.
   Writing a queued-only continuation manifest is not sufficient.
31. If the declared axis/data/compute scope is exhausted, do not stop. Escalate
   to additional data, features, tasks, model access, public resources,
   constructed hypothesis-test datasets, or compute. Do not report this as
   evidence of universal generalization.
32. When a meaningful bounded insight seems to have been reached, call the final
   Distiller synthesis. It must first check for stronger unresolved alternative
   axes. If such an axis exists, route back to the Investigator with that axis
   as the new primary hypothesis. Otherwise, contextualize the result against
   the model paper and write the paper-ready claim, interpretation, evidence
   boundary, and overclaim guardrails. This is the only terminal report shape;
   it must explain the finding scientifically, not merely list artifacts.

## Exploration Rule

Data-level overlap findings are hypotheses, not conclusions about model
generalization. Exact duplicates, high train-test similarity, high metadata
overlap, or suspicious split geometry must trigger a targeted feasible
behavioral test: retrain a downstream probe/head/baseline on controlled splits
and measure performance versus overlap.

For pretrained foundation models, benchmark-mode retraining means fresh
downstream probes, fine-tuning heads, adapters, or defensible baselines per
controlled split. It does not require pretraining the foundation model from
scratch.

For target-model benchmarks, run the target model if possible. A surrogate
baseline is a control, not a replacement for the target-model behavioral result.

Broad SPECTRA searches should span multiple axis classes when inputs allow:
surface/content similarity, target-model representation similarity,
metadata/context/provenance similarity, and scientific-mechanism similarity. An
agent should not stop after only k-mer, exact-identity, or other surface-level
axes unless richer axis classes are unavailable or outside the declared compute
budget; in that case it should request the missing features or compute rather
than conclude the model has no degradation axis.

Task coverage is part of the scientific claim. The agent should not choose a few
tasks because they feel representative. It should either run all feasible tasks
or produce a task-coverage plan showing the full task inventory, all-task screen
results, ranking criteria, selected tasks, omitted tasks, and concrete blockers.
Any negative or inconclusive result from a limited subset must be labeled
preliminary and must trigger expansion to remaining feasible tasks through a
launched continuation job or launched bounded fallback.

Replicated support is not the end of the investigation when support is mixed.
The agent should ask why an axis degrades performance in some tasks but not
others. It should build a supported-versus-failed task contrast and use that
contrast to test residual axes, such as model-representation support after
matching sequence similarity, motif-family support, metadata/provenance
interactions, task-family interactions, or class-conditional curves.

Model-space similarity is not itself a scientific explanation. If a model
embedding or representation axis is supported, the agent should treat it as a
pointer to what the model may or may not have learned from the data. It should
translate the embedding geometry into domain hypotheses by annotating high- and
low-support regions with scientifically meaningful variables. For Caduceus or
other sequence models, those variables include motif families, regulatory
element class, chromatin or assay context, GC/CpG and low-complexity structure,
species/provenance, and class-conditional sequence patterns.

Proxy findings are also not final explanations. A k-mer, GC/CpG, sequence
identity, length, entropy, low-complexity, crude regulatory-word, task-family,
metadata, or generic embedding axis can identify where performance changes, but
SPECTRA should ask what mechanism the proxy stands in for. The next step is to
test curated annotations, mechanism-level features, context interactions, or
mediation/residual curves. For sequence models, that means moving from surface
sequence support toward motif families, motif grammar, positional logic,
chromatin or assay context, conservation, repeats, regulatory element class, and
whether those variables explain the residual model-space signal after matching
simpler proxies.

Mechanism debt is a hard gate, not a suggestion. A supported proxy/model-space
axis is not closed by `mechanism_axis_scores.json` or
`mediation_test_results.json` if those artifacts still describe proxy-level
variables. Those artifacts only advance the loop. The run can only mark the
axis mechanism-resolved when the evidence includes a mechanism-level explanation
with controls, or it must continue through source-provenance recovery,
public-resource acquisition/mapping, constructed public/local hypothesis-test
datasets, residual axes, broader task coverage, or a launched continuation job.
"Coordinates are missing" is not enough when sequence-derived mechanism tests
can still be run, and running those sequence-derived tests is not enough when
they remain shallow proxies.

Missing mechanism resources are active search targets, not immediate blockers.
If a hypothesis needs curated public data, the agent should search official or
canonical sources, download or query a bounded version, record provenance,
map the resource to the dataset, validate the mapping, and run the next
mechanism or mediation test. For genomics, examples include reference genomes,
JASPAR/HOCOMOCO motifs, ENCODE cCRE/chromatin/TF tracks, conservation, repeats,
CpG islands, GENCODE/RefSeq, dbSNP, GTEx, and other assembly-matched resources.

If the local dataset contains only sequence strings, do not jump straight from
"no coordinates" to "unmappable." First inspect the original benchmark source:
repository files, dataset loaders, download scripts, source manifests, raw
FASTA/BED/CSV archives, paper supplementary tables, dataset cards, and package
metadata. Those sources often contain coordinates, accessions, or stable IDs
that were dropped from processed train/test files. Sequence-to-genome alignment
is a later recovery step, after source provenance fails.

The same provenance rule applies outside genomics. For molecules, recover
SMILES/InChIKey, compound IDs, assay IDs, target IDs, source database accessions,
or raw assay tables before querying ChEMBL/PubChem/BindingDB/UniProt/PDB. For
single-cell or perturbation tasks, recover gene IDs, guide IDs, perturbation
IDs, drug IDs, cell type, donor, batch, tissue, and protocol metadata before
joining GO/Reactome/LINCS/CELLxGENE/OpenTargets. For clinical or imaging tasks,
recover site/time/patient/encounter/code metadata or DICOM/scanner/study fields
before joining terminology, cohort, device, or acquisition resources.

If the current benchmark still cannot answer the mechanism question after those
steps, SPECTRA must construct a hypothesis-test dataset from public/local
resources, or launch a bounded construction fallback with a blocker for the full
construction. This is not a shortcut around the original
benchmark; it is a way to
test the proposed mechanism directly. The constructed dataset must have explicit
scientific units, labels, resource provenance, mapping validation, prospective
split features, leakage controls, matched or stratified confounder controls, and
fresh per-split training or probing when benchmark mode is feasible.

If the agent cannot run the behavioral test, it must report a concrete blocker:
missing labels, missing features, unavailable model code, missing checkpoint,
infeasible compute, or an environment failure. Missing packages or import
failures are not terminal blockers until the agent has inspected the target
repository and attempted an allowed environment/data/code repair. A data-only
screen may be reported as pre-benchmark evidence only.

Likewise, a missing annotation or reference is not a terminal blocker until the
agent has tried source-provenance recovery and public acquisition when network
access and licensing permit it. The blocker should name the actual failure: no
mappable identifiers, licensing or credential barrier, unavailable version,
excessive download size, network failure, or failed mapping validation. "No
mappable identifiers" means no identifiers were found after inspecting upstream
dataset provenance, not merely no IDs in the processed local CSV/parquet.

## Selection Rule

Select the strongest scientifically defensible, leakage-aware novelty axis, not
merely the curve that looks most monotonic.

Prospective axes can be used before labels are known and support split design.
Post-hoc axes can explain observed failures after labels are available, but must
not be presented as prospective validation axes.

If benchmark mode is feasible, a post-hoc fixed-prediction curve must not be
presented as the primary SPECTRA result.

## Required Audit Artifacts

- `audit_card.json`
- `spectral_properties.json`
- `mode_decision.json`
- `question_trace.json`
- `task_coverage_plan.json`
- `task_screen_ranking.csv`
- `task_selection_rationale.json`
- `untested_task_blockers.json`
- `split_stats.csv`
- `performance_by_overlap.csv`
- `retraining_manifest.csv`
- `spectral_curve.png`
- `similarity_hypothesis_scores.json`
- `axis_search_budget.json`
- `replication_evidence.csv`
- `replication_reflection.json`
- `supported_vs_failed_task_contrast.csv`
- `residual_axis_candidates.json`
- `residual_axis_scores.json`
- `model_space_biological_translation.json`
- `embedding_support_biological_contrast.csv`
- `domain_hypothesis_axis_candidates.json`
- `domain_hypothesis_scores.json`
- `explanatory_depth_assessment.json`
- `proxy_to_mechanism_plan.json`
- `mechanism_debt_register.json`
- `mechanism_execution_manifest.json`
- `mechanism_infeasibility_proof.json`
- `public_resource_acquisition_plan.json`
- `public_resource_search_log.json`
- `source_provenance_recovery_log.json`
- `hypothesis_test_dataset_plan.json`
- `constructed_dataset_manifest.json`
- `constructed_dataset_provenance.json`
- `constructed_dataset_schema.json`
- `constructed_dataset_mapping_validation.json`
- `constructed_dataset_leakage_audit.json`
- `constructed_dataset_spectra_results.json`
- `resource_manifest.json`
- `resource_mapping_validation.json`
- `mechanism_axis_scores.json`
- `mediation_test_results.json`
- `paper_ready_spectra_finding.md`
- `claim_boundary.json`
- `model_paper_context.md`
- `evidence_to_claim_table.csv`
- `overclaim_guardrails.md`
- `next_experiment.json`
- `blockers.json`
- `report.md`
