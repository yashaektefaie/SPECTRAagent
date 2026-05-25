# Generalizability Analysis Procedure

Version: 0.5.1

## Purpose

Evaluate whether a model generalizes across decreasing train-test similarity,
instead of relying only on a random benchmark split. When raw labeled data and a
trainable model, model code, or defensible lightweight baseline are available,
this means constructing similarity-controlled train/test splits and retraining a
fresh model for each split.

SPECTRA is an active similarity-hypothesis investigation. The agent should not
run one similarity metric and stop, and it should not stop at data geometry when
model behavior can be tested. It should ask generalization questions, test
scientifically plausible notions of similarity, score whether each one explains
performance degradation, record negative findings, and use those findings to
choose the next behavioral experiment, similarity definition, or computation
method. The default search goal is to uncover a scientifically defensible axis
where target-model performance changes. Non-explanatory axes are not stop
conditions while plausible axis classes, data, and compute remain. Exhausting
the current search scope is not evidence that no degradation axis exists and
must not be reported as evidence of universal generalization. Replicated support
is also not the end when support is mixed; the agent must explain the
heterogeneity and use it to derive the next residual or composite axes. When the
supported axis is model-space or embedding-space similarity, the agent must
translate that behavioral signal into scientific or biological hypotheses before
treating it as an explanation. A surface proxy or broad domain proxy is also not
an explanation by itself. SPECTRA should use proxy findings to push toward
mechanistic hypotheses about what the model learns smoothly, what it fails to
learn, and which scientific variables mediate the degradation. If the next
mechanism experiment needs public references or annotations that are absent from
the local bundle, the agent should try to acquire, map, and validate those
resources before reporting a blocker. If external resources cannot be joined
because local files lack stable identifiers, the agent should first inspect the
upstream dataset source, loaders, raw archives, manifests, documentation, and
supplementary files for recoverable provenance before attempting expensive
matching/alignment or reporting a mapping blocker.
If the current benchmark still cannot test the resulting mechanism hypothesis,
the agent must construct a defensible hypothesis-test dataset from public/local
resources and run fresh split-based SPECTRA experiments on that dataset, or
launch a bounded construction fallback with a concrete blocker for the full
construction.
Supported proxy axes create mechanism debt. The agent cannot satisfy this
procedure by saying the axis is proxy-level, running one local proxy-like test,
and listing deeper mechanism work as future work. Local input-derived mechanism
or mediation tests are the first tier, not a stopping condition. If those tests
do not produce a mechanism-level explanation with controls, the agent must
continue to source-provenance recovery, public-resource mapping, constructed
public/local hypothesis-test datasets, residual axes, broader task coverage, or
a launched continuation job. A queued-only manifest is not enough; if the larger
continuation cannot run in the current workspace, the agent must launch a
bounded fallback slice and record the concrete blocker for the larger job.

When the user supplies a generalization question and a model paper/reference,
the agent should call `start_spectra_audit_session` first. The returned session
object defines the Investigator, Distiller, Dataset Scout, Dataset Constructor,
and final Synthesis Distiller roles; clients with subagent support should spawn
those roles as directed, while single-agent clients should execute the same role
passes sequentially.
For broad requests such as "assess the generalizability of this model", the
first Investigator task is paper-first claim extraction: what generalization
claims does the paper make, what settings were actually evaluated, and which
claims remain under-tested enough to become SPECTRA hypotheses?

SPECTRA uses a cheap-first behavioral runtime policy. The agent should not
begin with expensive optimized classifiers, fine-tuning, all-pairs graph
construction, large ANN indexes, or full-dataset embedding when a bounded
behavioral slice can first test whether the live hypothesis has signal. Start
with data/schema/leakage checks, simple controls, cached or chunked
representations, and deterministic non-iterative probes such as
nearest-centroid/prototype scores, mean-difference linear scores,
kNN/prototype retrieval, or closed-form small linear baselines. Escalate to
logistic/SVM/ridge heads, fine-tuning, full-registry embeddings, or full-scale
similarity graphs only when the cheap probe is inconclusive or when the user’s
claim requires stronger calibrated deployment evidence. Before any heavy step,
write the expected time/resource budget, success criterion, timeout/fallback
condition, and cheaper fallback. After one timeout or runtime failure from a
solver family, switch to the fallback or a smaller slice rather than retrying
the same slow method repeatedly. Cheap-probe evidence is bounded evidence: it
can support exploratory applicability or screening recommendations, but
production claims require calibrated/tuned follow-up.

## Investigator Mode

The agent is a SPECTRA Investigator, not a procedure walker. The hypothesis
ledger is the active control state. Similarity definitions, computation
strategies, splitters, and reports are tools used to answer scientific
questions; they are not the goal.

The required loop is observe, interpret, hypothesize, discriminate, update, and
continue:

1. Observe what model behavior changed, stayed flat, contradicted expectation,
   or surprised the agent.
2. Interpret what the pattern implies about the model, task, data source,
   possible confounders, and possible mechanisms.
3. Maintain competing hypotheses for why performance changes or fails to
   change.
4. Choose the next experiment because it can separate live hypotheses.
5. If current rows cannot falsify the sharpest live hypothesis, acquire,
   download, or construct a fit-for-purpose public/local dataset that can test it.
6. Update the hypothesis ledger after the result.
7. Continue from the sharpest unresolved scientific question.

Do not run a similarity axis merely because it is available in the registry.
Before every experiment, state which hypotheses it distinguishes, what outcome
would weaken or falsify each one, and why this experiment is more informative
than a generic axis sweep.

Do not pull external data as a checklist expansion. Pull it only when the
hypothesis ledger shows that the current data cannot distinguish live
explanations. The dataset acquisition plan must name the hypothesis being
tested, required fields, leakage and confounder controls, bounded first slice,
and how the result will update beliefs.

When the Investigator has produced many mixed or proxy-level results, insert a
SPECTRA Distiller step before launching another broad run. The Distiller is
read-only: it turns curves, the hypothesis ledger, and belief updates into a
ranked scientific story plus a concrete handoff experiment. It should sharpen
the next question and identify overclaim risks, not execute a checklist.

When the loop appears to have produced a meaningful bounded insight, insert a
final Distiller synthesis step. Before writing a terminal report, this step must
check whether the evidence contains a stronger unresolved alternative axis. An
attenuated or negative result on the original hypothesis is not terminal if
another scientifically plausible axis produced a much larger model degradation
and has not yet been tested as the primary hypothesis with matched controls. In
that case, the Distiller routes back to the Investigator with that axis as the
new live hypothesis. Only after this stronger-axis check passes should the
Distiller write the paper-ready generalizability finding from the model paper,
Investigator trace, Dataset Scout/Constructor artifacts, controls, negative
findings, and final SPECTRA results. The output must include interpretation,
not only raw metrics or artifact lists.

If the Distiller determines that the current benchmark cannot answer the
scientific question, create a Dataset Scout handoff before construction unless
the replacement dataset is already justified. The Scout compares public/local
candidate datasets and resources, records rejected candidates, and preserves the
inconsistency ledger. The Distiller then decides whether to continue scouting or
promote a candidate to Dataset Constructor.

The Constructor recovers provenance, builds public/local resources, validates
labels and mappings, audits leakage and confounders, and returns SPECTRA-ready
split candidates. It does not make the model-generalization claim; it creates
the dataset package the Investigator needs for the next fresh split-based audit.
The Distiller then reviews the returned package and either hands it to the
Investigator for a bounded SPECTRA run or returns it to the Constructor with
blocking fixes. Nonblocking caveats such as inferred label names or unknown
assembly should constrain claim strength, but they should not cause indefinite
construction if the package can already test the next hypothesis.

Required investigator artifacts:

- `observations.md`
- `hypothesis_ledger.json`
- `competing_explanations.md`
- `why_this_next_experiment.md`
- `runtime_budget_and_fallbacks.md`
- `discriminating_experiment_plan.json`
- `falsifiable_predictions.json`
- `belief_update.md`
- `hypothesis_driven_acquisition_plan.json` when current rows cannot test the
  live mechanism hypothesis
- `external_dataset_decision_log.json`

A checkpoint that only says results are mixed, an axis is proxy-level, or
mechanism debt remains fails investigator mode unless it also gives competing
explanations, a belief update, and a launched or completed discriminating
experiment.

Task coverage must be comprehensive. The agent should run all feasible
tasks/datasets/seeds/model settings for the chosen question. If all tasks cannot
be run, it must enumerate the full task set, run an all-task data screen when
possible, rank tasks by split suspiciousness, scientific diversity, source
family, feasibility, sample size, and prior/internal signals, and justify every
omitted task with a concrete blocker and launch evidence for a continuation or
bounded fallback.

## Scientific Contract

The analysis is valid only when all of the following are true:

1. The dataset samples are clearly identified.
2. The predictive task and model output are defined.
3. The execution mode is recorded as benchmark mode or audit fallback mode.
4. Benchmark mode is used whenever raw labels and a trainable model or baseline are available.
5. The current scientific question and question trace are recorded.
6. Investigator mode is active: observations, competing hypotheses, falsifiable predictions, and belief updates are maintained.
7. One or more spectral properties are proposed as similarity hypotheses.
8. The spectral property graph is constructed exactly or approximated with a documented method.
9. Each axis is labeled as prospective, post-hoc explanatory, invalid, or pre-benchmark screening.
10. Data-level overlap findings are followed by a targeted feasible behavioral test.
11. An initial axis-search scope is declared before continuation criteria are considered: candidate axis classes, approximate compute budget, minimum target-model behavioral tests, and required scope expansions if current inputs fail. Scope exhaustion is not a terminal state.
12. A task-coverage plan is declared before behavioral testing. The plan enumerates all available tasks/datasets, ranks them using all-task screens and scientific diversity, and either runs all feasible tasks or documents concrete blockers for omitted tasks. A self-selected representative subset is preliminary only.
12. Behavioral runtime follows cheap-first escalation: bounded slices, controls, cached/chunked representations, and deterministic non-iterative probes precede expensive solvers or full-scale computation; every heavy step has a declared budget and fallback, and repeated retries of the same slow solver family are invalid.
12. In benchmark mode, train-test splits are generated across at least three spectral parameters.
13. In benchmark mode, a fresh model is trained independently for each split using the same training recipe.
14. In audit fallback mode, evaluation subsets are used only after documenting why retraining is impossible.
15. Missing packages, missing repo setup, or absent local imports are not terminal blockers until the agent inspects the target repository and attempts a reasonable environment/data/code repair.
16. Cross-split overlap decreases as spectral parameter increases, allowing only small stochastic deviations.
17. Model performance is reported as a function of both spectral parameter and measured cross-split overlap.
18. Candidate curves are scored as monotonic, localized, weak, non-explanatory, or not evaluable.
19. Every axis result has a next-experiment decision: behaviorally test, refine, execute candidate-axis replication, reflect on mixed replication, classify explanatory depth, enforce mechanism debt, translate model-space axes into domain hypotheses, try another axis, expand task coverage, attempt blocker recovery, construct public/local data, launch a continuation job or fallback, or escalate current-scope exhaustion. There is no terminal stop action inside the SPECTRA loop.
20. Supported proxy axes are assessed for explanatory depth. Surface proxies, broad metadata proxies, crude word counts, and generic embedding distances must trigger curated-annotation, mechanism-level, context-conditioned, or mediation tests when inputs allow.
21. Supported proxy, domain-proxy, model-space, or uncontrolled mechanism-like axes create mechanism debt. The debt is not satisfied by shallow local proxy tests. Those tests only advance the loop. Mechanism debt remains active until a mechanism-level explanation with controls is obtained, or the agent launches the next public-resource, constructed-dataset, residual-axis, broader-task, or compute-expansion step and records launch evidence.
22. Missing local annotations, references, ontologies, coordinates, or public benchmark resources trigger source-provenance recovery and public-resource acquisition attempts before they are treated as blockers.
23. The report distinguishes poor model performance, invalid split construction, non-explanatory similarity, proxy-level explanation, unsatisfied mechanism debt, preliminary limited-task evidence, pre-benchmark screening, recovery-attempted blockers, source-provenance recovery failures, public-resource acquisition failures, constructed hypothesis-test dataset evidence, current-scope exhaustion, and audit-mode limitations.

If a spectral property is not scientifically defensible, abandon that property and
choose the next axis; do not terminate the audit.

## Required Inputs

- Dataset description.
- Sample identifier definition.
- Prediction target.
- Model or model family.
- Raw labels and features when benchmark-mode retraining is feasible.
- Model training code or a defensible lightweight baseline when the original model cannot be retrained.
- Fixed predictions only when audit fallback mode is unavoidable.
- Candidate spectral properties or enough schema information to retrieve them from the registry.
- Similarity type: binary or weighted.
- Dataset size.
- Compute constraints.
- Evaluation metric.

## Required Outputs

- Spectral property rationale.
- Execution mode decision and fallback reason, if any.
- Current question and question trace.
- Observations, hypothesis ledger, competing explanations, falsifiable predictions, discriminating-experiment rationale, and belief update.
- Task coverage plan, all-task screen ranking, task-selection rationale, and blockers for omitted tasks.
- Similarity hypothesis order.
- Axis-search scope and tested axis classes.
- Similarity computation plan.
- Split generation plan.
- Retraining plan and per-split training manifest for benchmark mode.
- Split validation summary.
- Model evaluation plan.
- Curve score per tested similarity hypothesis.
- Failed-axis findings.
- Replication evidence table for candidate axes.
- Replication reflection comparing supported and non-supported tasks.
- Residual or composite axis candidates derived from the replication pattern.
- Residual-axis scores when those axes are executable.
- Model-space-to-domain translation for supported embedding or representation axes.
- Biological/scientific contrast between high- and low-support model-space regions.
- Domain-hypothesis axis candidates and scores when those axes are executable.
- Explanatory-depth assessment for supported proxy or domain-interpretable axes.
- Proxy-to-mechanism plan, mechanism-debt register, mechanism execution manifest, mechanism-axis scores, and mediation/control results when executable.
- Continuation manifest when no deeper mechanism or mediation test can be executed immediately.
- Public-resource acquisition plan, source-provenance recovery log, search log, resource manifest, mapping validation, and acquisition blockers when local resources are insufficient.
- Hypothesis-test dataset plan, constructed dataset manifest, provenance, schema, mapping validation, leakage audit, split stats, retraining manifest, SPECTRA results, and limitations when the current benchmark cannot test a mechanism hypothesis.
- Selected similarity axis and leakage risk, or current-scope exhaustion with concrete expansion requirements.
- Next-experiment decision per axis.
- Final paper-ready SPECTRA finding when the loop has a meaningful bounded insight.
- Model-paper context, evidence-to-claim table, claim boundary, and overclaim guardrails for the final finding.
- Concrete blockers when behavioral testing is not run.
- Generalizability report outline.
- Assumptions and limitations.

## Recommended Steps

1. Identify samples and prediction target.
2. Decide whether benchmark mode is feasible. If raw labeled data and a trainable model or baseline are available, benchmark mode is required.
3. If the target model cannot import or run, inspect repository dependency files, setup scripts, model cards, and local artifacts; attempt a reasonable installation, environment creation, or code-path repair before falling back to a surrogate baseline.
4. State the current scientific question.
5. Call `plan_iterative_similarity_search` to create a candidate axis order.
6. Declare an axis-search scope covering candidate axis classes, compute budget, and the data/features/tasks/model access/compute needed if the current scope fails.
7. Declare a task-coverage plan. Enumerate all available tasks/datasets, run an all-task data screen when possible, rank tasks by suspiciousness/diversity/feasibility/prior signals, and run all feasible tasks or document a blocker for each omitted task.
8. Declare the cheap-first runtime plan: bounded first slice, cheap controls, non-iterative probe, escalation criterion, and fallback if a heavy step times out.
9. Propose or validate the first spectral property.
10. Decide whether the spectral property graph should be binary or weighted.
11. Choose exact pairwise similarity, blocked pairwise similarity, approximate nearest neighbors, hashing, or domain-specific indexing.
12. If only data geometry has been tested, mark the result as pre-benchmark screening and choose a behavioral follow-up.
13. In benchmark mode, generate overlap-controlled train/test splits from the chosen similarity graph.
14. In benchmark mode, train a fresh model for each split and save a retraining manifest with split IDs, train IDs, test IDs, seeds, training command, and model artifact path.
15. In audit fallback mode, generate `pairwise_similarity.csv`, a measured spectral axis, or fixed-prediction evaluation subsets only after documenting why retraining cannot be done.
16. Run `run_spectra_audit` or an equivalent benchmark-mode evaluator on the candidate axis.
16. Score the resulting curve with `score_similarity_hypothesis_curve`.
17. Call `decide_next_spectra_experiment` after each axis result. For supported axes, pass the current explanatory-depth classification and mechanism-debt status; a replicated axis cannot terminate the loop while depth is unknown or mechanism debt is unresolved.
18. If the curve is non-explanatory or not evaluable, record that finding and try the next similarity hypothesis from an untested axis class while budget remains.
19. If tested tasks are negative or mixed and untested feasible tasks remain, expand task coverage before reporting the search as negative or exhausted.
20. If the curve is localized, refine the axis or try a composite axis around the failure region.
21. If the curve is monotonic or weakly supported, immediately replicate it on another task, dataset, seed, model setting, or stronger operating point when feasible; do not stop with a proposal when replication inputs are available.
22. After replication, compare supported and non-supported tasks/datasets and infer what the current axis is actually measuring.
23. If replication support is mixed, test at least one residual or composite axis when feasible, such as representation support after matching sequence similarity, motif-family support, task-context/provenance interactions, or class-conditional curves.
24. If a model-space or embedding-space axis is supported, translate it into domain hypotheses: annotate high- and low-support regions with available scientific features, ask what the model appears to learn smoothly or not learn smoothly, and test at least one domain-interpretable axis when feasible.
25. For every supported proxy or domain-interpretable axis, call `assess_explanatory_depth`. If the axis is k-mer support, GC/CpG, length, entropy, crude regulatory words, task labels, metadata/provenance, generic embeddings, or another proxy, use it to generate and test a curated annotation, mechanism-level feature, context interaction, or mediation/control experiment.
26. For every supported proxy, broad domain proxy, model-space pointer, or mechanism-like axis without controls, call `enforce_mechanism_debt_gate`. A proxy-only report is only a checkpoint. If local inputs contain sequences, structures, images, text, tables, metadata, or feature matrices, execute local input-derived mechanism or mediation tests, then continue to public-resource or constructed-dataset mechanism tests if the local tests remain proxy-level.
27. If the next mechanism test requires data not in the local bundle, call `plan_public_resource_acquisition`. If mapping identifiers are missing, inspect upstream dataset repositories, loaders, raw archives, manifests, README files, dataset cards, and supplementary files for coordinates or stable IDs before attempting alignment. Then search for official or canonical public resources, check license/version/size, download a useful resource when feasible, map it to local units, validate the mapping, and run the next mechanism or mediation test.
28. If the original benchmark still cannot test the mechanism hypothesis after local-derived mechanism tests, provenance recovery, and resource acquisition, call `plan_hypothesis_test_dataset_construction`. Construct a defensible dataset from public/local resources, or launch a bounded construction fallback with a concrete blocker for the full construction; define scientific units and labels, validate mappings, audit leakage, control obvious confounders, and run fresh benchmark-mode SPECTRA splits on that constructed dataset.
29. Plot performance against measured overlap.
30. Write a checkpoint report showing where performance degrades, the replication evidence table for candidate axes, the replication reflection, residual-axis results, model-space/domain translation, explanatory-depth assessment, mechanism-debt status, public-resource acquisition attempts, constructed hypothesis-test dataset evidence, mechanism-axis or mediation results, which axes failed, what was tested behaviorally, which recovery attempts were made, and the next executable continuation. Do not present unresolved blockers as terminal.
31. When a meaningful bounded insight seems to have been reached, call `synthesize_spectra_generalizability_finding` with the model paper context and all final artifacts. If the synthesis finds a stronger unresolved alternative axis, route back to the Investigator with that axis as the new primary hypothesis. Otherwise, the final report should be a paper-ready SPECTRA finding with interpretation, evidence boundary, model-paper context, an evidence-to-claim table, and overclaim guardrails.

## Validation Rules

- A higher spectral parameter should generally produce lower cross-split overlap.
- Train and test sizes should remain large enough for the intended evaluation metric.
- Weighted similarity analyses should report mean, standard deviation, maximum, and minimum cross-split similarity.
- Binary similarity analyses should report the fraction of samples with at least one cross-split neighbor.
- Reports must include negative findings and failed assumptions.
- Reports must distinguish prospective axes from post-hoc explanatory axes.
- The selected axis must be scientifically defensible, not merely the most monotonic curve.
- Benchmark-mode reports must show that each curve point used a fresh model trained only on that split's training units.
- Audit fallback reports must state that fixed-prediction curves are diagnostic and do not establish the full SPECTRA benchmark claim.
- Data-only overlap screens must be labeled pre-benchmark screening and must include a required next behavioral experiment unless behavior is already tested.
- Missing dependencies or import failures are not concrete blockers until the agent has tried an allowed repository/environment repair.
- Missing local annotations or reference resources are not concrete blockers until the agent has tried source-provenance recovery, public acquisition, mapping, and validation when resources are plausibly public and network access is allowed.
- If the original benchmark cannot test a mechanism hypothesis after those recovery attempts, the agent must construct a hypothesis-test dataset from public/local resources when a defensible dataset can be built, or launch a bounded construction fallback with a concrete blocker for the full construction; it must label the resulting evidence as an extension rather than a replacement for the original benchmark.
- A supported proxy or model-space axis cannot be closed by reflection alone or by shallow local mechanism tests. It requires mechanism-level evidence with controls, or a launched continuation to public resources, constructed-dataset tests, residual axes, broader task coverage, or compute/resource work.
- A negative or inconclusive run over a limited task subset is preliminary. A completed SPECTRA claim requires all feasible tasks or a ranked task-selection rationale with blockers for omitted tasks.
- Broad searches should include surface/content, representation, metadata/context/provenance, and scientific-mechanism axes when inputs allow.
- A checkpoint report is not a completed positive report. A single supported task is a candidate axis, not a completed finding, and must trigger replication across available tasks/datasets. Mixed replication must trigger residual-axis search when executable. Supported proxy axes are not completed explanations. K-mers, sequence identity, GC/CpG, length, entropy, crude motif words, metadata labels, and generic embeddings should be classified as proxy-level evidence unless the report shows how they map to a curated annotation or mechanistic hypothesis and tests mediation or controls against simpler proxies. If the declared search scope is exhausted first, launch concrete expansion work or a bounded fallback, not a universal generalization claim or queued-only manifest.

## Similarity-Hypothesis Status

- `monotonic_supported`: novelty under this axis broadly tracks performance degradation.
- `localized_supported`: a low-support region is harder, but the curve is not globally monotonic.
- `weak_supported`: the axis has a weak signal and should not be the only evidence.
- `not_explanatory`: the axis does not explain observed failure; this is a reportable finding.
- `not_evaluable`: the curve lacks enough usable numeric points or variation.

## Leakage Policy

- Prospective axes are usable before labels are known and valid for split design.
- Post-hoc axes are useful for explaining observed failures after labels are available.
- Invalid axes have label leakage, circularity, uncontrolled confounding, or too few usable points.
- A post-hoc axis can be selected for explanation, but the report must not present it as a prospective deployment split.
- If benchmark mode is feasible, post-hoc fixed-prediction binning is not an acceptable substitute for split construction and retraining.
- Data geometry without behavior is pre-benchmark screening, even when the overlap curve is monotonic.

## Implementation Guidance

- Use exact pairwise similarity for small datasets when feasible.
- Use chunked pairwise computation when all-pairs similarity is required but memory is limited.
- Use approximate nearest neighbors when the spectral property can be embedded.
- Use hashing, sketches, or domain-specific indexes when pairwise comparison is too expensive.
- Record every approximation because it changes what the split means scientifically.
- When a candidate axis fails, use the failure to choose the next similarity definition or computation strategy.
- Do not choose tasks because they feel representative. Use an all-task screen to rank tasks by overlap/suspiciousness, source family, scientific diversity, feasibility, sample size, and prior/internal signals. Run all feasible tasks when possible.
- If compute requires a task subset, include high-suspicion tasks and distinct scientific families, record omitted-task blockers, and label the result preliminary until remaining feasible tasks are run.
- When a replicated axis has mixed support, use the supported-versus-failed task contrast to choose the next similarity definition, not merely the next arbitrary registry entry.
- When a supported axis is model-space similarity, do not stop at embedding distance. Ask what biological or scientific variables define high- and low-support embedding regions, then test those variables as interpretable axes when feasible.
- When a supported axis is a proxy, do not stop at the proxy. Ask what mechanism the proxy might stand in for, then test curated annotations, mechanistic features, context interactions, or mediation/residual curves when current inputs allow.
- For sequence tasks, missing coordinates do not block all mechanism work. Run sequence-derived tests such as motif/PWM family support, motif grammar, CpG/low-complexity/repeat-like features, GC/length/k-mer matched residual curves, class-conditional curves, and model-space residuals after matching surface sequence support.
- Local benchmark-mode proxy-audit feasibility does not waive mechanism debt. If direct mechanism tests cannot be run on original rows, construct a public/local hypothesis-test dataset when feasible.
- If those resources are absent locally, try to acquire public resources before stopping. For genomics this can include reference genomes, JASPAR/HOCOMOCO motifs, ENCODE cCRE/chromatin/TF tracks, conservation, repeats, CpG islands, GENCODE/RefSeq, dbSNP, GTEx, or other mappable public resources. For molecules this can include ChEMBL, PubChem, BindingDB, UniProt, PDB, and assay metadata. For perturbation biology this can include GO, Reactome, LINCS, CELLxGENE, OpenTargets, and cell/perturbation metadata. For clinical or imaging tasks this can include terminology systems, cohort documentation, DICOM/scanner metadata, device/protocol metadata, and site/time context.
- If public resources cannot be mapped from local rows, inspect the original dataset source before expensive matching: benchmark repository, dataset loader code, download scripts, HuggingFace/TDC/OpenML pages, source manifests, raw FASTA/BED/CSV/assay/cohort archives, paper supplements, and metadata files.
- If the current benchmark still cannot answer the mechanism question, construct a defensible hypothesis-test dataset from public/local resources. The dataset must have explicit scientific units, non-circular labels, resource provenance, mapping validation, prospective split features, leakage controls, matched or stratified confounder controls, and fresh per-split training or probing when benchmark mode is feasible.
- Keep the training recipe fixed across spectral parameters unless the purpose is explicitly to compare training recipes.
- Do not let test units, test labels, fixed-model predictions, or learned test-set error enter prospective split construction.
- For pretrained foundation models, retrain or refit downstream probes, heads, adapters, or defensible baselines per split; do not require pretraining the foundation model from scratch.
- For target-model benchmarks, try to run the target model first. A surrogate baseline is a control, not a replacement, unless target-model recovery attempts fail and are documented.
- Prefer a targeted feasible behavioral experiment after a broad screen: one or a few suspicious tasks, one defensible model/probe, and at least three controlled overlap levels.
- If a candidate axis is non-explanatory, do not stop. Move to a different axis class unless the declared search scope is exhausted; if exhausted, escalate to additional data, features, tasks, model access, or compute.
- If a candidate axis is supported, do not stop at a next-experiment proposal while replication is executable. Run the replication and report the cross-task/dataset support pattern. After replication, do not terminate the loop; classify explanatory depth and continue to mechanism-level evidence, residual axes, public resources, constructed datasets, broader task coverage, or a launched continuation job/fallback.
- If a candidate axis is replicated but mixed, do not stop at the evidence table. Reflect on why the axis works in some datasets and fails in others, then test residual or composite axes when current inputs allow.
- If a model-space residual axis remains supported, do not present it as the biological explanation. Translate it into motif, context, provenance, composition, class-conditional, or other domain hypotheses and test at least one.
- If a translated domain axis is still shallow, such as GC/CpG, low complexity, broad regulatory words, or task family, do not present it as the mechanism. Use it to choose a deeper hypothesis, such as curated motif families, motif grammar, conservation, pathway/ontology, structural functional regions, biological context, or controlled mediation.
- If the deeper hypothesis needs a public resource, do not simply list it as missing. Recover source provenance if identifiers are absent, search, download or query a bounded version, record provenance, map it, validate it on a sample, and then continue the analysis when feasible.
- If a public/local hypothesis-test dataset is constructed, keep its claims scoped: it tests whether the proposed mechanism produces target-model degradation under controlled conditions, not whether unmapped original benchmark rows necessarily had the same mechanism.
- A final SPECTRA report is valid only if it contextualizes the result against the model paper and states the strongest claim the evidence supports. It must explain whether the finding is a supported degradation axis, an attenuation result, a negative finding, or a blocker, and it must state what not to claim. It is not valid to stop on a small attenuation/negative result while a larger unresolved degradation axis remains in the evidence.

## Continuation Conditions

Do not terminate the audit for the conditions below. Convert each one into a
checkpoint plus the next executable continuation:

- The unit of analysis is ambiguous: inspect data dictionaries, loaders, schemas, papers, and examples; if still ambiguous, create a schema-recovery task.
- The prediction target is unknown: inspect labels, metrics, repository configs, and benchmark docs; if still unknown, create a target-recovery task.
- The proposed spectral property is unrelated to expected model failure: abandon that property and choose another axis.
- Raw labels and a trainable model are available but the analysis only bins fixed predictions.
- A suspicious data-level axis is found but no behavioral test or concrete blocker is reported.
- There are fewer than three usable split difficulty levels.
- Test sets become too small to support the metric.
- Every candidate axis in the declared search scope is non-explanatory or not evaluable and the agent has reported concrete additional data, features, tasks, model access, or compute needed to continue.
- Tested tasks are negative or mixed but additional feasible tasks remain untested. Expand task coverage by launching remaining tasks or a bounded fallback slice.
- A replicated candidate axis is mixed and all residual/composite axes needed to explain the heterogeneity require unavailable data, features, model access, or compute.
- A supported model-space axis cannot be translated because all biological/domain features needed to interpret high- and low-support embedding regions are unavailable.
- A supported proxy axis cannot be pushed toward mechanism because all annotations, mechanistic features, metadata, controls, model activations, or compute needed for the next experiment are unavailable.
- A supported proxy axis has no executed mechanism/mediation test. Run local input-derived tests, source provenance recovery, public resources, constructed datasets, residual axes, or launched continuation work.
- Public resources needed for the mechanism test are not downloadable, not licensed for use, too large for the current budget, lack mappable identifiers after documented upstream source-provenance recovery, or fail mapping validation after documented acquisition attempts. Select an alternate public resource, construct a hypothesis-test dataset, reduce the validation slice, or launch a bounded acquisition/mapping fallback.
- A defensible hypothesis-test dataset cannot be constructed from the first public/local resource choice because required labels, features, provenance, license, or compute are unavailable after documented search. Try an alternate resource/data design or launch a bounded construction fallback.
- The target model environment, data, or hardware is still unavailable after documented repository inspection and recovery attempts. Switch to another executable target-model task, smaller slice, adapter/probe setting, or launched compute job/fallback; a surrogate baseline remains a control only.
