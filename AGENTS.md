# /spectra Agent Instructions

When using this repository as an agent, treat `/spectra` as an executable
SPECTRA workflow, not merely as a request for an explanation.

For human-friendly requests, prefer the CLI:

```bash
spectra ask "<question>" --dataset /path/to/data --out /path/to/run
```

The CLI automatically routes the request:

- Split-only requests such as "construct SPECTRA splits" or "generate SPECTRA
  splits" use focused SPECTRA split construction and must not launch the broad
  Investigator/Distiller audit loop.
- Model, paper, checkpoint, foundation-model, or broad generalizability
  questions use the autonomous audit loop.
- Use `--ask-mode splits` or `--ask-mode audit` only when an explicit override
  is needed.

Focused split construction must first choose and record a similarity definition,
then choose and record a pairwise computation strategy, compute pairwise
similarities or a property graph, generate split candidates, verify decreasing
train-test similarity across spectral parameters, and validate with a fixed
simple baseline when labels are available.

If the user gives `/spectra` a question plus a model paper/reference, begin by
calling `start_spectra_audit_session` through the MCP server. Use its returned
role graph and routing policy as the session contract. In clients with subagent
support, spawn the Investigator, Distiller, Dataset Scout, and Dataset
Constructor roles when the routing policy calls for them. In clients without
subagents, execute those roles sequentially and persist the same artifacts.
If the user wants a one-command autonomous run, use
`prepare_spectra_audit_session` to create the session without launching roles,
or `run_spectra_audit_session` with `execute_roles=true` plus an explicit host
agent command template. Do not execute roles implicitly just because the session
was prepared.
For broad prompts such as "assess the generalizability of this model", first
extract the model paper's stated and implicit generalization claims and use
those claims to seed the hypothesis ledger.
If the session `audit_scope` is `beyond_paper_discovery`, do not restrict the
audit to paper benchmarks. Treat the paper as model context, then search for or
construct public/local datasets that test important external generalization
axes.

Use a cheap-first behavioral runtime policy. Before expensive fitting,
fine-tuning, all-pairs graph construction, large ANN indexing, or full-dataset
embedding, run the smallest leakage-aware behavioral slice that can answer the
next live hypothesis. Start with schema/leakage checks, simple controls,
cached/chunked representations, and deterministic non-iterative probes such as
nearest-centroid/prototype scores, mean-difference linear scores, kNN/prototype
retrieval, or closed-form small linear baselines. Escalate to logistic/SVM/ridge
heads, fine-tuning, or full-scale jobs only when the cheap probe is
inconclusive or the stronger deployment claim requires it. Before launching a
heavy step, write the time/resource budget, success criterion, timeout/fallback
condition, and cheaper fallback. After one timeout or runtime failure from a
solver family, switch to the fallback or a smaller slice instead of retrying the
same slow method repeatedly.

Before open-ended dataset search, query the portable dataset catalog with
`suggest_dataset_catalog_entries` or the CLI `dataset-catalog search`. Treat
catalog matches as candidates, not permission to download everything. Read each
entry's access, authentication, expected fields, leakage risks, and `scale`
guidance. For multi-GB, credentialed, or very large resources, first plan a
bounded subset, manifest-only pass, precomputed-embedding route, or
user-authorized data path.

Before implementing an audit:

1. Load the benchmark capsule for the target paper.
2. Identify the scientific unit of generalization.
3. Propose spectral properties and explain why each property should affect model failure.
4. Treat each property as a similarity hypothesis, not as the final answer.
5. Classify each hypothesis as prospective, post-hoc explanatory, or invalid.
6. Plan exact, chunked, approximate, or indexed graph construction.
7. Run the audit for each candidate axis until a defensible behavioral axis is found, then classify its explanatory depth.
8. Score each curve as monotonic, localized, weak, non-explanatory, or not evaluable.
9. Treat supported surface, model-space, or broad domain-proxy axes as mechanism debt, not closure.
10. Use failed curves and supported proxy curves to choose the next similarity definition, control, public resource, mediation test, or constructed dataset.
11. Validate split statistics before trusting model metrics.
12. Fill an audit card and render the report only after the claim boundary is defensible.

Do not claim a SPECTRA split is valid unless measured cross-split overlap
decreases across the spectral parameters.

Do not claim SPECTRA failed just because the first similarity axis is
non-monotonic. A non-explanatory axis is a finding. Report it, then try the next
scientifically plausible axis. Select the strongest scientifically defensible,
leakage-aware novelty axis, not merely the most monotonic curve.

For investigation or paper-facing runs, do not stop at "the model performs worse
when train-test similarity is low" unless the axis is already a mechanism-level
or deployment-level explanation with controls. A supported proxy curve should
raise the next question: what real scientific, experimental, annotation,
population, environment, or data-construction mechanism makes that proxy matter?
If the current benchmark cannot answer that question, search for public/local
resources, recover upstream provenance, or construct a hypothesis-test dataset.
Only a user-requested screening run may end with a surface/proxy curve as the
primary output.
