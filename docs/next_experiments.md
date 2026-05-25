# Next Experiments

Date: 2026-05-07

## Current Evidence

The current `/spectra` evidence artifacts are now in place:

1. BOOM with-vs-without audit-quality pilot:
   - Vanilla: `8 / 14`
   - `/spectra`: `11 / 14`
   - Summary: `experiments/boom_pilot/scores.json`

2. BOOM numeric mini-audit:
   - ID RMSE: `0.04425166800519178`
   - Full OOD RMSE: `0.22963059182905268`
   - Lowest-overlap OOD RMSE: `0.25340949263725665`
   - Summary: `experiments/boom_numeric_mini_audit/summary.json`

3. Hard-blind model-generalizability pilot:
   - Vanilla: `16 / 16`
   - `/spectra`: `16 / 16`
   - Interpretation: both agents found the core failure; `/spectra` improved
     audit depth, not discovery.
   - Summary: `experiments/hard_blind_model_eval/scores.json`

4. Prompt-specificity ablation:
   - Broad generalizability: `19 / 22`
   - Explicit distance-from-train: `19 / 22`
   - `/spectra`: `22 / 22`
   - Interpretation: `/spectra` beat the direct distance prompt on audit
     completeness, but the direct distance prompt already recovered the main
     spectral curve.
   - Summary: `experiments/spectra_value_ablation/scores.json`

## Current Framing

Use the framing in `docs/spectra_framing_and_evidence_plan.md`:

> SPECTRA is the deterministic audit engine. `/spectra` is the agent-native
> interface to that engine.

The next work should demonstrate package utility and agent-interface utility
separately. The key standard is no longer whether an agent can be prompted to
plot performance versus train distance; it is whether SPECTRA provides
reproducible audit artifacts and whether `/spectra` helps agents configure and
run that audit reliably in real projects.

## Immediate Next Computational Step

Use and harden the package-backed CLI:

```bash
spectra audit --domain molecules ...
```

The initial generic CLI is implemented in `spectrae.cli` and documented in
`docs/spectra_cli.md`. The core path consumes either a measured spectral axis or
a pairwise train-eval similarity graph. The molecule Morgan/Tanimoto path is an
adapter/example, not the framework definition.

The first hard-blind molecule run is complete:

- Summary: `docs/spectra_cli_hard_blind_run.md`
- Artifact directory:
  `/ewsc/yektefai/spectra_assets/spectra_cli_hard_blind_molecular_audit`

Minimum hardening targets:

1. Require future `/spectra` agents to define a similarity notion and compute a
   pairwise similarity graph or measured spectral axis.
2. Require the agent to call the generic CLI/MCP tool on that graph.
3. Compare emitted artifacts against earlier ad hoc agent analyses.
4. Add domain adapters only as convenience layers around the generic graph/axis
   contract.
5. Add more formal schema validation for emitted audit cards.

Success criterion:

> Re-running the same command on the same inputs produces the same artifacts and
> same conclusions. `/spectra` agents must call the package and cite generated
> files, not merely produce an ad hoc narrative.

## Next Computational Extension

After the CLI works, upgrade the BOOM mini-audit from a post-hoc OOD
test-subset curve to real overlap-controlled split generation.

Minimum viable version:

1. Use BOOM 10k density molecules.
2. Build a sparse Morgan Tanimoto graph.
3. Generate train/test splits with decreasing measured train-test overlap.
4. Retrain the same random forest on each split.
5. Write `split_stats.csv`, `performance_by_overlap.csv`, `spectral_curve.svg`,
   `audit_card.json`, and `report.md`.
6. Compare random split, BOOM property-tail split, and generated
   overlap-controlled splits.

Success criterion:

> Measured cross-split overlap decreases across split levels, and model
> performance is reported against measured overlap rather than nominal split
> names.

## Immediate Next Agent-Benchmark Step

Run the prompt-specificity ablation in
`experiments/spectra_value_ablation/`:

1. Broad prompt: "Evaluate whether the model generalizes."
2. Explicit distance prompt: "Evaluate performance as a function of distance
   from train."
3. `/spectra`: execute the full protocol with novelty axes, property graphs,
   overlap validation, performance curves, and reusable audit artifacts.

Success criterion:

> `/spectra` must beat the explicit distance-from-train prompt on overlap
> validation, property-graph specificity, multi-axis audit quality, invalid-axis
> handling, and artifact completeness. Beating only the broad prompt is not
> enough.

Then repeat the three-condition benchmark for three more launch capsules:

- DART-Eval
- NABench
- Systema

Success criterion:

> `/spectra` improves average audit score across capsules beyond the explicit
> distance-from-train control, especially on validation and reusable audit
> artifacts.

## Claim Boundary

Current evidence supports:

> `/spectra` improves the quality and structure of agent-produced
> generalization audits, and its audit protocol can expose additional
> performance degradation along measured novelty axes.

Current evidence does not yet support:

> `/spectra` improves the trained model itself.

The next BOOM split-generation run is the first step toward stronger
computational evidence.
