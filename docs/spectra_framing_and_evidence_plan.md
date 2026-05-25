# SPECTRA Framing and Evidence Plan

Date: 2026-05-11

## Core Framing

SPECTRA should be framed as an executable standard for spectral
generalization auditing, not as a prompt that reminds agents to think about
generalizability.

The clean architecture is:

> SPECTRA is the deterministic audit engine. `/spectra` is the agent-native
> interface to that engine.

This means the core scientific contribution should live in code:

1. identify the scientific unit of generalization,
2. define train-test distance or overlap axes,
3. validate that stricter thresholds actually reduce overlap,
4. compute performance as a function of measured distance,
5. summarize the curve with reusable metrics and artifacts,
6. write an audit card/report.

Agents are not required to execute the core audit. A human or script should be
able to run the SPECTRA package directly on train/test/prediction artifacts.

## Role of `/spectra`

When a user asks an agent:

```text
/spectra analyze the generalizability of this model
```

the agent should use `/spectra` as an installed skill/package, not merely as a
style of reasoning.

The agent's job is to:

- inspect project artifacts,
- map local files into the SPECTRA input schema,
- identify the domain and scientific unit,
- choose or configure the relevant SPECTRA adapter,
- run the deterministic SPECTRA audit where possible,
- explain limitations and missing inputs,
- write the final report and suggest next experiments.

The package's job is to:

- consume agent/domain-defined similarity graphs or measured spectral axes,
- compute distances from those similarities when appropriate,
- build or summarize property graphs,
- validate overlap,
- compute performance curves,
- compute AUSPC or equivalent summaries,
- emit standardized artifacts.

This split keeps the scientific result reproducible. The agent adds convenience,
adaptation, and reporting, but the audit itself should not depend on the
agent's ad hoc reasoning.

## What We Should Not Claim

We should not claim:

> Vanilla agents cannot discover generalization failures.

Our hard-blind molecular experiments do not support that. Broad and
distance-from-train prompts found the key molecular failure.

We should also avoid claiming:

> `/spectra` is useful because it tells agents to plot performance versus
> distance from train.

That is too weak. A direct prompt can do that in domains where the distance
metric is obvious.

## Claim We Can Defend

The defensible claim is:

> SPECTRA provides a reproducible audit engine for measuring model performance
> as a function of scientifically meaningful train-test distance. `/spectra`
> makes that engine agent-native by helping agents discover inputs, configure
> domain adapters, run the audit, and produce standardized reports.

A shorter version:

> SPECTRA turns broad generalizability questions into executable, validated,
> domain-aware distance audits.

## How To Demonstrate Usefulness

The evidence should test the package and the agent interface separately.

### 1. Package Utility

Show that the deterministic package can run a complete audit from files:

```bash
spectra audit \
  --domain molecules \
  --train train.csv \
  --eval eval_predictions.csv \
  --smiles-col smiles \
  --target-col y_true \
  --pred-col y_pred \
  --out artifacts/spectra_audit
```

Expected outputs:

- `audit_card.json`
- `split_stats.csv`
- `performance_by_distance.csv`
- `spectral_curve.svg`
- `report.md`

Success criterion:

> Re-running the same command on the same inputs produces the same artifacts and
> the same conclusions.

This demonstrates that SPECTRA is not just an agent wrapper.

### 2. Agent Interface Utility

Compare three workflows:

1. human manually configures and runs the SPECTRA CLI,
2. agent without `/spectra` is asked to evaluate generalizability,
3. agent with `/spectra` is asked to analyze the model.

The `/spectra` agent should be judged on whether it correctly:

- finds the relevant local files,
- maps columns into the SPECTRA schema,
- selects the correct domain adapter,
- runs the package,
- notices missing inputs or invalid axes,
- cites the generated artifacts,
- writes a useful report.

Success criterion:

> `/spectra` reduces setup friction and produces standardized artifacts more
> reliably than a general agent prompt.

This demonstrates the value of the agent-native interface.

### 3. Direct Distance-Prompt Control

Keep the stricter ablation:

1. broad prompt,
2. explicit distance-from-train prompt,
3. `/spectra` package-backed prompt.

The key comparison is condition 2 versus condition 3.

Success criterion:

> `/spectra` must outperform the explicit distance prompt on artifact
> completeness, overlap validation, invalid-axis handling, and reproducibility,
> not merely on producing a distance curve.

### 4. Cross-Domain Evidence

Run the same evaluation across domains where the right distance metric is less
obvious:

- molecules,
- regulatory DNA,
- nucleotide fitness,
- perturbation response,
- geometric/scientific distribution shifts.

Success criterion:

> SPECTRA provides useful domain defaults and validation checks across multiple
> scientific units, while preserving the same artifact schema.

This is the strongest evidence that SPECTRA is an audit standard rather than a
random wrapper around one molecular metric.

## Next Build Step

The first package-backed CLI entry point is now implemented:

```bash
spectra audit --domain molecules ...
```

See `docs/spectra_cli.md`.

Minimum viable implementation status:

1. read evaluation prediction files,
2. consume a generic measured spectral axis or pairwise train-eval similarity graph,
3. compute aggregate metrics,
4. validate novelty/overlap over thresholds,
5. write standardized artifacts,
6. keep molecule Morgan Tanimoto as only an optional adapter/example,
7. expose the same functionality through the MCP `/spectra` skill.

Once that works, the agent benchmark should require agents to call the CLI and
cite its emitted files. That is the cleanest way to show that `/spectra` is an
interface to reproducible infrastructure, not just a prompt.
