# SPECTRA CLI

Date: 2026-05-11

The package-backed audit engine is exposed through:

```bash
spectra audit --eval eval_predictions.csv --axis-col measured_similarity ...
```

During local development, the same command can be run without installing the
console script:

```bash
python -m spectrae.cli audit --eval eval_predictions.csv --axis-col measured_similarity ...
```

## Core Generic Audit

The core SPECTRA audit is domain-agnostic. It consumes model predictions plus a
measured spectral axis. The axis can come from any domain-specific similarity
notion: molecular fingerprints, sequence identity, gene-network distance,
cell-type distance, hospital/site shift, embedding distance, or another
scientifically justified property.

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

For a distance axis where larger values mean farther from train, use:

```bash
spectra audit \
  --eval eval_predictions.csv \
  --out artifacts/spectra_audit \
  --target-col y_true \
  --pred-col y_pred \
  --axis-col train_distance \
  --axis-type distance
```

## Pairwise Similarity Graph Audit

The preferred framework-level workflow is:

1. An agent, human, or domain adapter defines a scientifically meaningful
   similarity function.
2. It computes a pairwise train-eval similarity graph.
3. SPECTRA consumes that graph and calculates the spectral performance curve.

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
  --pred-col y_pred \
  --scientific-unit sequence
```

The pairwise similarity CSV is long-form:

```text
sample_id,train_id,similarity
eval_00001,train_00001,0.72
eval_00001,train_00002,0.13
```

SPECTRA reduces the graph to a max-train-similarity axis for each evaluation
unit, validates monotonic novelty across thresholds, and reports performance as
a function of measured train-test overlap.

## Molecule Adapter Example

The molecule adapter is an example convenience adapter, not the core SPECTRA
abstraction. It computes a pairwise Morgan-fingerprint similarity notion and
then runs the same audit pattern.

```bash
spectra audit \
  --mode adapter \
  --domain molecules \
  --train train.csv \
  --eval eval_predictions.csv \
  --out artifacts/spectra_audit \
  --smiles-col smiles \
  --train-target-col y \
  --eval-target-col y_true \
  --pred-col y_pred
```

The molecule adapter computes aggregate regression metrics, maximum train-set
similarity for each evaluation molecule, nested performance curves over
decreasing train-test overlap, overlap validation, and AUSPC.

When RDKit is installed, the default similarity is Morgan radius-2 Tanimoto with
1,024-bit fingerprints. If RDKit is unavailable, the audit falls back to a
transparent SMILES character n-gram Jaccard similarity and records that warning
in the audit card.

## Artifacts

The CLI writes:

- `audit_card.json`
- `split_stats.csv`
- `performance_by_axis.csv` for generic/pairwise audits
- `eval_with_axis.csv` for generic/pairwise audits
- `spectral_curve.svg`
- `report.md`

Adapter-specific runs may also emit compatibility names such as
`performance_by_distance.csv` or `eval_with_distance.csv`.

The key reproducibility contract is:

> Re-running the same command on the same inputs should produce the same
> conclusions and artifact schema.

The `/spectra` agent skill should call this CLI or the equivalent MCP tool and
cite these generated files rather than writing only an ad hoc narrative.

## Example Run

The first package-backed hard-blind molecular run is documented in
`docs/spectra_cli_hard_blind_run.md`.
