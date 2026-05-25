# SPECTRA Targeted Operating-Point Registry

SPECTRA has two evaluation modes:

1. **Comprehensive audit mode**: estimate performance across the full
   train-similarity or novelty curve.
2. **Targeted operating-point mode**: evaluate a named deployment condition,
   such as scaffold holdout, chromosome holdout, leave-site-out validation, or
   contiguous mutational-region split.

This registry supports the second mode. Each entry describes a literature-backed
split or validation operator, the novelty axis it targets, the required columns,
how it maps onto a SPECTRA curve, quality gates, and known failure modes.

## Agent Flow

1. Decide whether the user wants the full SPECTRA curve or one targeted
   deployment point.
2. If targeted, call `suggest_operating_point_methods` with the dataset,
   deployment question, data type, and novelty axis.
3. Build the split assignment requested by the selected method.
4. Measure the resulting train-test similarity using the similarity-definition
   and similarity-computation registries.
5. Report both the named operating point and where it lands on the SPECTRA
   similarity curve.

## CLI

List methods:

```bash
spectra operating-points list
```

Suggest a method:

```bash
spectra operating-points suggest \
  --dataset-description "regulatory DNA enhancer examples across chromosomes and cell types" \
  --deployment-question "test generalization to unseen chromosomes" \
  --data-type regulatory_dna \
  --novelty-axis chromosome
```

Get one method:

```bash
spectra operating-points get chromosome_loco_holdout
```

Validate the bundled registry:

```bash
spectra operating-points validate
```

## Current Methods

The bundled registry currently includes 28 operating-point methods:

- `bioactivity_step_forward_split`
- `biological_sequence_homology_cluster_split`
- `chromosome_loco_holdout`
- `contiguous_mutational_region_split`
- `cross_species_taxon_holdout`
- `drug_target_cold_start_split`
- `graph_covariate_concept_shift_split`
- `group_k_fold_holdout`
- `leave_assay_out_bioactivity_split`
- `leave_one_domain_out_benchmark`
- `leave_site_out_external_validation`
- `materials_leave_cluster_composition_split`
- `medical_imaging_scanner_site_holdout`
- `molecular_fingerprint_cluster_split`
- `molecular_property_extreme_split`
- `molecular_scaffold_split`
- `molecular_umap_cluster_split`
- `perturbation_systematic_variation_confounder_holdout`
- `perturbation_unseen_cell_type_condition_split`
- `perturbation_unseen_perturbation_split`
- `protein_family_remote_homology_holdout`
- `random_iid_split_baseline`
- `regulatory_cross_cell_type_assay_holdout`
- `rna_structurally_dissimilar_split`
- `spatial_block_buffered_cv`
- `spatiotemporal_forecast_horizon_split`
- `targeted_intended_use_validation`
- `temporal_forward_split`

## Framing

Existing split protocols should not be described as obsolete. The stronger
claim is:

> SPECTRA locates existing split protocols on a broader novelty spectrum. If
> the goal is discovery, use the comprehensive curve. If the goal is a known
> deployment condition, use the targeted operating point and then measure where
> it falls on the SPECTRA curve.
