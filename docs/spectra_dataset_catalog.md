# SPECTRA Dataset Catalog

Date: 2026-05-21

## Purpose

The dataset catalog is the portable data-resource layer for `/spectra`.
Benchmark capsules describe papers; the dataset catalog describes data resources
that a fresh SPECTRA agent can find, access, validate, and use for an audit
without knowing any local run paths from this machine.

Entries intentionally avoid absolute local paths. They contain source URLs,
access requirements, expected fields, recommended SPECTRA axes, leakage risks,
related benchmark capsules, related memory entries, and scale/default-use
guidance.

## Current Entries

| Entry | Domain | Main Use |
| --- | --- | --- |
| `boom_benchmark_datasets` | molecular ML | Chemical novelty/OOD property curves. |
| `dart_eval_synapse` | regulatory DNA | DNA foundation-model regulatory audits; requires Synapse access. |
| `nabench_assays` | sequence fitness | Mutational-position and local-context generalization. |
| `perturbench_huggingface` | single-cell perturbation | Perturbation/cell-state/pathway novelty audits. |
| `systema_zenodo` | single-cell perturbation | Systematic variation and perturbation-specific evaluation. |
| `umap_virtual_screening_nci60` | virtual screening | Chemical cluster and NCI-60 activity generalization. |
| `gess_qmof` | materials / GDL | Scientific distribution shifts beyond biology. |
| `open_problems_pbmc_dge` | drug perturbation | Hard agent-discovery capsule candidate. |
| `encode_ccre_screen` | regulatory DNA | Coordinate-backed cCRE regulatory annotations. |
| `dc_tap_final_table` | regulatory perturbation | Distal element-gene perturbational deployment tests. |
| `crispr_comparison_epcrispr` | regulatory perturbation | CRISPR companion/source-control tests. |
| `bbbc021_cell_painting` | bioimage / Cell Painting | Compound, morphology, plate, and dose generalization. |
| `coos7_zenodo` | microscopy | Out-of-sample cell-image generalization. |
| `drugood_chembl` | molecular ML / drug discovery | Assay-domain, target-family, and chemical novelty audits. |
| `good_graph_ood` | graph ML | Graph OOD curves over domain, structure, and neighborhood novelty. |
| `matbench_materials` | materials science | Composition, structure, and property-support generalization. |
| `mimic_eicu_physionet` | clinical EHR | Credentialed clinical site/time/phenotype generalization planning. |
| `proteingym_aws` | protein fitness | Protein mutation, assay, family, and structural-site novelty. |
| `tableshift_benchmark` | tabular ML | Public/credentialed tabular distribution-shift operating points. |
| `ucr_uea_time_series` | time series | DTW, frequency, subject, device, and acquisition novelty. |
| `wilds_camelyon17` | medical imaging | Hospital-shift histopathology and stain/style support. |
| `wilds_povertymap` | geospatial ML | Country, geography, environmental imagery, and survey-support shift. |
| `wilds_rxrx1` | cellular microscopy | Batch, plate, morphology, and acquisition-style shift. |
| `flip_protein_fitness` | protein fitness | FLIP GB1/AAV/thermostability mutation and extrapolation splits. |
| `flip2_protein_fitness` | protein fitness | FLIP2 number/position/mutation/fitness/wild-type generalization. |
| `eurocropsml_zenodo` | geospatial time series | Transnational few-shot crop-type and parcel time-series transfer. |
| `wild_time_benchmark` | temporal shift | Temporal distribution-shift operating points across modalities. |
| `woods_time_series` | time series | Sequential/domain OOD generalization benchmarks. |
| `rxrx2_cell_painting` | bioimage / Cell Painting | Large compound, replicate, dose, and batch generalization resource. |
| `welqrate_drug_discovery` | molecular ML / virtual screening | Curated drug-discovery screening benchmark and split schemes. |
| `admeood_drug_property` | molecular ML / ADMET | Drug-property OOD candidate with noise/conflict shift; source verification required. |
| `causalbench_single_cell_grn` | single-cell perturbation / GRN | Perturb-seq network-inference generalization over targets, cell lines, and priors. |
| `scperturb_harmonized` | single-cell perturbation | Harmonized public perturbation resource for cross-dataset/cell/target audits. |
| `cross_species_tf_binding_morale` | regulatory genomics | Cross-species TF binding audits over species, motif, repeat, and sequence novelty. |
| `chexphoto_cxr_robustness` | medical imaging | Chest X-ray photo/acquisition robustness; Redivis or dataset terms likely required. |
| `geobench_earth_monitoring` | earth observation | Multi-task geospatial foundation-model audits over geography, sensor, task, and season. |
| `landcovernet_radiant` | earth observation / land cover | Global land-cover geographic, sensor, biome, and class-support audits. |
| `pdebench_sciml` | scientific ML / physics | PDE surrogate and neural-operator audits over parameters, regimes, and IC support. |
| `open_catalyst_oc20` | materials / catalysis | Adsorbate-surface OOD curves; full dataset is very large, use metadata/subsets first. |

## Scale Policy

Dataset catalog entries are not automatic download instructions. Each entry may
include a `scale` block:

- `download_size`: a human-readable estimate or warning.
- `spectra_default`: the recommended first action for an autonomous SPECTRA
  run, such as `full_selected_task`, `selected_curated_subset`,
  `sampled_images_or_precomputed_embeddings`, or
  `metadata_plan_or_user_authorized_subset`.

The autonomous runner should use this field before downloading data. If an entry
is credentialed, multi-GB, or potentially much larger, Dataset Scout should first
build a manifest, choose a task-specific subset, or request a user-authorized
path rather than pulling the full resource blindly.

## CLI

```sh
python -m spectrae.cli dataset-catalog list
python -m spectrae.cli dataset-catalog search --query "Caduceus regulatory DNA perturbational generalization"
python -m spectrae.cli dataset-catalog suggest --audit-question "find external regulatory data to test Caduceus generalization" --domain regulatory_dna
python -m spectrae.cli dataset-catalog render dc_tap_final_table
python -m spectrae.cli dataset-catalog validate
```

## Agent Policy

Dataset Scout should query this catalog before open-ended web search. A matching
entry is not a complete dataset package. The agent still has to download or
recover the data, pin source versions or hashes for publication, validate schema
and identifiers, audit leakage, and construct SPECTRA-ready train/test splits.

Memory entries should link to catalog IDs rather than local directories. This
keeps prior-run lessons portable while still preserving the historical result.

Entries populated from Paperclip include a `paperclip_provenance` block with
search IDs, map IDs, and indexed paper IDs. Use that as a literature-mining
trace, not as a replacement for official source/license verification.

See also:

- `docs/dataset_catalog_mining_rounds_20260521.md`
- `docs/paperclip_dataset_scan_20260521.md`
