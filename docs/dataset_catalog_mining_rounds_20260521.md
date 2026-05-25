# Dataset Catalog Mining Rounds

Date: 2026-05-21

## Context

Paperclip MCP access was not available in this execution environment during the
first pass, so the initial catalog used a Paperclip-style procedure over
primary-source benchmark pages, repository metadata, prior SPECTRA benchmark
capsules, and the existing SPECTRA memory. The Paperclip CLI was later verified
and used for real searches and maps; see
`docs/paperclip_dataset_scan_20260521.md`.

## Acceptance Criteria

A dataset was added when it had:

- a portable public source, repository, data portal, or credentialed-access page;
- a plausible scientific unit of generalization;
- at least one meaningful SPECTRA similarity axis;
- enough expected fields for an agent to plan data loading and leakage checks;
- realistic access and scale guidance.

Large resources were not excluded, but they were marked so agents do not pull
them by default.

## Round 1: Existing SPECTRA Anchor Benchmarks

These entries were already present and remain the launch core:

| Catalog ID | Modality | Main SPECTRA Use |
| --- | --- | --- |
| `boom_benchmark_datasets` | molecules | Chemical novelty and molecular OOD property curves. |
| `dart_eval_synapse` | regulatory DNA | DNA foundation-model regulatory-context audits. |
| `nabench_assays` | nucleotide fitness | Mutational-position and local-context extrapolation. |
| `perturbench_huggingface` | single-cell perturbation | Perturbation, pathway, and cell-state novelty. |
| `systema_zenodo` | single-cell perturbation | Systematic variation and confounding audits. |
| `umap_virtual_screening_nci60` | virtual screening | Cluster, scaffold, and NCI-60 activity generalization. |
| `gess_qmof` | materials / geometric ML | Scientific distribution shifts beyond biology. |
| `open_problems_pbmc_dge` | drug perturbation | Agent-discovery stress test for perturbational data. |
| `encode_ccre_screen` | regulatory DNA | Coordinate-backed cCRE annotation and regulatory context. |
| `dc_tap_final_table` | regulatory perturbation | Distal element-gene perturbational deployment tests. |
| `crispr_comparison_epcrispr` | regulatory perturbation | CRISPR companion/source-control tests. |

## Round 2: Bioimage and Medical Imaging Datasets

Added entries:

| Catalog ID | Feasibility | Reason |
| --- | --- | --- |
| `bbbc021_cell_painting` | use profiles or sampled images first | Good compound, concentration, morphology, and plate axes. |
| `coos7_zenodo` | feasible but not instant | Direct out-of-sample microscopy benchmark. |
| `wilds_camelyon17` | sampled images or embeddings | Hospital-shift histopathology with patient/slide leakage risks. |
| `wilds_rxrx1` | sampled images or embeddings | Batch-shift cellular microscopy with plate/acquisition artifacts. |

Deferred full pulls:

- Full high-content microscopy image collections should not be automatic
  default downloads. Agents should start from profiles, embeddings, or
  stratified image subsets.
- Large WILDS image datasets should be treated as reusable benchmark resources,
  not scratch downloads for every exploratory run.

## Round 3: Molecules, Proteins, Materials, and Graphs

Added entries:

| Catalog ID | Feasibility | Reason |
| --- | --- | --- |
| `drugood_chembl` | selected curated subset | Assay-domain and chemical novelty shifts. |
| `proteingym_aws` | single assay or family subset | Protein variant, position, family, and structural-site novelty. |
| `matbench_materials` | full selected task | Lightweight materials property tasks. |
| `good_graph_ood` | selected official dataset | Graph OOD domains and structural similarity axes. |

Deferred full pulls:

- Reconstructing all ChEMBL-derived data from scratch is unnecessary for first
  SPECTRA audits; start with realized DrugOOD subsets.
- ProteinGym is broad; agents should begin with one DMS assay, protein family,
  or clinical-variant slice and expand only when the question requires it.
- Massive materials corpora such as full simulation repositories are useful for
  future catalog entries, but Matbench is the better first portable target.

## Round 4: Clinical, Tabular, Time-Series, and Geospatial Datasets

Added entries:

| Catalog ID | Feasibility | Reason |
| --- | --- | --- |
| `tableshift_benchmark` | public tasks first | Tabular distribution-shift operating points across domains. |
| `mimic_eicu_physionet` | credentialed planning only | Clinical site, time, phenotype, and care-pathway audits. |
| `ucr_uea_time_series` | full selected dataset | Lightweight time-series similarity and metadata-support audits. |
| `wilds_povertymap` | fold subset or embeddings | Geospatial country, geography, and environmental-image shift. |

Deferred full pulls:

- Credentialed clinical EHR data must not be pulled, copied, or sent to external
  services without explicit authorized access and data-use compliance.
- Time-series archive datasets without subject/device metadata are useful for
  shape-support demos, but not strong mechanism claims.
- Geospatial image datasets should be explored with folds, metadata, and
  embeddings before full imagery downloads.

## Current Outcome

The portable dataset catalog started with 23 entries across molecules,
regulatory DNA, nucleotide fitness, protein fitness, single-cell perturbation,
bioimage, medical imaging, materials science, graph ML, tabular ML, clinical
EHR, time-series, and geospatial prediction. Paperclip-backed passes expanded
this to 39 entries by adding FLIP, FLIP2, EuroCropsML, Wild-Time, WOODS, RxRx2,
WelQrate, ADMEOOD, CausalBench, scPerturb, MORALE cross-species TF binding,
CheXphoto, GEO-Bench, LandCoverNet, PDEBench, and OC20.

The catalog is designed for Dataset Scout and the autonomous `/spectra` runner:

1. search the catalog by audit question and modality;
2. inspect access, scale, and expected fields;
3. choose a feasible subset or authorized data source;
4. construct a manifest and leakage plan;
5. hand the dataset plan to Investigator for SPECTRA axis testing.
