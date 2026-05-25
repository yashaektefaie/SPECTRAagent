# SPECTRA Similarity Definition Registry

The `/spectra` MCP server now includes a literature-backed similarity definition
registry. The registry is the agent-facing knowledge layer between a broad
instruction such as "audit generalization" and the deterministic SPECTRA audit
engine.

## Framing

SPECTRA core remains domain-agnostic. It consumes either:

- a train-eval `pairwise_similarity.csv`, or
- an evaluation table with a precomputed spectral axis.

The registry helps an agent choose or construct that input. Each entry names a
scientific unit, a data type, a candidate similarity definition, required input
columns, expected SPECTRA output format, limitations, quality gates, citations,
and an executable Python example.

## Seed-Mining Sources

The initial registry was mined from Paperclip searches and full-text grep over
papers about scientific generalizability, OOD prediction, and benchmark splits.
The seed set now includes:

- BOOM and related molecular OOD papers.
- UMAP-based clustering split for virtual screening.
- DART-Eval for regulatory DNA models.
- NABench for nucleotide fitness prediction.
- ProteinGym/FLIP-style protein fitness and mutational-regime papers.
- Drug-target affinity papers on ligand, protein sequence, and GO similarity bias.
- PerturBench for cellular perturbation response.
- GeSS for scientific distribution shifts in geometric deep learning.
- RNA3DB for structurally dissimilar RNA structure splits.
- Materials OOD papers on composition and structure-space extrapolation.
- RainShift and TorchSpatial for geographic generalization and location encoders.
- Medical-imaging scanner-domain shift studies.
- Bio-ontology semantic similarity papers.
- Clinical prediction validation papers covering temporal, geographical, and domain generalizability.
- Graph OOD, time-series OOD, homology-aware sequence splitting, protein
  structure alignment, distribution shift detection, experimental batch-effect,
  EHR concept-relatedness, and image/radiomics harmonization papers.
- Microbiome beta-diversity, phylogenetic distance, mass-spectrometry spectral
  similarity, biomedical text semantic distance, point-cloud distances,
  hyperspectral spectral-angle methods, patient similarity, graph kernels, SOAP
  atomistic kernels, and spatial prediction-distance diagnostics.
- Multi-omics similarity-network fusion, interactome/network-medicine
  proximity, connectome comparison, persistence-diagram distances, genotype
  IBD/IBS relatedness, expression-signature connectivity, Cell Painting
  morphology profiles, protein binding-pocket similarity, climate niche overlap,
  and survival metric-learning papers.

Entries are marked `seed_mined` until a domain expert promotes them to
`human_reviewed`.

## CLI

List entries:

```bash
spectra similarity-definitions list
```

Suggest entries for a dataset:

```bash
spectra similarity-definitions suggest \
  --data-type molecule \
  --dataset-description "SMILES molecular property prediction dataset" \
  --task-description "audit generalization to chemically novel compounds" \
  --required-input sample_id \
  --required-input train_id \
  --required-input smiles
```

Get a definition:

```bash
spectra similarity-definitions get molecules_morgan_tanimoto
```

Get an executable script:

```bash
spectra similarity-definitions script molecules_morgan_tanimoto
```

Validate the bundled registry:

```bash
spectra similarity-definitions validate
```

## MCP Tools

The SPECTRA MCP server exposes:

- `list_similarity_definitions`
- `get_similarity_definition`
- `suggest_similarity_definitions`
- `get_similarity_example_script`
- `render_similarity_definition`
- `validate_similarity_registry`

Expected agent flow:

1. Read the dataset schema and task description.
2. Call `suggest_similarity_definitions`.
3. Select one or more candidate definitions.
4. Verify required inputs and limitations.
5. Generate `pairwise_similarity.csv` or an axis file using the example script as a starting point.
6. Call `run_spectra_audit`.
7. Report the chosen axis, citation provenance, quality gates, and limitations.

## Current Entries

The bundled registry currently has 54 seed-mined entries:

- Molecules and DTA: `molecules_morgan_tanimoto`, `molecules_umap_cluster`, `molecules_butina_fingerprint_cluster`, `molecules_bemis_murcko_scaffold`, `drug_target_ligand_target_composite`, `drug_target_protein_sequence_go_integrated`
- Sequences, structures, and phylogeny: `biological_sequence_alignment_homology`, `phylogenetic_patristic_distance_similarity`, `regulatory_dna_chromosome_holdout`, `regulatory_dna_embedding_cosine_variant_effect`, `regulatory_dna_motif_bag_similarity`, `nucleotide_contiguous_mutation_region`, `protein_mutational_regime_hamming`, `protein_positional_mutation_coverage`, `protein_structure_tm_score_similarity`, `protein_binding_pocket_similarity`, `rna_structural_component_similarity`
- Perturbation, ontology, microbiome, and clinical records: `single_cell_covariate_transfer`, `single_cell_logfc_cosine`, `cell_morphology_profile_similarity`, `gene_expression_signature_connectivity_similarity`, `ontology_annotation_jaccard_similarity`, `microbiome_beta_diversity_similarity`, `ehr_concept_relatedness_similarity`, `clinical_tabular_patient_similarity`, `longitudinal_patient_trajectory_similarity`, `survival_metric_learned_patient_similarity`
- Materials, graphs, geometry, time series, and AI4Science: `materials_composition_formula_similarity`, `materials_structure_ofm_embedding_similarity`, `atomistic_soap_kernel_similarity`, `gdl_shift_factor_distance`, `graph_neighborhood_jaccard_similarity`, `graph_wasserstein_wl_kernel_similarity`, `point_cloud_point_set_distance_similarity`, `topological_persistence_diagram_similarity`, `time_series_dtw_shape_similarity`, `domain_distribution_mmd_similarity`, `environment_confounder_domain_similarity`
- Geography, clinical, imaging, spectra, and experimental metadata: `geospatial_haversine_region_distance`, `geospatial_location_embedding_similarity`, `spatial_prediction_horizon_similarity`, `medical_imaging_scanner_protocol_domain`, `imaging_histogram_radiomics_similarity`, `image_perceptual_embedding_similarity`, `hyperspectral_spectral_angle_similarity`, `mass_spectrometry_spectral_cosine_similarity`, `clinical_temporal_site_domain`, `experimental_batch_protocol_similarity`
- Multi-omics, networks, population genetics, and ecology: `multi_omics_similarity_network_fusion`, `interactome_network_proximity_similarity`, `connectome_matrix_correlation_similarity`, `genotype_ibd_ibs_similarity`, `climate_niche_environmental_similarity`
- Text and document data: `biomedical_text_semantic_similarity`

The registry also includes generic example generators for sequence identity,
embedding cosine, metadata/group similarity, set overlap, geodesic distance,
formula composition, molecule fingerprints, alignment identity, time-series
DTW, graph-neighborhood overlap, domain-level MMD similarity, histogram
similarity, composition/Bray-Curtis similarity, point-cloud Chamfer similarity,
hyperspectral spectral-angle similarity, MS/MS peak-list cosine similarity,
regularized Mahalanobis similarity, precomputed metric conversion, and
composite similarity. The latest pass also adds vector Pearson/correlation
similarity, token-level event-sequence alignment, network set-proximity
similarity, and genotype IBS dosage similarity.

## Release Claim

Do not claim that the registry is exhaustive over all possible scientific
similarity definitions. A defensible claim is:

> We performed a systematic seed-mining pass across major scientific data
> modalities and released an extensible, evidence-backed similarity-definition
> registry. The registry gives agents reusable starting points for constructing
> SPECTRA pairwise-similarity files, while leaving domain experts free to add or
> promote definitions.

The important product claim is not that no one can define another similarity.
It is that `/spectra` gives agents a broad, literature-backed starting library
and a standard contract for adding new definitions without changing the SPECTRA
audit engine.

## Important Constraint

The registry should not make SPECTRA molecule-specific. Molecule Tanimoto is one
entry. The core abstraction is:

> The agent defines or retrieves a scientifically justified similarity notion,
> computes train-eval similarity, and SPECTRA computes the spectral performance
> curve.
