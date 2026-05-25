# SPECTRA Similarity Computation Strategy Registry

The similarity-definition registry answers: "what should count as similarity
for this scientific unit?" This registry answers the next question: "how should
an agent compute enough train-eval similarity edges for SPECTRA without doing
unnecessary all-pairs work?"

## Framing

SPECTRA core remains domain-agnostic. It accepts a `pairwise_similarity.csv`
with at least:

- `sample_id`
- `train_id`
- `similarity`

The computation registry gives agents reusable, literature-backed strategies
for producing that file at different scales. Each strategy declares the
compatible data shapes, exactness, required inputs, tunable parameters, quality
gates, failure modes, citations, and a prototype Python script.

## Seed-Mining Sources

The initial pass was mined from Paperclip searches and full-text inspection over
large-scale similarity search papers. The mined strategy families include:

- exact chunked all-pairs baselines,
- angular/random-projection LSH for cosine and angular distance,
- p-stable LSH for Lp distances,
- HNSW and FAISS-style dense vector approximate nearest-neighbor search,
- filtered ANN with metadata, cohort, temporal, and range constraints,
- NN-Descent and kNN graph construction,
- SSD/disk-backed vector ANN and vector similarity joins,
- GPU-accelerated similarity joins and top-k vector search,
- hybrid multimodal and multi-metric indexes for composite similarities,
- MIPS-specific asymmetric transforms, norm partitioning, and quantization,
- multi-vector late-interaction and vector-set search,
- MinHash/LSH for Jaccard, Tanimoto, shingles, and k-mers,
- weighted MinHash and consistent weighted sampling for generalized Jaccard,
- binary Hamming and multi-index hashing,
- streaming similarity joins over sliding windows and sketches,
- filter-and-verify similarity joins for thresholded graphs,
- edit-distance string similarity joins,
- structured graph/tree/JSON/AST edit-distance filter-and-verify search,
- sparse-vector inverted indexes and sparse MIPS,
- seed-and-extend sequence search,
- genomic sketching with Mash, sourmash, and FracMinHash,
- molecular fingerprint Tanimoto indexing,
- sparse-matrix and GPU cosine acceleration for mass-spectrometry peak lists,
- lower-bound pruning for DTW and expensive distances,
- blocking/filtering for record-like data,
- metric/spatial trees for radius and nearest-neighbor search,
- matrix-profile, symbolic, and subsequence indexes for time series,
- trajectory metric indexing and filter-and-refine search,
- Sinkhorn/sliced-Wasserstein approximations for distributions and histograms,
- ontology/taxonomic semantic similarity joins,
- graph-kernel sketches, random graph embeddings, and generic kernel feature-map approximations,
- learned candidate filters,
- privacy-preserving and federated similarity search,
- pivot, permutation, and Ptolemaic metric indexes,
- generic proxy-prefilter plus exact rerank pipelines,
- distributed partitioned ANN search.

## CLI

List strategies:

```bash
spectra similarity-computation list
```

Suggest a computation strategy:

```bash
spectra similarity-computation suggest \
  --dataset-description "1M regulatory DNA embeddings with cosine similarity" \
  --similarity-definition "embedding cosine similarity" \
  --data-shape dense_vector \
  --data-size "1 million eval/train vectors" \
  --required-input "dense embeddings"
```

Get one strategy:

```bash
spectra similarity-computation get dense_embedding_hnsw_topk
```

Get a prototype script:

```bash
spectra similarity-computation script minhash_lsh_jaccard_candidates
```

Validate the bundled registry:

```bash
spectra similarity-computation validate
```

## MCP Tools

The SPECTRA MCP server exposes:

- `list_similarity_computation_strategies`
- `get_similarity_computation_strategy`
- `suggest_similarity_computation_strategies`
- `get_similarity_computation_example_script`
- `render_similarity_computation_strategy`
- `validate_similarity_computation_registry`

Expected agent flow:

1. Read the dataset schema and scientific task.
2. Call `suggest_similarity_definitions` to pick a defensible similarity notion.
3. Call `suggest_similarity_computation_strategies` with the selected definition, data shape, scale, and available inputs.
4. Use the chosen prototype script or an equivalent production implementation to emit `pairwise_similarity.csv`.
5. Validate approximation recall, candidate coverage, or exactness assumptions.
6. Run `run_spectra_audit`.
7. Report the definition, computation strategy, quality gates, and limitations.

## Current Strategies

The bundled registry currently includes 38 seed strategies:

- `angular_random_projection_lsh`
- `binary_hamming_multi_index_hashing`
- `exact_chunked_all_pairs`
- `dense_embedding_hnsw_topk`
- `disk_backed_vector_ann_search`
- `faiss_ivf_pq_compressed_topk`
- `filtered_ann_attribute_constrained_search`
- `genomic_sketch_fracminhash_search`
- `graph_kernel_random_feature_sketch`
- `gpu_accelerated_similarity_join`
- `hybrid_multimodal_similarity_index`
- `kernel_approximation_feature_map`
- `learned_candidate_filter_similarity_join`
- `lp_stable_lsh_distance`
- `maximum_inner_product_search_transform`
- `minhash_lsh_jaccard_candidates`
- `mass_spectrometry_sparse_cosine_acceleration`
- `multi_vector_late_interaction_search`
- `nn_descent_neighbor_graph`
- `pivot_permutation_metric_index`
- `privacy_preserving_similarity_search`
- `sparse_inverted_index_topk`
- `semantic_ontology_similarity_join`
- `sequence_seed_prefilter_alignment_rerank`
- `sinkhorn_wasserstein_histogram_search`
- `molecular_fingerprint_tanimoto_index`
- `streaming_similarity_join_sketch`
- `string_edit_similarity_join`
- `structured_edit_distance_filter_verify`
- `time_series_matrix_profile_index`
- `dtw_lower_bound_pruning`
- `blocking_candidate_generation`
- `metric_tree_radius_knn`
- `threshold_similarity_join_filter_verify`
- `trajectory_metric_index_filter_refine`
- `two_stage_prefilter_rerank`
- `distributed_partitioned_ann_search`
- `weighted_minhash_generalized_jaccard`

## Release Claim

A defensible release claim is:

> `/spectra` separates scientific similarity definition from scalable
> similarity computation. Agents can mine or choose a domain-appropriate
> similarity notion, then select a computation strategy that preserves the audit
> contract while reducing all-pairs cost when appropriate.

This is stronger than a molecule-specific package and more useful than a random
wrapper: it gives agents a structured bridge from paper-mined similarity notions
to reproducible SPECTRA input files.
