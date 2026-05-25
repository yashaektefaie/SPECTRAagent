import csv
import importlib.util
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "spectrae" / "similarity_computation_registry.py"
    spec = importlib.util.spec_from_file_location("similarity_computation_registry_direct", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimilarityComputationRegistryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.out_dir = self.root / "build" / "test_similarity_computation_registry"
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True)

    def test_registry_loads_and_validates(self):
        mod = load_module()
        catalog = mod.list_similarity_computation_strategies()
        self.assertGreaterEqual(catalog["count"], 38)
        ids = {item["id"] for item in catalog["strategies"]}
        self.assertIn("angular_random_projection_lsh", ids)
        self.assertIn("binary_hamming_multi_index_hashing", ids)
        self.assertIn("exact_chunked_all_pairs", ids)
        self.assertIn("dense_embedding_hnsw_topk", ids)
        self.assertIn("disk_backed_vector_ann_search", ids)
        self.assertIn("filtered_ann_attribute_constrained_search", ids)
        self.assertIn("genomic_sketch_fracminhash_search", ids)
        self.assertIn("graph_kernel_random_feature_sketch", ids)
        self.assertIn("gpu_accelerated_similarity_join", ids)
        self.assertIn("hybrid_multimodal_similarity_index", ids)
        self.assertIn("kernel_approximation_feature_map", ids)
        self.assertIn("learned_candidate_filter_similarity_join", ids)
        self.assertIn("lp_stable_lsh_distance", ids)
        self.assertIn("maximum_inner_product_search_transform", ids)
        self.assertIn("minhash_lsh_jaccard_candidates", ids)
        self.assertIn("mass_spectrometry_sparse_cosine_acceleration", ids)
        self.assertIn("multi_vector_late_interaction_search", ids)
        self.assertIn("nn_descent_neighbor_graph", ids)
        self.assertIn("pivot_permutation_metric_index", ids)
        self.assertIn("privacy_preserving_similarity_search", ids)
        self.assertIn("sequence_seed_prefilter_alignment_rerank", ids)
        self.assertIn("semantic_ontology_similarity_join", ids)
        self.assertIn("sinkhorn_wasserstein_histogram_search", ids)
        self.assertIn("streaming_similarity_join_sketch", ids)
        self.assertIn("string_edit_similarity_join", ids)
        self.assertIn("structured_edit_distance_filter_verify", ids)
        self.assertIn("time_series_matrix_profile_index", ids)
        self.assertIn("blocking_candidate_generation", ids)
        self.assertIn("threshold_similarity_join_filter_verify", ids)
        self.assertIn("trajectory_metric_index_filter_refine", ids)
        self.assertIn("distributed_partitioned_ann_search", ids)
        self.assertIn("weighted_minhash_generalized_jaccard", ids)

        validation = mod.validate_similarity_computation_registry()
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["count"], catalog["count"])

    def test_search_suggests_scale_relevant_strategies(self):
        mod = load_module()
        dense = mod.suggest_similarity_computation_strategies(
            dataset_description="one million biomedical text embeddings",
            similarity_definition="cosine embedding similarity",
            data_shape="dense_vector",
            data_size="1 million vectors",
            required_inputs=["dense embeddings"],
            top_k=5,
        )
        dense_ids = [item["id"] for item in dense["results"]]
        self.assertIn("dense_embedding_hnsw_topk", dense_ids)

        sets = mod.suggest_similarity_computation_strategies(
            dataset_description="large sequence kmer sets with Jaccard similarity",
            similarity_definition="jaccard set overlap",
            data_shape="set_features",
            data_size="large",
            required_inputs=["set-valued features"],
            top_k=5,
        )
        set_ids = [item["id"] for item in sets["results"]]
        self.assertIn("minhash_lsh_jaccard_candidates", set_ids)

        records = mod.suggest_similarity_computation_strategies(
            dataset_description="clinical records with hospital and assay metadata blocks",
            similarity_definition="metadata record similarity",
            data_shape="tabular",
            data_size="large",
            required_inputs=["blocking columns or cheap blocking rule"],
            top_k=5,
        )
        record_ids = [item["id"] for item in records["results"]]
        self.assertIn("blocking_candidate_generation", record_ids)

        weighted = mod.suggest_similarity_computation_strategies(
            dataset_description="weighted ontology and abundance feature profiles",
            similarity_definition="generalized weighted Jaccard",
            data_shape="weighted_set_features",
            data_size="large",
            required_inputs=["nonnegative weighted feature sets"],
            top_k=5,
        )
        weighted_ids = [item["id"] for item in weighted["results"]]
        self.assertIn("weighted_minhash_generalized_jaccard", weighted_ids)

        spectra = mod.suggest_similarity_computation_strategies(
            dataset_description="MS/MS peak lists for repository scale spectral library search",
            similarity_definition="mass spectrum cosine similarity",
            data_shape="msms_peak_list",
            data_size="large",
            required_inputs=["peak lists with mz and intensity"],
            top_k=5,
        )
        spectra_ids = [item["id"] for item in spectra["results"]]
        self.assertIn("mass_spectrometry_sparse_cosine_acceleration", spectra_ids)

        histograms = mod.suggest_similarity_computation_strategies(
            dataset_description="histogram distributions where bin geometry matters",
            similarity_definition="Wasserstein optimal transport distance",
            data_shape="histogram",
            data_size="large",
            required_inputs=["normalized histograms or distributions"],
            top_k=5,
        )
        histogram_ids = [item["id"] for item in histograms["results"]]
        self.assertIn("sinkhorn_wasserstein_histogram_search", histogram_ids)

        hamming = mod.suggest_similarity_computation_strategies(
            dataset_description="binary hash code embeddings with Hamming distance",
            similarity_definition="binary code Hamming similarity",
            data_shape="binary_code",
            data_size="large",
            required_inputs=["binary codes"],
            top_k=5,
        )
        hamming_ids = [item["id"] for item in hamming["results"]]
        self.assertIn("binary_hamming_multi_index_hashing", hamming_ids)

        strings = mod.suggest_similarity_computation_strategies(
            dataset_description="large collection of string identifiers and barcodes",
            similarity_definition="edit distance string similarity",
            data_shape="string",
            data_size="large",
            required_inputs=["strings"],
            top_k=5,
        )
        string_ids = [item["id"] for item in strings["results"]]
        self.assertIn("string_edit_similarity_join", string_ids)

        kernels = mod.suggest_similarity_computation_strategies(
            dataset_description="large RBF kernel similarity over numeric scientific features",
            similarity_definition="random Fourier feature RBF kernel",
            data_shape="kernelized_samples",
            data_size="large",
            required_inputs=["input features or kernel landmarks"],
            top_k=5,
        )
        kernel_ids = [item["id"] for item in kernels["results"]]
        self.assertIn("kernel_approximation_feature_map", kernel_ids)

        filtered = mod.suggest_similarity_computation_strategies(
            dataset_description="large patient embeddings filtered by site time window and cohort metadata",
            similarity_definition="cosine similarity with attribute filters",
            data_shape="attribute_filtered_vector",
            data_size="large",
            required_inputs=["filter attributes", "filter predicate"],
            top_k=5,
        )
        filtered_ids = [item["id"] for item in filtered["results"]]
        self.assertIn("filtered_ann_attribute_constrained_search", filtered_ids)

        streaming = mod.suggest_similarity_computation_strategies(
            dataset_description="streaming set events over sliding windows",
            similarity_definition="streaming Jaccard similarity",
            data_shape="sliding_window_stream",
            data_size="large",
            required_inputs=["window definition", "similarity sketch parameters"],
            top_k=5,
        )
        streaming_ids = [item["id"] for item in streaming["results"]]
        self.assertIn("streaming_similarity_join_sketch", streaming_ids)

        time_series = mod.suggest_similarity_computation_strategies(
            dataset_description="long sensor traces with matrix profile subsequence search",
            similarity_definition="z-normalized time series shape similarity",
            data_shape="subsequence_windows",
            data_size="large",
            required_inputs=["window length or subsequence definition"],
            top_k=5,
        )
        time_series_ids = [item["id"] for item in time_series["results"]]
        self.assertIn("time_series_matrix_profile_index", time_series_ids)

        structured = mod.suggest_similarity_computation_strategies(
            dataset_description="large JSON documents and ASTs compared by structured edit distance",
            similarity_definition="tree edit distance",
            data_shape="json_document",
            data_size="large",
            required_inputs=["structured objects", "edit cost model"],
            top_k=5,
        )
        structured_ids = [item["id"] for item in structured["results"]]
        self.assertIn("structured_edit_distance_filter_verify", structured_ids)

        mips = mod.suggest_similarity_computation_strategies(
            dataset_description="recommendation embeddings scored by maximum inner product",
            similarity_definition="dot product similarity",
            data_shape="recommendation_embedding",
            data_size="large",
            required_inputs=["inner product scorer"],
            top_k=5,
        )
        mips_ids = [item["id"] for item in mips["results"]]
        self.assertIn("maximum_inner_product_search_transform", mips_ids)

        multi_vector = mod.suggest_similarity_computation_strategies(
            dataset_description="protein domains represented as sets of local vectors",
            similarity_definition="multi-vector MaxSim similarity",
            data_shape="set_of_vectors",
            data_size="large",
            required_inputs=["sets of local vectors", "late interaction scorer"],
            top_k=5,
        )
        multi_vector_ids = [item["id"] for item in multi_vector["results"]]
        self.assertIn("multi_vector_late_interaction_search", multi_vector_ids)

        private = mod.suggest_similarity_computation_strategies(
            dataset_description="federated genomic records across hospitals using secure near neighbor search",
            similarity_definition="privacy preserving genomic similarity",
            data_shape="federated_genomic_data",
            data_size="large",
            required_inputs=["privacy protocol"],
            top_k=5,
        )
        private_ids = [item["id"] for item in private["results"]]
        self.assertIn("privacy_preserving_similarity_search", private_ids)

        gpu = mod.suggest_similarity_computation_strategies(
            dataset_description="GPU accelerated exact top-k cosine similarity join",
            similarity_definition="cosine similarity",
            data_shape="binary_quantized_vector",
            data_size="large",
            required_inputs=["gpu execution configuration"],
            top_k=5,
        )
        gpu_ids = [item["id"] for item in gpu["results"]]
        self.assertIn("gpu_accelerated_similarity_join", gpu_ids)

        ontology = mod.suggest_similarity_computation_strategies(
            dataset_description="gene ontology annotated samples with semantic overlap",
            similarity_definition="ontology semantic similarity",
            data_shape="gene_ontology_terms",
            data_size="large",
            required_inputs=["ontology annotations", "ontology graph or taxonomy"],
            top_k=5,
        )
        ontology_ids = [item["id"] for item in ontology["results"]]
        self.assertIn("semantic_ontology_similarity_join", ontology_ids)

        multimodal = mod.suggest_similarity_computation_strategies(
            dataset_description="geo-tagged images with spatial visual and text similarity",
            similarity_definition="composite multimodal similarity",
            data_shape="geo_tagged_image",
            data_size="large",
            required_inputs=["multimodal features", "composite similarity formula"],
            top_k=5,
        )
        multimodal_ids = [item["id"] for item in multimodal["results"]]
        self.assertIn("hybrid_multimodal_similarity_index", multimodal_ids)

    def test_dense_vector_example_script_runs(self):
        train_path = self.out_dir / "dense_train.csv"
        eval_path = self.out_dir / "dense_eval.csv"
        out_path = self.out_dir / "dense_pairwise.csv"
        with train_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "emb_0", "emb_1"])
            writer.writeheader()
            writer.writerows(
                [
                    {"sample_id": "t0", "emb_0": "1", "emb_1": "0"},
                    {"sample_id": "t1", "emb_0": "0", "emb_1": "1"},
                ]
            )
        with eval_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "emb_0", "emb_1"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "emb_0": "1", "emb_1": "0"})

        script = self.root / "spectrae" / "similarity_computation" / "examples" / "dense_vector_topk_similarity.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(out_path),
                "--top-k",
                "1",
            ],
            check=True,
        )
        with out_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["train_id"], "t0")
        self.assertAlmostEqual(float(rows[0]["similarity"]), 1.0)

    def test_set_blocking_and_bitset_scripts_run(self):
        train_path = self.out_dir / "train.csv"
        eval_path = self.out_dir / "eval.csv"
        minhash_path = self.out_dir / "minhash_pairwise.csv"
        blocking_path = self.out_dir / "blocking_pairwise.csv"
        bitset_path = self.out_dir / "bitset_pairwise.csv"
        with train_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "tokens", "site", "fingerprint"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"sample_id": "t0", "tokens": "a;b;c", "site": "A", "fingerprint": "1100"},
                    {"sample_id": "t1", "tokens": "x;y", "site": "B", "fingerprint": "0011"},
                ]
            )
        with eval_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "tokens", "site", "fingerprint"],
            )
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "tokens": "a;b", "site": "A", "fingerprint": "1000"})

        examples = self.root / "spectrae" / "similarity_computation" / "examples"
        subprocess.run(
            [
                sys.executable,
                str(examples / "minhash_lsh_jaccard.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(minhash_path),
                "--set-col",
                "tokens",
                "--fallback-exact",
                "--top-k",
                "1",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(examples / "blocking_candidate_similarity.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(blocking_path),
                "--block-cols",
                "site",
                "--token-col",
                "tokens",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(examples / "bitset_tanimoto_topk.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(bitset_path),
                "--fingerprint-col",
                "fingerprint",
                "--top-k",
                "1",
            ],
            check=True,
        )

        for path in [minhash_path, blocking_path, bitset_path]:
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["sample_id"], "e0")
            self.assertEqual(rows[0]["train_id"], "t0")

    def test_second_pass_strategy_scripts_run(self):
        train_path = self.out_dir / "second_pass_train.csv"
        eval_path = self.out_dir / "second_pass_eval.csv"
        random_lsh_path = self.out_dir / "random_lsh_pairwise.csv"
        weighted_path = self.out_dir / "weighted_pairwise.csv"
        threshold_path = self.out_dir / "threshold_pairwise.csv"
        sinkhorn_path = self.out_dir / "sinkhorn_pairwise.csv"
        peaks_path = self.out_dir / "peaks_pairwise.csv"
        hamming_path = self.out_dir / "hamming_pairwise.csv"
        edit_path = self.out_dir / "edit_pairwise.csv"
        with train_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "emb_0",
                    "emb_1",
                    "weighted_terms",
                    "terms",
                    "bin_0",
                    "bin_1",
                    "peaks",
                    "code",
                    "name",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "sample_id": "t0",
                        "emb_0": "1",
                        "emb_1": "0",
                        "weighted_terms": "a:2;b:1",
                        "terms": "a;b",
                        "bin_0": "1",
                        "bin_1": "0",
                        "peaks": "100.000:10;150.000:3",
                        "code": "1100",
                        "name": "ABCD",
                    },
                    {
                        "sample_id": "t1",
                        "emb_0": "0",
                        "emb_1": "1",
                        "weighted_terms": "x:1",
                        "terms": "x",
                        "bin_0": "0",
                        "bin_1": "1",
                        "peaks": "200.000:5",
                        "code": "0011",
                        "name": "WXYZ",
                    },
                ]
            )
        with eval_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "emb_0",
                    "emb_1",
                    "weighted_terms",
                    "terms",
                    "bin_0",
                    "bin_1",
                    "peaks",
                    "code",
                    "name",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "e0",
                    "emb_0": "1",
                    "emb_1": "0",
                    "weighted_terms": "a:1;b:1",
                    "terms": "a;b",
                    "bin_0": "1",
                    "bin_1": "0",
                    "peaks": "100.000:4;150.000:1",
                    "code": "1000",
                    "name": "ABCE",
                }
            )

        examples = self.root / "spectrae" / "similarity_computation" / "examples"
        subprocess.run(
            [
                sys.executable,
                str(examples / "random_projection_lsh_cosine.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(random_lsh_path),
                "--num-bits",
                "2",
                "--num-tables",
                "2",
                "--fallback-exact",
                "--top-k",
                "1",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(examples / "weighted_jaccard_topk.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(weighted_path),
                "--feature-col",
                "weighted_terms",
                "--top-k",
                "1",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(examples / "set_similarity_threshold_join.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(threshold_path),
                "--set-col",
                "terms",
                "--threshold",
                "0.5",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(examples / "histogram_sinkhorn_topk.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(sinkhorn_path),
                "--histogram-prefix",
                "bin_",
                "--top-k",
                "1",
                "--iterations",
                "10",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(examples / "sparse_peak_cosine_topk.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(peaks_path),
                "--peak-col",
                "peaks",
                "--top-k",
                "1",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(examples / "binary_hamming_topk.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(hamming_path),
                "--code-col",
                "code",
                "--top-k",
                "1",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(examples / "string_qgram_edit_join.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(edit_path),
                "--text-col",
                "name",
                "--q",
                "2",
                "--max-distance",
                "1",
                "--fallback-exact",
            ],
            check=True,
        )

        for path in [random_lsh_path, weighted_path, threshold_path, sinkhorn_path, peaks_path, hamming_path, edit_path]:
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["sample_id"], "e0")
            self.assertEqual(rows[0]["train_id"], "t0")

    def test_script_resource_round_trip(self):
        mod = load_module()
        script = mod.get_similarity_computation_example_script("minhash_lsh_jaccard_candidates")
        self.assertEqual(script["mime_type"], "text/x-python")
        self.assertIn("MinHash", script["content"])
        rendered = mod.render_similarity_computation_strategy("minhash_lsh_jaccard_candidates")
        self.assertEqual(rendered["mime_type"], "text/markdown")
        self.assertIn("MinHash LSH", rendered["content"])


if __name__ == "__main__":
    unittest.main()
