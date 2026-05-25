import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "spectrae" / "similarity_registry.py"
    spec = importlib.util.spec_from_file_location("similarity_registry_direct", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimilarityRegistryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.out_dir = self.root / "build" / "test_similarity_registry"
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True)

    def test_registry_loads_and_validates(self):
        mod = load_module()
        catalog = mod.list_similarity_definitions()
        self.assertGreaterEqual(catalog["count"], 54)
        ids = {item["id"] for item in catalog["definitions"]}
        self.assertIn("atomistic_soap_kernel_similarity", ids)
        self.assertIn("biological_sequence_alignment_homology", ids)
        self.assertIn("biomedical_text_semantic_similarity", ids)
        self.assertIn("cell_morphology_profile_similarity", ids)
        self.assertIn("climate_niche_environmental_similarity", ids)
        self.assertIn("connectome_matrix_correlation_similarity", ids)
        self.assertIn("clinical_tabular_patient_similarity", ids)
        self.assertIn("gene_expression_signature_connectivity_similarity", ids)
        self.assertIn("genotype_ibd_ibs_similarity", ids)
        self.assertIn("interactome_network_proximity_similarity", ids)
        self.assertIn("longitudinal_patient_trajectory_similarity", ids)
        self.assertIn("molecules_morgan_tanimoto", ids)
        self.assertIn("multi_omics_similarity_network_fusion", ids)
        self.assertIn("protein_binding_pocket_similarity", ids)
        self.assertIn("single_cell_logfc_cosine", ids)
        self.assertIn("protein_mutational_regime_hamming", ids)
        self.assertIn("protein_structure_tm_score_similarity", ids)
        self.assertIn("survival_metric_learned_patient_similarity", ids)
        self.assertIn("time_series_dtw_shape_similarity", ids)
        self.assertIn("topological_persistence_diagram_similarity", ids)
        self.assertIn("graph_neighborhood_jaccard_similarity", ids)
        self.assertIn("domain_distribution_mmd_similarity", ids)
        self.assertIn("experimental_batch_protocol_similarity", ids)
        self.assertIn("mass_spectrometry_spectral_cosine_similarity", ids)
        self.assertIn("microbiome_beta_diversity_similarity", ids)
        self.assertIn("point_cloud_point_set_distance_similarity", ids)
        self.assertIn("spatial_prediction_horizon_similarity", ids)
        self.assertIn("hyperspectral_spectral_angle_similarity", ids)
        self.assertIn("imaging_histogram_radiomics_similarity", ids)
        self.assertIn("materials_composition_formula_similarity", ids)
        self.assertIn("geospatial_haversine_region_distance", ids)
        self.assertIn("medical_imaging_scanner_protocol_domain", ids)

        validation = mod.validate_similarity_registry()
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["count"], catalog["count"])

    def test_search_suggests_domain_relevant_entries(self):
        mod = load_module()
        result = mod.suggest_similarity_definitions(
            dataset_description="SMILES molecular property model for virtual screening",
            task_description="audit generalization to chemically novel molecules",
            data_type="molecule",
            required_inputs=["sample_id", "train_id", "smiles"],
            top_k=3,
        )
        ids = [item["id"] for item in result["results"]]
        self.assertIn("molecules_morgan_tanimoto", ids)

        material_result = mod.suggest_similarity_definitions(
            dataset_description="materials property prediction from formulas and crystal structures",
            task_description="audit composition and structure OOD generalization",
            data_type="material",
            required_inputs=["sample_id", "train_id", "formula"],
            top_k=5,
        )
        material_ids = [item["id"] for item in material_result["results"]]
        self.assertIn("materials_composition_formula_similarity", material_ids)
        self.assertIn("materials_structure_ofm_embedding_similarity", material_ids)

        graph_result = mod.suggest_similarity_definitions(
            dataset_description="node classification graph with edge list and graph neighborhoods",
            task_description="audit graph OOD generalization across topology and neighborhood overlap",
            data_type="graph",
            required_inputs=["sample_id", "train_id", "node_id", "edge_list"],
            top_k=5,
        )
        graph_ids = [item["id"] for item in graph_result["results"]]
        self.assertIn("graph_neighborhood_jaccard_similarity", graph_ids)

        time_series_result = mod.suggest_similarity_definitions(
            dataset_description="sensor time series windows with temporal domains and source devices",
            task_description="audit OOD generalization as time series shape and domain similarity changes",
            data_type="time_series",
            required_inputs=["sample_id", "train_id", "series"],
            top_k=5,
        )
        time_series_ids = [item["id"] for item in time_series_result["results"]]
        self.assertIn("time_series_dtw_shape_similarity", time_series_ids)

        ms_result = mod.suggest_similarity_definitions(
            dataset_description="MS/MS peak lists for metabolomics molecular networking",
            task_description="audit spectral library generalization to novel spectra",
            data_type="mass_spectrometry",
            required_inputs=["sample_id", "train_id", "peak_list"],
            top_k=5,
        )
        ms_ids = [item["id"] for item in ms_result["results"]]
        self.assertIn("mass_spectrometry_spectral_cosine_similarity", ms_ids)

        text_result = mod.suggest_similarity_definitions(
            dataset_description="clinical notes embedded with biomedical sentence transformer vectors",
            task_description="audit NLP generalization across specialties and note styles",
            data_type="clinical text",
            required_inputs=["sample_id", "train_id", "embedding_columns_or_text_encoder"],
            top_k=5,
        )
        text_ids = [item["id"] for item in text_result["results"]]
        self.assertIn("biomedical_text_semantic_similarity", text_ids)

        omics_result = mod.suggest_similarity_definitions(
            dataset_description="multi-omics patient samples with RNA methylation and miRNA features",
            task_description="audit generalization across molecular subtypes and fused sample similarity",
            data_type="multi_omics",
            required_inputs=["sample_id", "train_id", "omics_view_embeddings_or_precomputed_similarities"],
            top_k=5,
        )
        omics_ids = [item["id"] for item in omics_result["results"]]
        self.assertIn("multi_omics_similarity_network_fusion", omics_ids)

        trajectory_result = mod.suggest_similarity_definitions(
            dataset_description="longitudinal EHR event trajectories for patient risk prediction",
            task_description="audit trajectory novelty with event sequence alignment",
            data_type="longitudinal_ehr",
            required_inputs=["sample_id", "train_id", "event_sequence_or_time_series"],
            top_k=5,
        )
        trajectory_ids = [item["id"] for item in trajectory_result["results"]]
        self.assertIn("longitudinal_patient_trajectory_similarity", trajectory_ids)

    def test_example_script_runs(self):
        train_path = self.out_dir / "train.csv"
        eval_path = self.out_dir / "eval.csv"
        out_path = self.out_dir / "pairwise_similarity.csv"
        with train_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "sequence"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "sequence": "AAAA"},
                    {"train_id": "t1", "sequence": "AATT"},
                ]
            )
        with eval_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "sequence": "AAAT"})

        script = (
            self.root
            / "spectrae"
            / "similarity_definitions"
            / "examples"
            / "sequence_hamming_similarity.py"
        )
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
            ],
            check=True,
        )
        with out_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["sample_id"], "e0")
        self.assertAlmostEqual(float(rows[0]["similarity"]), 0.75)

    def test_script_resource_round_trip(self):
        mod = load_module()
        script = mod.get_similarity_example_script("regulatory_dna_motif_bag_similarity")
        self.assertEqual(script["mime_type"], "text/x-python")
        self.assertIn("cosine", script["content"])
        rendered = mod.render_similarity_definition("regulatory_dna_motif_bag_similarity")
        self.assertEqual(rendered["mime_type"], "text/markdown")
        self.assertIn("Bag-of-motifs", rendered["content"])

    def test_expanded_generic_scripts_run(self):
        train_path = self.out_dir / "train_generic.csv"
        eval_path = self.out_dir / "eval_generic.csv"
        formula_path = self.out_dir / "formula_pairwise.csv"
        set_path = self.out_dir / "set_pairwise.csv"
        geo_path = self.out_dir / "geo_pairwise.csv"
        composite_path = self.out_dir / "composite_pairwise.csv"
        with train_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["train_id", "formula", "terms", "lat", "lon"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "formula": "H2O", "terms": "a;b", "lat": "0", "lon": "0"},
                    {"train_id": "t1", "formula": "CO2", "terms": "c", "lat": "10", "lon": "0"},
                ]
            )
        with eval_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "formula", "terms", "lat", "lon"],
            )
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "formula": "H2O", "terms": "a", "lat": "0", "lon": "0"})

        examples = self.root / "spectrae" / "similarity_definitions" / "examples"
        subprocess.run(
            [
                sys.executable,
                str(examples / "formula_composition_similarity.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(formula_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(examples / "set_jaccard_similarity.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(set_path),
                "--set-col",
                "terms",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(examples / "geospatial_haversine_similarity.py"),
                "--train",
                str(train_path),
                "--eval",
                str(eval_path),
                "--out",
                str(geo_path),
                "--scale-km",
                "1000",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(examples / "composite_pairwise_similarity.py"),
                "--inputs",
                str(formula_path),
                str(set_path),
                "--method",
                "min",
                "--out",
                str(composite_path),
            ],
            check=True,
        )

        with composite_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[0]["similarity"]), 0.5)

    def test_curated_primitive_scripts_run(self):
        examples = self.root / "spectrae" / "similarity_definitions" / "examples"

        seq_train = self.out_dir / "seq_train.csv"
        seq_eval = self.out_dir / "seq_eval.csv"
        seq_out = self.out_dir / "alignment_pairwise.csv"
        with seq_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "sequence"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "sequence": "ACGT"},
                    {"train_id": "t1", "sequence": "AAAA"},
                ]
            )
        with seq_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "sequence": "ACGT"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "alignment_identity_similarity.py"),
                "--train",
                str(seq_train),
                "--eval",
                str(seq_eval),
                "--out",
                str(seq_out),
            ],
            check=True,
        )
        with seq_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[0]["similarity"]), 1.0)

        graph_edges = self.out_dir / "edges.csv"
        graph_train = self.out_dir / "graph_train.csv"
        graph_eval = self.out_dir / "graph_eval.csv"
        graph_out = self.out_dir / "graph_pairwise.csv"
        with graph_edges.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source", "target"])
            writer.writeheader()
            writer.writerows(
                [
                    {"source": "A", "target": "B"},
                    {"source": "B", "target": "C"},
                    {"source": "C", "target": "D"},
                ]
            )
        with graph_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "node_id"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "node_id": "A"},
                    {"train_id": "t1", "node_id": "D"},
                ]
            )
        with graph_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "node_id"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "node_id": "B"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "graph_neighborhood_similarity.py"),
                "--edges",
                str(graph_edges),
                "--train",
                str(graph_train),
                "--eval",
                str(graph_eval),
                "--out",
                str(graph_out),
                "--k",
                "1",
            ],
            check=True,
        )
        with graph_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[1]["similarity"]), 0.5)

        ts_train = self.out_dir / "ts_train.csv"
        ts_eval = self.out_dir / "ts_eval.csv"
        ts_out = self.out_dir / "ts_pairwise.csv"
        with ts_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "series"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "series": "0;1;2"},
                    {"train_id": "t1", "series": "10;10;10"},
                ]
            )
        with ts_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "series"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "series": "0;1;2"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "time_series_dtw_similarity.py"),
                "--train",
                str(ts_train),
                "--eval",
                str(ts_eval),
                "--out",
                str(ts_out),
            ],
            check=True,
        )
        with ts_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(rows[0]["similarity"]), 1.0)
        self.assertGreater(float(rows[0]["similarity"]), float(rows[1]["similarity"]))

        metric_in = self.out_dir / "metric.csv"
        metric_out = self.out_dir / "metric_pairwise.csv"
        with metric_in.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "train_id", "metric"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "train_id": "t0", "metric": "1.0"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "metric_to_similarity.py"),
                "--input",
                str(metric_in),
                "--out",
                str(metric_out),
                "--metric-kind",
                "distance",
                "--distance-method",
                "linear",
                "--scale",
                "2.0",
            ],
            check=True,
        )
        with metric_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(rows[0]["similarity"]), 0.5)

        domain_train = self.out_dir / "domain_train.csv"
        domain_eval = self.out_dir / "domain_eval.csv"
        domain_out = self.out_dir / "domain_pairwise.csv"
        with domain_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "domain", "feat_0"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "domain": "d0", "feat_0": "0.0"},
                    {"train_id": "t1", "domain": "d1", "feat_0": "10.0"},
                ]
            )
        with domain_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "domain", "feat_0"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "domain": "d0", "feat_0": "0.1"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "domain_distribution_similarity.py"),
                "--train",
                str(domain_train),
                "--eval",
                str(domain_eval),
                "--out",
                str(domain_out),
                "--domain-col",
                "domain",
                "--feature-cols-prefix",
                "feat_",
            ],
            check=True,
        )
        with domain_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreater(float(rows[0]["similarity"]), float(rows[1]["similarity"]))

        hist_train = self.out_dir / "hist_train.csv"
        hist_eval = self.out_dir / "hist_eval.csv"
        hist_out = self.out_dir / "hist_pairwise.csv"
        with hist_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "hist_0", "hist_1"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "hist_0": "1", "hist_1": "0"},
                    {"train_id": "t1", "hist_0": "0", "hist_1": "1"},
                ]
            )
        with hist_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "hist_0", "hist_1"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "hist_0": "1", "hist_1": "0"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "histogram_js_similarity.py"),
                "--train",
                str(hist_train),
                "--eval",
                str(hist_eval),
                "--out",
                str(hist_out),
                "--hist-cols-prefix",
                "hist_",
            ],
            check=True,
        )
        with hist_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(rows[0]["similarity"]), 1.0)
        self.assertGreater(float(rows[0]["similarity"]), float(rows[1]["similarity"]))

    def test_second_pass_scripts_run(self):
        examples = self.root / "spectrae" / "similarity_definitions" / "examples"

        comp_train = self.out_dir / "comp_train.csv"
        comp_eval = self.out_dir / "comp_eval.csv"
        comp_out = self.out_dir / "comp_pairwise.csv"
        with comp_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "taxon_a", "taxon_b"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "taxon_a": "1", "taxon_b": "0"},
                    {"train_id": "t1", "taxon_a": "0", "taxon_b": "1"},
                ]
            )
        with comp_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "taxon_a", "taxon_b"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "taxon_a": "1", "taxon_b": "0"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "composition_braycurtis_similarity.py"),
                "--train",
                str(comp_train),
                "--eval",
                str(comp_eval),
                "--out",
                str(comp_out),
                "--feature-cols-prefix",
                "taxon_",
            ],
            check=True,
        )
        with comp_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(rows[0]["similarity"]), 1.0)
        self.assertAlmostEqual(float(rows[1]["similarity"]), 0.0)

        spectrum_train = self.out_dir / "spectrum_train.csv"
        spectrum_eval = self.out_dir / "spectrum_eval.csv"
        spectrum_out = self.out_dir / "spectrum_pairwise.csv"
        with spectrum_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "peaks"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "peaks": "100:1;200:1"},
                    {"train_id": "t1", "peaks": "150:1"},
                ]
            )
        with spectrum_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "peaks"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "peaks": "100:1;200:1"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "spectrum_cosine_similarity.py"),
                "--train",
                str(spectrum_train),
                "--eval",
                str(spectrum_eval),
                "--out",
                str(spectrum_out),
            ],
            check=True,
        )
        with spectrum_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(rows[0]["similarity"]), 1.0)
        self.assertAlmostEqual(float(rows[1]["similarity"]), 0.0)

        angle_train = self.out_dir / "angle_train.csv"
        angle_eval = self.out_dir / "angle_eval.csv"
        angle_out = self.out_dir / "angle_pairwise.csv"
        with angle_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "band_0", "band_1"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "band_0": "1", "band_1": "0"},
                    {"train_id": "t1", "band_0": "0", "band_1": "1"},
                ]
            )
        with angle_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "band_0", "band_1"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "band_0": "1", "band_1": "0"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "spectral_angle_similarity.py"),
                "--train",
                str(angle_train),
                "--eval",
                str(angle_eval),
                "--out",
                str(angle_out),
                "--band-cols-prefix",
                "band_",
            ],
            check=True,
        )
        with angle_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(rows[0]["similarity"]), 1.0)
        self.assertGreater(float(rows[0]["similarity"]), float(rows[1]["similarity"]))

        cloud_train = self.out_dir / "cloud_train.csv"
        cloud_eval = self.out_dir / "cloud_eval.csv"
        cloud_out = self.out_dir / "cloud_pairwise.csv"
        with cloud_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "points"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "points": "0:0:0;1:0:0"},
                    {"train_id": "t1", "points": "10:0:0;11:0:0"},
                ]
            )
        with cloud_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "points"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "points": "0:0:0;1:0:0"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "point_cloud_chamfer_similarity.py"),
                "--train",
                str(cloud_train),
                "--eval",
                str(cloud_eval),
                "--out",
                str(cloud_out),
            ],
            check=True,
        )
        with cloud_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(rows[0]["similarity"]), 1.0)
        self.assertGreater(float(rows[0]["similarity"]), float(rows[1]["similarity"]))

        tab_train = self.out_dir / "tab_train.csv"
        tab_eval = self.out_dir / "tab_eval.csv"
        tab_out = self.out_dir / "tab_pairwise.csv"
        with tab_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "cov_0"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "cov_0": "0"},
                    {"train_id": "t1", "cov_0": "10"},
                ]
            )
        with tab_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "cov_0"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "cov_0": "0"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "tabular_mahalanobis_similarity.py"),
                "--train",
                str(tab_train),
                "--eval",
                str(tab_eval),
                "--out",
                str(tab_out),
                "--feature-cols-prefix",
                "cov_",
            ],
            check=True,
        )
        with tab_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(rows[0]["similarity"]), 1.0)
        self.assertGreater(float(rows[0]["similarity"]), float(rows[1]["similarity"]))

    def test_third_pass_scripts_run(self):
        examples = self.root / "spectrae" / "similarity_definitions" / "examples"

        vector_train = self.out_dir / "vector_train.csv"
        vector_eval = self.out_dir / "vector_eval.csv"
        vector_out = self.out_dir / "vector_pairwise.csv"
        with vector_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "feat_0", "feat_1", "feat_2"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "feat_0": "1", "feat_1": "2", "feat_2": "3"},
                    {"train_id": "t1", "feat_0": "3", "feat_1": "2", "feat_2": "1"},
                ]
            )
        with vector_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "feat_0", "feat_1", "feat_2"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "feat_0": "1", "feat_1": "2", "feat_2": "3"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "vector_correlation_similarity.py"),
                "--train",
                str(vector_train),
                "--eval",
                str(vector_eval),
                "--out",
                str(vector_out),
                "--feature-cols-prefix",
                "feat_",
            ],
            check=True,
        )
        with vector_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(rows[0]["similarity"]), 1.0)
        self.assertAlmostEqual(float(rows[1]["similarity"]), 0.0)

        event_train = self.out_dir / "event_train.csv"
        event_eval = self.out_dir / "event_eval.csv"
        event_out = self.out_dir / "event_pairwise.csv"
        with event_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "events"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "events": "A;B;C"},
                    {"train_id": "t1", "events": "X;Y;Z"},
                ]
            )
        with event_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "events"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "events": "A;B;D"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "event_sequence_alignment_similarity.py"),
                "--train",
                str(event_train),
                "--eval",
                str(event_eval),
                "--out",
                str(event_out),
            ],
            check=True,
        )
        with event_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreater(float(rows[0]["similarity"]), float(rows[1]["similarity"]))

        network_edges = self.out_dir / "network_edges.csv"
        network_train = self.out_dir / "network_train.csv"
        network_eval = self.out_dir / "network_eval.csv"
        network_out = self.out_dir / "network_pairwise.csv"
        with network_edges.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source", "target"])
            writer.writeheader()
            writer.writerows(
                [
                    {"source": "A", "target": "B"},
                    {"source": "B", "target": "C"},
                    {"source": "X", "target": "Y"},
                ]
            )
        with network_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "entities"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "entities": "A"},
                    {"train_id": "t1", "entities": "Y"},
                ]
            )
        with network_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "entities"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "entities": "B"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "network_set_proximity_similarity.py"),
                "--edges",
                str(network_edges),
                "--train",
                str(network_train),
                "--eval",
                str(network_eval),
                "--out",
                str(network_out),
                "--scale",
                "1.0",
            ],
            check=True,
        )
        with network_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreater(float(rows[0]["similarity"]), float(rows[1]["similarity"]))

        genotype_train = self.out_dir / "genotype_train.csv"
        genotype_eval = self.out_dir / "genotype_eval.csv"
        genotype_out = self.out_dir / "genotype_pairwise.csv"
        with genotype_train.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["train_id", "snp_0", "snp_1"])
            writer.writeheader()
            writer.writerows(
                [
                    {"train_id": "t0", "snp_0": "0", "snp_1": "1"},
                    {"train_id": "t1", "snp_0": "2", "snp_1": "2"},
                ]
            )
        with genotype_eval.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "snp_0", "snp_1"])
            writer.writeheader()
            writer.writerow({"sample_id": "e0", "snp_0": "0", "snp_1": "1"})
        subprocess.run(
            [
                sys.executable,
                str(examples / "genotype_ibs_similarity.py"),
                "--train",
                str(genotype_train),
                "--eval",
                str(genotype_eval),
                "--out",
                str(genotype_out),
                "--marker-cols-prefix",
                "snp_",
            ],
            check=True,
        )
        with genotype_out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(rows[0]["similarity"]), 1.0)
        self.assertLess(float(rows[1]["similarity"]), 1.0)


if __name__ == "__main__":
    unittest.main()
