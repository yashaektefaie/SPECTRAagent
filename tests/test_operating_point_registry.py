import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "spectrae" / "operating_point_registry.py"
    spec = importlib.util.spec_from_file_location("operating_point_registry_direct", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OperatingPointRegistryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_registry_loads_and_validates(self):
        mod = load_module()
        catalog = mod.list_operating_point_methods()
        self.assertGreaterEqual(catalog["count"], 28)
        ids = {item["id"] for item in catalog["methods"]}
        expected = {
            "random_iid_split_baseline",
            "group_k_fold_holdout",
            "leave_one_domain_out_benchmark",
            "temporal_forward_split",
            "leave_site_out_external_validation",
            "targeted_intended_use_validation",
            "spatial_block_buffered_cv",
            "molecular_scaffold_split",
            "molecular_fingerprint_cluster_split",
            "molecular_umap_cluster_split",
            "molecular_property_extreme_split",
            "bioactivity_step_forward_split",
            "drug_target_cold_start_split",
            "leave_assay_out_bioactivity_split",
            "chromosome_loco_holdout",
            "regulatory_cross_cell_type_assay_holdout",
            "contiguous_mutational_region_split",
            "biological_sequence_homology_cluster_split",
            "protein_family_remote_homology_holdout",
            "cross_species_taxon_holdout",
            "perturbation_unseen_perturbation_split",
            "perturbation_unseen_cell_type_condition_split",
            "perturbation_systematic_variation_confounder_holdout",
            "medical_imaging_scanner_site_holdout",
            "materials_leave_cluster_composition_split",
            "graph_covariate_concept_shift_split",
            "rna_structurally_dissimilar_split",
            "spatiotemporal_forecast_horizon_split",
        }
        self.assertTrue(expected.issubset(ids))

        validation = mod.validate_operating_point_registry()
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["count"], catalog["count"])

    def test_search_suggests_domain_relevant_methods(self):
        mod = load_module()
        cases = [
            (
                "small molecule virtual screening where random and scaffold splits are too easy",
                "evaluate unseen chemical neighborhoods",
                "small_molecule",
                "chemical cluster",
                "molecular_umap_cluster_split",
            ),
            (
                "regulatory DNA enhancer promoter prediction across chromosomes",
                "evaluate unseen chromosomes",
                "regulatory_dna",
                "chromosome",
                "chromosome_loco_holdout",
            ),
            (
                "nucleotide fitness prediction with random versus contiguous cross validation",
                "evaluate extrapolation to unseen mutational regions",
                "nucleotide_assay",
                "mutational region",
                "contiguous_mutational_region_split",
            ),
            (
                "single-cell perturbation response for unseen gene perturbations",
                "predict responses to perturbations absent from training",
                "single_cell",
                "perturbation identity",
                "perturbation_unseen_perturbation_split",
            ),
            (
                "clinical sepsis model evaluated across hospitals",
                "test external validation at unseen sites",
                "clinical_record",
                "site",
                "leave_site_out_external_validation",
            ),
            (
                "geospatial environmental model with spatial autocorrelation",
                "test new locations separated from train",
                "geospatial",
                "geographic distance",
                "spatial_block_buffered_cv",
            ),
            (
                "protein function classifier with homologous sequences",
                "test unseen homology clusters",
                "protein",
                "sequence identity",
                "biological_sequence_homology_cluster_split",
            ),
            (
                "materials property prediction over composition clusters",
                "test novel material families",
                "materials",
                "composition",
                "materials_leave_cluster_composition_split",
            ),
            (
                "medical imaging model across scanner vendors and acquisition protocols",
                "test unseen scanner domains",
                "medical_image",
                "scanner",
                "medical_imaging_scanner_site_holdout",
            ),
            (
                "graph neural network benchmark under covariate and concept shifts",
                "test graph distribution shifts",
                "graph",
                "concept shift",
                "graph_covariate_concept_shift_split",
            ),
        ]

        for dataset, deployment, data_type, axis, expected_id in cases:
            with self.subTest(expected_id=expected_id):
                result = mod.suggest_operating_point_methods(
                    dataset_description=dataset,
                    deployment_question=deployment,
                    data_type=data_type,
                    novelty_axis=axis,
                    top_k=5,
                )
                ids = [item["id"] for item in result["results"]]
                self.assertIn(expected_id, ids)

    def test_render_method(self):
        mod = load_module()
        rendered = mod.render_operating_point_method("chromosome_loco_holdout")
        self.assertEqual(rendered["mime_type"], "text/markdown")
        self.assertIn("Leave-One-Chromosome-Out", rendered["content"])
        self.assertIn("Quality Gates", rendered["content"])

    def test_cli_validate_and_suggest(self):
        validate = subprocess.run(
            [sys.executable, "-m", "spectrae.cli", "operating-points", "validate"],
            cwd=str(self.root),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(validate.stdout)
        self.assertTrue(payload["valid"])
        self.assertGreaterEqual(payload["count"], 28)

        suggest = subprocess.run(
            [
                sys.executable,
                "-m",
                "spectrae.cli",
                "operating-points",
                "suggest",
                "--dataset-description",
                "drug target interaction benchmark with unseen drugs and targets",
                "--deployment-question",
                "test cold start entity generalization",
                "--data-type",
                "drug_target_interaction",
                "--novelty-axis",
                "drug target identity",
                "--top-k",
                "3",
            ],
            cwd=str(self.root),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(suggest.stdout)
        ids = [item["id"] for item in payload["results"]]
        self.assertIn("drug_target_cold_start_split", ids)


if __name__ == "__main__":
    unittest.main()
