import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "spectrae" / "dataset_catalog_registry.py"
    spec = importlib.util.spec_from_file_location("dataset_catalog_registry_direct", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DatasetCatalogTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_catalog_loads_and_validates(self):
        mod = load_module()
        catalog = mod.list_dataset_entries()
        self.assertGreaterEqual(catalog["count"], 31)
        ids = {item["id"] for item in catalog["datasets"]}
        self.assertIn("dc_tap_final_table", ids)
        self.assertIn("crispr_comparison_epcrispr", ids)
        self.assertIn("encode_ccre_screen", ids)
        self.assertIn("boom_benchmark_datasets", ids)
        self.assertIn("wilds_camelyon17", ids)
        self.assertIn("proteingym_aws", ids)
        self.assertIn("matbench_materials", ids)
        self.assertIn("mimic_eicu_physionet", ids)
        self.assertIn("flip2_protein_fitness", ids)
        self.assertIn("eurocropsml_zenodo", ids)
        self.assertIn("woods_time_series", ids)
        self.assertIn("welqrate_drug_discovery", ids)

        validation = mod.validate_dataset_catalog()
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["count"], catalog["count"])

    def test_search_suggests_relevant_datasets(self):
        mod = load_module()
        regulatory = mod.search_dataset_entries(
            query="Caduceus regulatory DNA distal enhancer gene perturbational generalization",
            domain="regulatory_dna",
            model_family="DNA foundation model",
            top_k=4,
        )
        regulatory_ids = [item["id"] for item in regulatory["results"]]
        self.assertIn("dc_tap_final_table", regulatory_ids)
        self.assertIn("crispr_comparison_epcrispr", regulatory_ids)

        molecules = mod.search_dataset_entries(
            query="molecular OOD property prediction chemical novelty",
            domain="molecular_ml",
            data_type="SMILES",
            top_k=3,
        )
        molecule_ids = [item["id"] for item in molecules["results"]]
        self.assertIn("boom_benchmark_datasets", molecule_ids)

        proteins = mod.search_dataset_entries(
            query="protein mutation effect fitness assay family extrapolation",
            domain="protein_engineering",
            data_type="protein sequence",
            top_k=3,
        )
        protein_ids = [item["id"] for item in proteins["results"]]
        self.assertIn("proteingym_aws", protein_ids)

        imaging = mod.search_dataset_entries(
            query="hospital histopathology external validation stain shift",
            domain="medical_imaging",
            top_k=3,
        )
        imaging_ids = [item["id"] for item in imaging["results"]]
        self.assertIn("wilds_camelyon17", imaging_ids)

    def test_render_entry(self):
        mod = load_module()
        rendered = mod.render_dataset_entry("dc_tap_final_table")
        self.assertEqual(rendered["mime_type"], "text/markdown")
        self.assertIn("Access Instructions", rendered["content"])
        self.assertIn("Final DC-TAP", rendered["content"])

    def test_no_packaged_entry_contains_local_paths(self):
        forbidden = ["/ewsc/", "/home/unix/", "/tmp/", "spectra_caduceus_", "spectra_assets/"]
        entry_dir = self.root / "spectrae" / "dataset_catalog" / "entries"
        for path in entry_dir.glob("*.json"):
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                with self.subTest(path=path.name, marker=marker):
                    self.assertNotIn(marker, text)

    def test_cli_validate_and_search(self):
        validate = subprocess.run(
            [sys.executable, "-m", "spectrae.cli", "dataset-catalog", "validate"],
            cwd=str(self.root),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(validate.stdout)
        self.assertTrue(payload["valid"])
        self.assertGreaterEqual(payload["count"], 31)

        search = subprocess.run(
            [
                sys.executable,
                "-m",
                "spectrae.cli",
                "dataset-catalog",
                "search",
                "--query",
                "regulatory DNA enhancer gene perturbation",
                "--domain",
                "regulatory_dna",
                "--top-k",
                "4",
            ],
            cwd=str(self.root),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(search.stdout)
        ids = [item["id"] for item in payload["results"]]
        self.assertIn("dc_tap_final_table", ids)


if __name__ == "__main__":
    unittest.main()
