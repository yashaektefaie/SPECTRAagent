import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "spectrae" / "memory_registry.py"
    spec = importlib.util.spec_from_file_location("memory_registry_direct", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryRegistryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_registry_loads_and_validates(self):
        mod = load_module()
        catalog = mod.list_memory_entries()
        self.assertGreaterEqual(catalog["count"], 4)
        ids = {item["id"] for item in catalog["entries"]}
        self.assertIn("caduceus_external_perturbational_memory", ids)
        self.assertIn("caduceus_strict_sequence_memory", ids)
        self.assertIn("boom_numeric_mini_audit_memory", ids)
        self.assertIn("cross_domain_agent_ablation_memory", ids)

        validation = mod.validate_memory_registry()
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["count"], catalog["count"])

    def test_search_suggests_relevant_prior_memory(self):
        mod = load_module()
        caduceus = mod.search_memory_entries(
            query="Caduceus pretrained DNA regulatory generalization external enhancer gene",
            domain="regulatory_dna",
            model_family="DNA foundation model",
            model_name="Caduceus",
            top_k=2,
        )
        caduceus_ids = [item["id"] for item in caduceus["results"]]
        self.assertIn("caduceus_external_perturbational_memory", caduceus_ids)

        boom = mod.search_memory_entries(
            query="molecular OOD property prediction chemical novelty BOOM",
            domain="molecular_ml",
            data_type="small molecule",
            top_k=3,
        )
        boom_ids = [item["id"] for item in boom["results"]]
        self.assertIn("boom_numeric_mini_audit_memory", boom_ids)

    def test_render_entry(self):
        mod = load_module()
        rendered = mod.render_memory_entry("caduceus_external_perturbational_memory")
        self.assertEqual(rendered["mime_type"], "text/markdown")
        self.assertIn("DC-TAP", rendered["content"])
        self.assertIn("Do Not Repeat", rendered["content"])

    def test_no_packaged_entry_contains_local_paths(self):
        forbidden = ["/ewsc/", "/home/unix/", "/tmp/", "spectra_caduceus_", "spectra_assets/"]
        entry_dir = self.root / "spectrae" / "spectra_memory" / "entries"
        for path in entry_dir.glob("*.json"):
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                with self.subTest(path=path.name, marker=marker):
                    self.assertNotIn(marker, text)

    def test_cli_validate_and_search(self):
        validate = subprocess.run(
            [sys.executable, "-m", "spectrae.cli", "memory", "validate"],
            cwd=str(self.root),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(validate.stdout)
        self.assertTrue(payload["valid"])
        self.assertGreaterEqual(payload["count"], 4)

        search = subprocess.run(
            [
                sys.executable,
                "-m",
                "spectrae.cli",
                "memory",
                "search",
                "--query",
                "Caduceus regulatory DNA perturbational generalization",
                "--domain",
                "regulatory_dna",
                "--top-k",
                "2",
            ],
            cwd=str(self.root),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(search.stdout)
        ids = [item["id"] for item in payload["results"]]
        self.assertIn("caduceus_external_perturbational_memory", ids)


if __name__ == "__main__":
    unittest.main()
