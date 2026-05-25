import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "spectrae" / "spectra_benchmarks.py"
    spec = importlib.util.spec_from_file_location("spectra_benchmarks_direct", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpectraBenchmarkTests(unittest.TestCase):
    def test_capsules_load(self):
        mod = load_module()
        capsules = mod.list_benchmark_capsules()
        self.assertGreaterEqual(capsules["count"], 6)
        self.assertIn("boom", {item["id"] for item in capsules["capsules"]})

    def test_audit_card_template_and_score(self):
        mod = load_module()
        card = mod.create_audit_card_template("boom")
        self.assertEqual(card["capsule_id"], "boom")
        validation = mod.validate_audit_card(card)
        self.assertFalse(validation["valid"])
        self.assertIn("property_graphs", validation["empty_fields"])

    def test_auspc(self):
        mod = load_module()
        result = mod.compute_auspc(
            [
                {"cross_split_overlap": 0.9, "performance": 0.85},
                {"cross_split_overlap": 0.5, "performance": 0.72},
                {"cross_split_overlap": 0.1, "performance": 0.61},
            ]
        )
        self.assertEqual(result["point_count"], 3)
        self.assertGreater(result["auspc"], 0)

    def test_fetch_plan_is_safe_by_default(self):
        mod = load_module()
        plan = mod.fetch_benchmark_assets(["boom"], dry_run=True)
        self.assertTrue(plan["dry_run"])
        self.assertTrue(
            all(asset["kind"] in {"paper_pdf", "paper_page"} for asset in plan["assets"])
        )


if __name__ == "__main__":
    unittest.main()
