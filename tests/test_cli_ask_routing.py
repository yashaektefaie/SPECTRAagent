import argparse
import contextlib
import io
import json
import shutil
import unittest
from pathlib import Path

from spectrae.cli import _infer_ask_mode, main


class AskRoutingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/ewsc/yektefai/spectra_ask_routing_tests")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset = self.root / "toy.csv"
        self.dataset.write_text("SMILES,Y\nCCO,0\nCCN,1\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _args(self, **overrides):
        values = {
            "ask_mode": "auto",
            "constraints": "",
            "dataset_description": "",
            "domain": "auto",
            "model_artifact": "",
            "model_paper": "",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_auto_routes_split_construction_questions_to_splits_mode(self):
        mode = _infer_ask_mode(
            self._args(),
            "Construct SPECTRA splits for this molecular dataset.",
        )

        self.assertEqual(mode, "splits")

    def test_auto_routes_model_generalizability_questions_to_audit_mode(self):
        mode = _infer_ask_mode(
            self._args(model_artifact="checkpoint.pt", model_paper="paper.pdf"),
            "Audit this model's generalizability from the paper and checkpoint.",
        )

        self.assertEqual(mode, "audit")

    def test_split_mode_dry_run_does_not_prepare_general_agent_session(self):
        out = self.root / "split_mode"
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            status = main(
                [
                    "ask",
                    "construct SPECTRA splits for this dataset",
                    "--dataset",
                    str(self.dataset),
                    "--out",
                    str(out),
                    "--dry-run",
                ]
            )

        self.assertEqual(status, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["ask_mode"], "splits")
        self.assertFalse(payload["uses_general_audit_loop"])
        self.assertTrue((out / "split_construction_prompt.md").exists())
        self.assertTrue((out / "split_construction_manifest.json").exists())
        self.assertFalse((out / "session_state.json").exists())
        self.assertFalse((out / "role_prompts").exists())

    def test_split_mode_prompt_encodes_canonical_spectra_procedure(self):
        out = self.root / "split_procedure"
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            status = main(
                [
                    "ask",
                    "generate SPECTRA splits for this dataset",
                    "--dataset",
                    str(self.dataset),
                    "--out",
                    str(out),
                    "--dry-run",
                ]
            )

        self.assertEqual(status, 0)
        prompt = (out / "split_construction_prompt.md").read_text(encoding="utf-8")
        self.assertIn("Canonical SPECTRA split-construction procedure", prompt)
        self.assertIn("spectra similarity-definitions suggest", prompt)
        self.assertIn("spectra similarity-computation suggest", prompt)
        self.assertIn("Compute pairwise similarities", prompt)
        self.assertIn("Verify that train-test similarity decreases", prompt)
        self.assertIn("performance decreases as train-test similarity decreases", prompt)
        self.assertIn("similarity_definition_selection.json", prompt)
        self.assertIn("similarity_computation_selection.json", prompt)


if __name__ == "__main__":
    unittest.main()
