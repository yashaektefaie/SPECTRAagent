import csv
import json
import shutil
import unittest
from pathlib import Path


class SpectraAuditTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.out_dir = self.root / "build" / "test_spectra_audit"
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True)
        self.train_path = self.out_dir / "train.csv"
        self.eval_path = self.out_dir / "eval_predictions.csv"
        with self.train_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["smiles", "y"])
            writer.writeheader()
            writer.writerows(
                [
                    {"smiles": "C", "y": 1.0},
                    {"smiles": "CC", "y": 2.0},
                    {"smiles": "CCC", "y": 3.0},
                    {"smiles": "O", "y": 0.5},
                ]
            )
        with self.eval_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "smiles", "y_true", "y_pred"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"sample_id": "e0", "smiles": "C", "y_true": 1.0, "y_pred": 1.1},
                    {"sample_id": "e1", "smiles": "CCCC", "y_true": 4.0, "y_pred": 2.8},
                    {"sample_id": "e2", "smiles": "N", "y_true": 0.2, "y_pred": 0.7},
                    {"sample_id": "e3", "smiles": "O", "y_true": 0.5, "y_pred": 0.55},
                ]
            )

    def test_molecule_audit_writes_standard_artifacts(self):
        from spectrae.audit import MoleculeAuditConfig, run_molecule_audit

        artifact_dir = self.out_dir / "artifacts"
        result = run_molecule_audit(
            MoleculeAuditConfig(
                train_path=str(self.train_path),
                eval_path=str(self.eval_path),
                output_dir=str(artifact_dir),
                sample_id_col="sample_id",
                thresholds=(1.0, 0.8, 0.5, 0.2),
                prefer_rdkit=False,
            )
        )
        self.assertEqual(result["status"], "completed")
        expected = {
            "audit_card",
            "split_stats",
            "performance_by_distance",
            "eval_with_distance",
            "spectral_curve",
            "report",
        }
        self.assertTrue(expected.issubset(result["artifacts"]))
        for path in result["artifacts"].values():
            self.assertTrue(Path(path).exists(), path)

        with Path(result["artifacts"]["audit_card"]).open(encoding="utf-8") as handle:
            card = json.load(handle)
        self.assertEqual(card["domain"], "molecules")
        self.assertEqual(card["scientific_unit"], "molecule")
        self.assertIn("performance_distance_curve", card)
        self.assertTrue(card["performance_distance_curve"])

    def test_pairwise_similarity_audit_is_domain_agnostic(self):
        from spectrae.audit import PairwiseSimilarityAuditConfig, run_pairwise_similarity_audit

        sim_path = self.out_dir / "pairwise_similarity.csv"
        with sim_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "train_id", "similarity"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"sample_id": "e0", "train_id": "t0", "similarity": 1.0},
                    {"sample_id": "e0", "train_id": "t1", "similarity": 0.5},
                    {"sample_id": "e1", "train_id": "t0", "similarity": 0.6},
                    {"sample_id": "e1", "train_id": "t1", "similarity": 0.7},
                    {"sample_id": "e2", "train_id": "t0", "similarity": 0.1},
                    {"sample_id": "e2", "train_id": "t1", "similarity": 0.2},
                    {"sample_id": "e3", "train_id": "t0", "similarity": 0.8},
                    {"sample_id": "e3", "train_id": "t1", "similarity": 0.4},
                ]
            )

        artifact_dir = self.out_dir / "pairwise_artifacts"
        result = run_pairwise_similarity_audit(
            PairwiseSimilarityAuditConfig(
                eval_path=str(self.eval_path),
                similarity_path=str(sim_path),
                output_dir=str(artifact_dir),
                domain="sequences",
                scientific_unit="sequence",
                thresholds=(0.9, 0.6, 0.3),
            )
        )
        self.assertEqual(result["status"], "completed")
        self.assertTrue(Path(result["artifacts"]["performance_by_axis"]).exists())
        with Path(result["artifacts"]["audit_card"]).open(encoding="utf-8") as handle:
            card = json.load(handle)
        self.assertEqual(card["domain"], "sequences")
        self.assertEqual(card["scientific_unit"], "sequence")
        self.assertEqual(card["spectral_axis"]["source"], "pairwise_train_eval_similarity_graph")
        self.assertEqual(card["property_graphs"][0]["edge_count"], 8)


if __name__ == "__main__":
    unittest.main()
