"""Native focused SPECTRA split construction for common molecular CSV inputs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from .similarity_computation_registry import (
    load_similarity_computation_strategy,
    suggest_similarity_computation_strategies,
)
from .similarity_registry import load_similarity_definition, suggest_similarity_definitions


class UnsupportedFocusedSplitDataset(ValueError):
    """Raised when the native focused split constructor cannot handle a dataset."""


@dataclass(frozen=True)
class FocusedSplitConfig:
    dataset_path: Path
    output_root: Path
    question: str
    dataset_description: str
    domain: str
    constraints: str
    smiles_col: str = "SMILES"
    label_col: str = "Y"
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    include_chirality: bool = True
    thresholds: tuple[float, ...] = (0.80, 0.70, 0.60, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20)
    target_test_fraction: float = 0.20
    min_test_fraction: float = 0.15
    max_test_fraction: float = 0.25
    random_state: int = 20260522
    full_pairwise_ordered_row_cap: int = 2_000_000
    graph_edge_row_cap: int = 2_000_000
    quantile_pair_cap: int = 2_000_000
    nearest_neighbor_k: int = 10


def run_focused_split_construction(
    *,
    dataset_path: str,
    output_root: str,
    question: str,
    dataset_description: str,
    domain: str,
    constraints: str,
) -> dict[str, Any]:
    """Construct molecular SPECTRA splits and write the standard split artifacts."""

    config = FocusedSplitConfig(
        dataset_path=Path(dataset_path).expanduser().resolve(),
        output_root=Path(output_root).expanduser().resolve(),
        question=question,
        dataset_description=dataset_description,
        domain=domain,
        constraints=constraints,
    )
    return FocusedMolecularSplitBuilder(config).run()


class FocusedMolecularSplitBuilder:
    def __init__(self, config: FocusedSplitConfig):
        self.config = config
        self.cache_dir = config.output_root / "cache"
        self.pairwise_dir = config.output_root / "pairwise_similarity"
        self.graph_dir = config.output_root / "property_similarity_graph"
        self.assignments_dir = config.output_root / "split_assignments"
        self.diagnostics_dir = config.output_root / "diagnostics"
        self.baseline_dir = config.output_root / "baseline_validation"

    def run(self) -> dict[str, Any]:
        self._ensure_dependencies()
        self._prepare_dirs()

        df = self._load_dataset()
        y = self._load_binary_labels(df)
        feature_df, fingerprints, fp_matrix = self._canonicalize_and_featurize(df)

        definition_selection, computation_selection = self._select_similarity_procedure(df)
        similarity = self._compute_similarity_matrix(fingerprints)
        self._write_pairwise_outputs(feature_df, similarity)

        split_summaries: list[dict[str, Any]] = []
        assignments: dict[str, pd.DataFrame] = {}
        graph_summaries: list[dict[str, Any]] = []

        for threshold in self.config.thresholds:
            split_id = self._split_id(threshold)
            component_labels, components, graph_summary = self._components_for_threshold(
                similarity=similarity,
                threshold=threshold,
                sample_ids=feature_df["sample_id"].tolist(),
            )
            graph_summaries.append(graph_summary)
            selected_components, status, test_n, min_test_n, max_test_n = self._select_test_components(
                components=components,
                n_samples=len(df),
            )
            summary, assignment_df = self._summarize_split(
                split_id=split_id,
                threshold=threshold,
                feature_df=feature_df,
                y=y,
                similarity=similarity,
                component_labels=component_labels,
                components=components,
                selected_components=selected_components,
                status=status,
                test_n=test_n,
                min_test_n=min_test_n,
                max_test_n=max_test_n,
            )
            split_summaries.append(summary)
            assignments[split_id] = assignment_df
            assignment_df.to_csv(self.assignments_dir / f"{split_id}_assignments.csv", index=False)

        graph_df = pd.DataFrame(graph_summaries)
        graph_df.to_csv(self.graph_dir / "threshold_graph_summary.csv", index=False)

        sweep_df = pd.DataFrame(split_summaries)
        sweep_df.to_csv(self.diagnostics_dir / "spectral_parameter_sweep.csv", index=False)
        train_test_df = self._train_test_similarity_table(sweep_df)
        train_test_df.to_csv(self.diagnostics_dir / "train_test_similarity.csv", index=False)
        self._write_label_balance_table(sweep_df)

        baseline_df, predictions_df = self._run_baseline(
            split_summaries=split_summaries,
            assignments=assignments,
            feature_df=feature_df,
            fp_matrix=fp_matrix,
            y=y,
        )
        baseline_df.to_csv(self.baseline_dir / "baseline_metrics.csv", index=False)
        predictions_df.to_csv(self.baseline_dir / "baseline_predictions.csv", index=False)

        recommendation, recommendation_reason = self._choose_recommended_split(
            sweep_df=sweep_df,
            baseline_df=baseline_df,
            overall_positive_rate=float(np.mean(y)),
        )
        validation = self._validation_summary(sweep_df=sweep_df, baseline_df=baseline_df)
        manifest = self._write_manifest(
            df=df,
            y=y,
            definition_selection=definition_selection,
            computation_selection=computation_selection,
            sweep_df=sweep_df,
            baseline_df=baseline_df,
            validation=validation,
            recommended_split_id=recommendation,
            recommendation_reason=recommendation_reason,
        )
        self._write_baseline_summary(baseline_df=baseline_df, validation=validation)
        self._write_report(
            manifest=manifest,
            sweep_df=sweep_df,
            baseline_df=baseline_df,
            validation=validation,
            recommendation_reason=recommendation_reason,
        )

        return {
            "status": "ok",
            "constructor": "native_focused_molecular_spectra",
            "dataset": self.config.dataset_path.name,
            "n_rows": int(len(df)),
            "valid_split_count": int((sweep_df["status"] == "valid").sum()),
            "recommended_split_id": recommendation,
            "train_test_similarity_decreases": validation["train_test_similarity_strictly_decreases"],
            "baseline_roc_auc_decreases_overall": validation["baseline_roc_auc_decreases_overall"],
            "output_root": str(self.config.output_root),
            "report": str(self.config.output_root / "split_construction_report.md"),
        }

    def _ensure_dependencies(self) -> None:
        try:
            import rdkit  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            raise UnsupportedFocusedSplitDataset("RDKit is required for native molecular split construction") from exc

    def _prepare_dirs(self) -> None:
        for directory in [
            self.cache_dir,
            self.pairwise_dir,
            self.graph_dir,
            self.assignments_dir,
            self.diagnostics_dir,
            self.baseline_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self) -> pd.DataFrame:
        if not self.config.dataset_path.exists():
            raise UnsupportedFocusedSplitDataset(f"Dataset path does not exist: {self.config.dataset_path}")
        if self.config.dataset_path.suffix.lower() != ".csv":
            raise UnsupportedFocusedSplitDataset("Native focused splits currently support CSV datasets only")
        df = pd.read_csv(self.config.dataset_path)
        if self.config.smiles_col not in df.columns:
            raise UnsupportedFocusedSplitDataset(
                f"Expected a {self.config.smiles_col!r} column for molecular split construction"
            )
        if self.config.label_col not in df.columns:
            raise UnsupportedFocusedSplitDataset(
                f"Expected a {self.config.label_col!r} label column for baseline validation"
            )
        if df.empty:
            raise UnsupportedFocusedSplitDataset("Dataset is empty")
        return df

    def _load_binary_labels(self, df: pd.DataFrame) -> np.ndarray:
        values = df[self.config.label_col]
        if values.isna().any():
            raise UnsupportedFocusedSplitDataset("Native focused splits require non-null binary labels")
        unique_values = list(pd.Series(values).drop_duplicates())
        if len(unique_values) != 2:
            raise UnsupportedFocusedSplitDataset(
                f"Native baseline validation requires exactly two classes; found {len(unique_values)}"
            )
        if set(unique_values) == {0, 1}:
            return values.to_numpy(dtype=int)
        ordered = sorted(unique_values)
        mapping = {ordered[0]: 0, ordered[1]: 1}
        return values.map(mapping).to_numpy(dtype=int)

    def _canonicalize_and_featurize(self, df: pd.DataFrame):
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.config.fingerprint_radius,
            fpSize=self.config.fingerprint_bits,
            includeChirality=self.config.include_chirality,
        )
        molecules = []
        fingerprints = []
        feature_rows = []
        invalid_rows = []
        sample_width = max(5, len(str(len(df))))

        for row_index, smiles_value in enumerate(df[self.config.smiles_col].astype(str)):
            smiles = smiles_value.strip()
            sample_id = f"mol_{row_index:0{sample_width}d}"
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_rows.append({"row_index": row_index, "sample_id": sample_id, "smiles": smiles})
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if not scaffold:
                scaffold = "acyclic_or_no_murcko_scaffold"
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((self.config.fingerprint_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            molecules.append(mol)
            fingerprints.append(fp)
            feature_rows.append(
                {
                    "sample_id": sample_id,
                    "row_index": row_index,
                    "smiles": smiles,
                    "canonical_smiles": canonical,
                    "murcko_scaffold": scaffold,
                    "morgan_radius": self.config.fingerprint_radius,
                    "morgan_bits": self.config.fingerprint_bits,
                    "morgan_include_chirality": self.config.include_chirality,
                    "fingerprint_on_bits": int(arr.sum()),
                }
            )

        if invalid_rows:
            self._write_json(self.diagnostics_dir / "invalid_smiles.json", {"invalid_rows": invalid_rows})
            raise UnsupportedFocusedSplitDataset(
                f"Found {len(invalid_rows)} invalid SMILES; see diagnostics/invalid_smiles.json"
            )

        fp_matrix = np.zeros((len(fingerprints), self.config.fingerprint_bits), dtype=np.uint8)
        for index, fp in enumerate(fingerprints):
            DataStructs.ConvertToNumpyArray(fp, fp_matrix[index])

        return pd.DataFrame(feature_rows), fingerprints, fp_matrix

    def _select_similarity_procedure(self, df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
        dataset_name = self.config.dataset_path.name
        description = (
            f"{dataset_name}: {len(df)} small-molecule rows represented by SMILES with a binary "
            f"{self.config.label_col} label. Need molecular structural similarity for prospective "
            "SPECTRA split construction."
        )
        task_description = (
            "Construct SPECTRA train/test splits from SMILES using dataset features only, then validate "
            "with a fixed simple baseline after split construction."
        )
        definition_suggest = suggest_similarity_definitions(
            dataset_description=description,
            task_description=task_description,
            data_type="molecular",
            required_inputs=[self.config.smiles_col],
            top_k=5,
        )
        definition_selected = load_similarity_definition("molecules_morgan_tanimoto")
        computation_suggest = suggest_similarity_computation_strategies(
            dataset_description=description,
            similarity_definition=definition_selected["definition"],
            data_shape=f"CSV table with {len(df)} SMILES rows and a binary label",
            data_size=str(len(df)),
            required_inputs=[self.config.smiles_col],
            top_k=5,
        )
        computation_selected = load_similarity_computation_strategy("exact_chunked_all_pairs")

        self._write_json(self.cache_dir / "registry_similarity_definition_suggest.json", definition_suggest)
        self._write_json(self.cache_dir / "registry_similarity_definition_selected.json", definition_selected)
        self._write_json(self.cache_dir / "registry_similarity_computation_suggest.json", computation_suggest)
        self._write_json(self.cache_dir / "registry_similarity_computation_selected.json", computation_selected)

        definition_selection = {
            "selected_id": definition_selected["id"],
            "selected_registry_entry": definition_selected,
            "registry_suggestion_path": str(self.cache_dir / "registry_similarity_definition_suggest.json"),
            "registry_suggestions_top_ids": [item["id"] for item in definition_suggest.get("results", [])],
            "custom_definition_used": False,
            "selection_rationale": (
                "The dataset contains small molecules with SMILES strings, so the registry-backed "
                "Morgan fingerprint Tanimoto definition directly matches the scientific unit and "
                "the desired train-neighbor similarity axis."
            ),
            "parameters": {
                "canonicalization": "RDKit canonical isomeric SMILES",
                "fingerprint": "Morgan",
                "radius": self.config.fingerprint_radius,
                "bits": self.config.fingerprint_bits,
                "include_chirality": self.config.include_chirality,
                "similarity": "Tanimoto",
                "axis": "max_train_morgan_tanimoto",
            },
        }
        computation_selection = {
            "selected_id": computation_selected["id"],
            "selected_registry_entry": computation_selected,
            "registry_suggestion_path": str(self.cache_dir / "registry_similarity_computation_suggest.json"),
            "registry_suggestions_top_ids": [item["id"] for item in computation_suggest.get("results", [])],
            "custom_strategy_used": False,
            "selection_rationale": (
                "Exact RDKit BulkTanimoto all-pairs computation is tractable for these TDC-sized "
                "CSV datasets and preserves the full threshold graph needed for SPECTRA sweeps. "
                "Large long-form pairwise CSVs are capped on disk, but split construction uses the "
                "exact in-memory similarity matrix."
            ),
            "parameters": {
                "n_molecules": int(len(df)),
                "ordered_pairs_excluding_self": int(len(df) * (len(df) - 1)),
                "retained_on_disk": "nearest-neighbor table, quantiles, threshold graph summaries, and capped full pairwise CSV",
            },
        }
        self._write_json(self.config.output_root / "similarity_definition_selection.json", definition_selection)
        self._write_json(self.config.output_root / "similarity_computation_selection.json", computation_selection)
        return definition_selection, computation_selection

    def _compute_similarity_matrix(self, fingerprints: list[Any]) -> np.ndarray:
        from rdkit import DataStructs

        n_samples = len(fingerprints)
        similarity = np.eye(n_samples, dtype=np.float32)
        for i, fp in enumerate(fingerprints):
            scores = DataStructs.BulkTanimotoSimilarity(fp, fingerprints[i + 1 :])
            if scores:
                similarity[i, i + 1 :] = scores
                similarity[i + 1 :, i] = scores
        return similarity

    def _write_pairwise_outputs(self, feature_df: pd.DataFrame, similarity: np.ndarray) -> None:
        feature_df.to_csv(self.pairwise_dir / "molecule_features.csv", index=False)
        self._write_nearest_neighbors(feature_df, similarity)
        self._write_pairwise_quantiles(similarity)

        n_samples = similarity.shape[0]
        ordered_rows = n_samples * (n_samples - 1)
        manifest = {
            "exact_similarity_computed": True,
            "matrix_shape": [int(n_samples), int(n_samples)],
            "similarity": "RDKit Morgan fingerprint Tanimoto",
            "full_ordered_pairwise_rows": int(ordered_rows),
            "full_pairwise_csv_cap": self.config.full_pairwise_ordered_row_cap,
            "full_pairwise_csv_written": ordered_rows <= self.config.full_pairwise_ordered_row_cap,
            "nearest_neighbors": str(self.pairwise_dir / f"nearest_neighbors_top{self.config.nearest_neighbor_k}.csv"),
            "quantiles": str(self.pairwise_dir / "pairwise_similarity_quantiles.csv"),
        }
        if ordered_rows <= self.config.full_pairwise_ordered_row_cap:
            self._write_full_pairwise_csv(feature_df, similarity)
            manifest["full_pairwise_csv"] = str(self.pairwise_dir / "pairwise_similarity.csv")
        else:
            manifest["full_pairwise_csv"] = None
            manifest["disk_retention_reason"] = (
                "Full long-form ordered pairwise output was capped to avoid very large CSV artifacts; "
                "the exact matrix was still used for graph construction and diagnostics."
            )
        self._write_json(self.pairwise_dir / "pairwise_similarity_manifest.json", manifest)

    def _write_full_pairwise_csv(self, feature_df: pd.DataFrame, similarity: np.ndarray) -> None:
        sample_ids = feature_df["sample_id"].tolist()
        with (self.pairwise_dir / "pairwise_similarity.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "train_id", "similarity"])
            for i, sample_id in enumerate(sample_ids):
                for j, train_id in enumerate(sample_ids):
                    if i != j:
                        writer.writerow([sample_id, train_id, f"{float(similarity[i, j]):.8f}"])

    def _write_nearest_neighbors(self, feature_df: pd.DataFrame, similarity: np.ndarray) -> None:
        sample_ids = feature_df["sample_id"].tolist()
        k = min(self.config.nearest_neighbor_k, len(sample_ids) - 1)
        with (self.pairwise_dir / f"nearest_neighbors_top{k}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "neighbor_id", "rank", "similarity"])
            for i, sample_id in enumerate(sample_ids):
                candidate_count = min(k + 1, len(sample_ids))
                candidate_indices = np.argpartition(similarity[i], -candidate_count)[-candidate_count:]
                candidate_indices = [int(j) for j in candidate_indices if int(j) != i]
                ranked = sorted(candidate_indices, key=lambda j: (-float(similarity[i, j]), sample_ids[j]))[:k]
                for rank, j in enumerate(ranked, start=1):
                    writer.writerow([sample_id, sample_ids[j], rank, f"{float(similarity[i, j]):.8f}"])

    def _write_pairwise_quantiles(self, similarity: np.ndarray) -> None:
        n_samples = similarity.shape[0]
        upper_pair_count = n_samples * (n_samples - 1) // 2
        quantile_points = np.array([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
        if upper_pair_count <= self.config.quantile_pair_cap:
            values = similarity[np.triu_indices(n_samples, k=1)]
            method = "exact_all_upper_triangle_pairs"
            sample_size = int(len(values))
        else:
            rng = np.random.default_rng(self.config.random_state)
            rows = rng.integers(0, n_samples, size=self.config.quantile_pair_cap)
            cols = rng.integers(0, n_samples - 1, size=self.config.quantile_pair_cap)
            cols = np.where(cols >= rows, cols + 1, cols)
            values = similarity[rows, cols]
            method = "deterministic_random_pair_sample"
            sample_size = int(self.config.quantile_pair_cap)
        pd.DataFrame(
            {
                "quantile": quantile_points,
                "tanimoto_similarity": np.quantile(values, quantile_points),
                "method": method,
                "sample_size": sample_size,
                "total_upper_pairs": int(upper_pair_count),
            }
        ).to_csv(self.pairwise_dir / "pairwise_similarity_quantiles.csv", index=False)

    def _components_for_threshold(
        self,
        *,
        similarity: np.ndarray,
        threshold: float,
        sample_ids: list[str],
    ) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
        split_id = self._split_id(threshold)
        graph = similarity >= threshold
        np.fill_diagonal(graph, False)
        edge_count = int(np.count_nonzero(np.triu(graph, k=1)))
        n_components, component_labels = connected_components(csr_matrix(graph), directed=False)
        components = self._component_table(similarity=similarity, component_labels=component_labels)
        components.to_csv(self.graph_dir / f"{split_id}_components.csv", index=False)
        self._write_threshold_edges_if_small(
            graph=graph,
            similarity=similarity,
            sample_ids=sample_ids,
            split_id=split_id,
            edge_count=edge_count,
        )
        return component_labels, components, {
            "split_id": split_id,
            "spectral_threshold": threshold,
            "edge_rule": "Morgan fingerprint Tanimoto >= threshold",
            "n_edges": edge_count,
            "n_components": int(n_components),
            "largest_component_size": int(components["component_size"].max()),
            "edges_csv_written": edge_count <= self.config.graph_edge_row_cap,
        }

    def _component_table(self, *, similarity: np.ndarray, component_labels: np.ndarray) -> pd.DataFrame:
        component_rows = []
        all_indices = np.arange(similarity.shape[0])
        for component_id in range(int(component_labels.max()) + 1):
            indices = np.where(component_labels == component_id)[0]
            outside_mask = np.ones(similarity.shape[0], dtype=bool)
            outside_mask[indices] = False
            outside_indices = all_indices[outside_mask]
            if len(outside_indices):
                outside_scores = similarity[np.ix_(indices, outside_indices)]
                external_max = float(outside_scores.max())
                external_mean = float(outside_scores.mean())
            else:
                external_max = math.nan
                external_mean = math.nan
            component_rows.append(
                {
                    "component_id": component_id,
                    "component_size": int(len(indices)),
                    "min_row_index": int(indices.min()),
                    "external_max_similarity": external_max,
                    "external_mean_similarity": external_mean,
                }
            )
        return pd.DataFrame(component_rows)

    def _write_threshold_edges_if_small(
        self,
        *,
        graph: np.ndarray,
        similarity: np.ndarray,
        sample_ids: list[str],
        split_id: str,
        edge_count: int,
    ) -> None:
        if edge_count > self.config.graph_edge_row_cap:
            self._write_json(
                self.graph_dir / f"{split_id}_edges_manifest.json",
                {
                    "edges_csv_written": False,
                    "n_edges": int(edge_count),
                    "edge_row_cap": self.config.graph_edge_row_cap,
                    "reason": "Threshold graph is too large for a compact edge CSV artifact.",
                },
            )
            return
        rows, cols = np.where(np.triu(graph, k=1))
        with (self.graph_dir / f"{split_id}_edges.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id_1", "sample_id_2", "similarity"])
            for row, col in zip(rows, cols):
                writer.writerow([sample_ids[int(row)], sample_ids[int(col)], f"{float(similarity[row, col]):.8f}"])

    def _select_test_components(
        self,
        *,
        components: pd.DataFrame,
        n_samples: int,
    ) -> tuple[set[int], str, int, int, int]:
        target_n = int(round(self.config.target_test_fraction * n_samples))
        min_n = int(math.ceil(self.config.min_test_fraction * n_samples))
        max_n = int(math.floor(self.config.max_test_fraction * n_samples))
        selected: set[int] = set()
        test_n = 0

        ordered = components.sort_values(
            by=["external_max_similarity", "external_mean_similarity", "component_size", "min_row_index"],
            ascending=[False, False, True, True],
            kind="mergesort",
            na_position="last",
        )
        for row in ordered.itertuples(index=False):
            component_id = int(row.component_id)
            component_size = int(row.component_size)
            if component_size > max_n:
                continue
            if test_n + component_size <= max_n:
                selected.add(component_id)
                test_n += component_size
            if test_n >= target_n:
                break

        if test_n < min_n:
            for row in ordered.itertuples(index=False):
                component_id = int(row.component_id)
                component_size = int(row.component_size)
                if component_id in selected:
                    continue
                if test_n + component_size <= max_n:
                    selected.add(component_id)
                    test_n += component_size
                if test_n >= min_n:
                    break

        status = "valid" if min_n <= test_n <= max_n else "invalid_test_size"
        return selected, status, test_n, min_n, max_n

    def _summarize_split(
        self,
        *,
        split_id: str,
        threshold: float,
        feature_df: pd.DataFrame,
        y: np.ndarray,
        similarity: np.ndarray,
        component_labels: np.ndarray,
        components: pd.DataFrame,
        selected_components: set[int],
        status: str,
        test_n: int,
        min_test_n: int,
        max_test_n: int,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        test_mask = np.isin(component_labels, list(selected_components))
        train_mask = ~test_mask
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        if len(train_indices) and len(test_indices):
            cross_scores = similarity[np.ix_(test_indices, train_indices)]
            per_test_max = cross_scores.max(axis=1)
            per_test_mean = cross_scores.mean(axis=1)
            violations = int((cross_scores >= threshold).sum())
        else:
            per_test_max = np.array([], dtype=float)
            per_test_mean = np.array([], dtype=float)
            violations = 0
            status = "invalid_empty_train_or_test"
        if violations:
            status = "invalid_threshold_violation"

        assignment_df = self._assignment_table(
            feature_df=feature_df,
            threshold=threshold,
            test_mask=test_mask,
            component_labels=component_labels,
            test_indices=test_indices,
            per_test_max=per_test_max,
            per_test_mean=per_test_mean,
        )
        self._write_per_test_similarity(split_id, threshold, assignment_df)

        train_labels = y[train_indices]
        test_labels = y[test_indices]
        train_scaffolds = set(feature_df.loc[train_indices, "murcko_scaffold"])
        test_scaffolds = set(feature_df.loc[test_indices, "murcko_scaffold"])
        shared_test_scaffolds = test_scaffolds & train_scaffolds
        if len(test_indices):
            overlap_fraction = float(feature_df.loc[test_indices, "murcko_scaffold"].isin(shared_test_scaffolds).mean())
        else:
            overlap_fraction = math.nan

        selected_component_sizes = components.loc[
            components["component_id"].isin(selected_components),
            "component_size",
        ]
        summary = {
            "split_id": split_id,
            "spectral_threshold": threshold,
            "status": status,
            "n_train": int(len(train_indices)),
            "n_test": int(len(test_indices)),
            "test_fraction": float(len(test_indices) / len(y)),
            "target_test_fraction": self.config.target_test_fraction,
            "min_test_n": int(min_test_n),
            "max_test_n": int(max_test_n),
            "n_components": int(components.shape[0]),
            "n_selected_components": int(len(selected_components)),
            "largest_component_size": int(components["component_size"].max()),
            "selected_component_size_max": int(selected_component_sizes.max()) if len(selected_component_sizes) else 0,
            "train_test_edges_at_or_above_threshold": violations,
            "max_train_similarity_mean": float(np.mean(per_test_max)) if len(per_test_max) else math.nan,
            "max_train_similarity_median": float(np.median(per_test_max)) if len(per_test_max) else math.nan,
            "max_train_similarity_p10": float(np.quantile(per_test_max, 0.10)) if len(per_test_max) else math.nan,
            "max_train_similarity_p90": float(np.quantile(per_test_max, 0.90)) if len(per_test_max) else math.nan,
            "max_train_similarity_max": float(np.max(per_test_max)) if len(per_test_max) else math.nan,
            "mean_train_similarity_mean": float(np.mean(per_test_mean)) if len(per_test_mean) else math.nan,
            "train_positive_rate": float(np.mean(train_labels)) if len(train_labels) else math.nan,
            "test_positive_rate": float(np.mean(test_labels)) if len(test_labels) else math.nan,
            "train_positive_count": int(np.sum(train_labels == 1)),
            "train_negative_count": int(np.sum(train_labels == 0)),
            "test_positive_count": int(np.sum(test_labels == 1)),
            "test_negative_count": int(np.sum(test_labels == 0)),
            "train_scaffold_count": int(len(train_scaffolds)),
            "test_scaffold_count": int(len(test_scaffolds)),
            "test_scaffold_overlap_fraction": overlap_fraction,
        }
        components_out = components.copy()
        components_out["selected_for_test"] = components_out["component_id"].isin(selected_components)
        components_out["split_id"] = split_id
        components_out["spectral_threshold"] = threshold
        components_out.to_csv(self.assignments_dir / f"{split_id}_components.csv", index=False)
        return summary, assignment_df

    def _assignment_table(
        self,
        *,
        feature_df: pd.DataFrame,
        threshold: float,
        test_mask: np.ndarray,
        component_labels: np.ndarray,
        test_indices: np.ndarray,
        per_test_max: np.ndarray,
        per_test_mean: np.ndarray,
    ) -> pd.DataFrame:
        max_by_index = {int(idx): float(per_test_max[pos]) for pos, idx in enumerate(test_indices)}
        mean_by_index = {int(idx): float(per_test_mean[pos]) for pos, idx in enumerate(test_indices)}
        rows = []
        for i, row in feature_df.iterrows():
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "row_index": int(row["row_index"]),
                    "smiles": row["smiles"],
                    "canonical_smiles": row["canonical_smiles"],
                    "murcko_scaffold": row["murcko_scaffold"],
                    "split": "test" if bool(test_mask[i]) else "train",
                    "spectral_threshold": threshold,
                    "component_id": int(component_labels[i]),
                    "max_train_morgan_tanimoto": max_by_index.get(i, math.nan),
                    "mean_train_morgan_tanimoto": mean_by_index.get(i, math.nan),
                }
            )
        return pd.DataFrame(rows)

    def _write_per_test_similarity(self, split_id: str, threshold: float, assignment_df: pd.DataFrame) -> None:
        per_test = assignment_df.loc[
            assignment_df["split"] == "test",
            ["sample_id", "row_index", "max_train_morgan_tanimoto", "mean_train_morgan_tanimoto"],
        ].copy()
        per_test.insert(0, "spectral_threshold", threshold)
        per_test.insert(0, "split_id", split_id)
        per_test.to_csv(self.diagnostics_dir / f"{split_id}_per_test_similarity.csv", index=False)

    def _train_test_similarity_table(self, sweep_df: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "split_id",
            "spectral_threshold",
            "status",
            "n_train",
            "n_test",
            "max_train_similarity_mean",
            "max_train_similarity_median",
            "max_train_similarity_p10",
            "max_train_similarity_p90",
            "max_train_similarity_max",
            "train_test_edges_at_or_above_threshold",
        ]
        return sweep_df[columns].copy()

    def _write_label_balance_table(self, sweep_df: pd.DataFrame) -> None:
        columns = [
            "split_id",
            "spectral_threshold",
            "status",
            "n_train",
            "n_test",
            "train_positive_rate",
            "test_positive_rate",
            "train_positive_count",
            "train_negative_count",
            "test_positive_count",
            "test_negative_count",
        ]
        sweep_df[columns].to_csv(self.diagnostics_dir / "label_balance_by_split.csv", index=False)

    def _run_baseline(
        self,
        *,
        split_summaries: list[dict[str, Any]],
        assignments: dict[str, pd.DataFrame],
        feature_df: pd.DataFrame,
        fp_matrix: np.ndarray,
        y: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        metric_rows = []
        prediction_rows = []
        for summary in split_summaries:
            split_id = str(summary["split_id"])
            if summary["status"] != "valid":
                metric_rows.append(
                    {
                        "split_id": split_id,
                        "spectral_threshold": summary["spectral_threshold"],
                        "status": summary["status"],
                    }
                )
                continue
            assignment_df = assignments[split_id]
            train_indices = assignment_df.index[assignment_df["split"] == "train"].to_numpy()
            test_indices = assignment_df.index[assignment_df["split"] == "test"].to_numpy()
            y_train = y[train_indices]
            y_test = y[test_indices]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                metric_rows.append(
                    {
                        "split_id": split_id,
                        "spectral_threshold": summary["spectral_threshold"],
                        "status": "skipped_single_class_train_or_test",
                        "n_train": int(len(train_indices)),
                        "n_test": int(len(test_indices)),
                        "max_train_similarity_mean": summary["max_train_similarity_mean"],
                    }
                )
                continue

            model = LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=self.config.random_state,
                solver="liblinear",
            )
            model.fit(fp_matrix[train_indices], y_train)
            y_score = model.predict_proba(fp_matrix[test_indices])[:, 1]
            y_pred = (y_score >= 0.5).astype(int)
            metric_rows.append(
                {
                    "split_id": split_id,
                    "spectral_threshold": summary["spectral_threshold"],
                    "status": "valid",
                    "n_train": int(len(train_indices)),
                    "n_test": int(len(test_indices)),
                    "test_positive_rate": summary["test_positive_rate"],
                    "max_train_similarity_mean": summary["max_train_similarity_mean"],
                    "max_train_similarity_median": summary["max_train_similarity_median"],
                    "roc_auc": float(roc_auc_score(y_test, y_score)),
                    "average_precision": float(average_precision_score(y_test, y_score)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                    "mcc": float(matthews_corrcoef(y_test, y_pred)),
                    "brier": float(brier_score_loss(y_test, y_score)),
                }
            )
            for local_pos, sample_index in enumerate(test_indices):
                prediction_rows.append(
                    {
                        "split_id": split_id,
                        "sample_id": feature_df.loc[sample_index, "sample_id"],
                        "row_index": int(feature_df.loc[sample_index, "row_index"]),
                        "y_true": int(y_test[local_pos]),
                        "y_score": float(y_score[local_pos]),
                        "y_pred": int(y_pred[local_pos]),
                        "max_train_morgan_tanimoto": float(
                            assignment_df.loc[sample_index, "max_train_morgan_tanimoto"]
                        ),
                    }
                )
        return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)

    def _choose_recommended_split(
        self,
        *,
        sweep_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        overall_positive_rate: float,
    ) -> tuple[str | None, str]:
        valid = sweep_df[sweep_df["status"] == "valid"].copy()
        if valid.empty:
            return None, "No valid threshold-component split was generated."

        baseline_valid = baseline_df[baseline_df["status"] == "valid"].copy()
        if not baseline_valid.empty:
            merged = valid.merge(baseline_valid[["split_id", "roc_auc"]], on="split_id", how="left")
            easiest = merged.sort_values("max_train_similarity_mean", ascending=False).iloc[0]
            auc_floor = float(easiest["roc_auc"]) - 0.15 if not pd.isna(easiest["roc_auc"]) else math.nan
            candidates = merged[
                (merged["roc_auc"].notna())
                & (merged["roc_auc"] <= auc_floor)
                & ((merged["test_positive_rate"] - overall_positive_rate).abs() <= 0.10)
            ]
            if not candidates.empty:
                picked = candidates.sort_values(
                    by=["max_train_similarity_mean", "test_fraction"],
                    ascending=[True, True],
                ).iloc[0]
                return (
                    str(picked["split_id"]),
                    "Lowest train-test similarity among splits with at least 0.15 AUROC drop from the easiest level and label prevalence within 0.10 of the full dataset.",
                )

        picked = valid.sort_values("max_train_similarity_mean", ascending=True).iloc[0]
        return (
            str(picked["split_id"]),
            "Lowest train-test similarity valid split; baseline trend did not satisfy the stricter degradation rule.",
        )

    def _validation_summary(self, *, sweep_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict[str, Any]:
        valid_sweep = sweep_df[sweep_df["status"] == "valid"].copy()
        similarity_values = valid_sweep["max_train_similarity_mean"].dropna().tolist()
        similarity_decreases = all(
            similarity_values[i] > similarity_values[i + 1] for i in range(len(similarity_values) - 1)
        )
        baseline_valid = baseline_df[baseline_df["status"] == "valid"].copy()
        if not baseline_valid.empty:
            first_auc = float(baseline_valid.iloc[0]["roc_auc"])
            last_auc = float(baseline_valid.iloc[-1]["roc_auc"])
            first_ap = float(baseline_valid.iloc[0]["average_precision"])
            last_ap = float(baseline_valid.iloc[-1]["average_precision"])
            auc_decreases = last_auc < first_auc
            ap_decreases = last_ap < first_ap
        else:
            first_auc = last_auc = first_ap = last_ap = math.nan
            auc_decreases = ap_decreases = False
        return {
            "valid_split_count": int((sweep_df["status"] == "valid").sum()),
            "train_test_similarity_strictly_decreases": bool(similarity_decreases),
            "baseline_roc_auc_decreases_overall": bool(auc_decreases),
            "baseline_average_precision_decreases_overall": bool(ap_decreases),
            "easiest_valid_split_roc_auc": first_auc,
            "hardest_valid_split_roc_auc": last_auc,
            "easiest_valid_split_average_precision": first_ap,
            "hardest_valid_split_average_precision": last_ap,
        }

    def _write_manifest(
        self,
        *,
        df: pd.DataFrame,
        y: np.ndarray,
        definition_selection: dict[str, Any],
        computation_selection: dict[str, Any],
        sweep_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        validation: dict[str, Any],
        recommended_split_id: str | None,
        recommendation_reason: str,
    ) -> dict[str, Any]:
        manifest = {
            "mode": "spectra_split_construction",
            "run_type": "focused_spectra_split_construction",
            "constructor": "native_focused_molecular_spectra",
            "status": "completed",
            "uses_general_audit_loop": False,
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "question": self.config.question,
            "dataset_description": self.config.dataset_description,
            "domain": self.config.domain,
            "constraints": self.config.constraints,
            "output_root": str(self.config.output_root),
            "dataset": {
                "path": str(self.config.dataset_path),
                "n_rows": int(len(df)),
                "columns": list(df.columns),
                "row_id": "generated sample_id mol_00000..",
                "scientific_unit_column": self.config.smiles_col,
                "label_column": self.config.label_col,
                "task_type": "binary_classification",
                "n_positive": int(np.sum(y == 1)),
                "n_negative": int(np.sum(y == 0)),
                "positive_rate": float(np.mean(y)),
                "invalid_smiles": 0,
            },
            "similarity_definition": {
                "selected_id": definition_selection["selected_id"],
                "selection_file": str(self.config.output_root / "similarity_definition_selection.json"),
            },
            "similarity_computation": {
                "selected_id": computation_selection["selected_id"],
                "selection_file": str(self.config.output_root / "similarity_computation_selection.json"),
            },
            "split_construction": {
                "method": "thresholded Morgan-Tanimoto graph connected components",
                "edge_rule": "Morgan fingerprint Tanimoto >= spectral_threshold",
                "membership_features": ["canonical_smiles", "Morgan fingerprint Tanimoto graph"],
                "labels_used_for_membership": False,
                "target_test_fraction": self.config.target_test_fraction,
                "thresholds": list(self.config.thresholds),
                "selection_rule": (
                    "For each threshold, keep connected components intact and select boundary-near "
                    "components deterministically until the test set is near 20% of rows."
                ),
            },
            "validation": validation,
            "recommended_split_id": recommended_split_id,
            "recommendation_reason": recommendation_reason,
            "outputs": {
                "pairwise_similarity_dir": str(self.pairwise_dir),
                "property_similarity_graph_dir": str(self.graph_dir),
                "split_assignments_dir": str(self.assignments_dir),
                "diagnostics_dir": str(self.diagnostics_dir),
                "baseline_validation_dir": str(self.baseline_dir),
                "baseline_metrics": str(self.baseline_dir / "baseline_metrics.csv"),
                "spectral_parameter_sweep": str(self.diagnostics_dir / "spectral_parameter_sweep.csv"),
                "report": str(self.config.output_root / "split_construction_report.md"),
            },
        }
        self._write_json(self.config.output_root / "split_construction_manifest.json", manifest)
        return manifest

    def _write_baseline_summary(self, *, baseline_df: pd.DataFrame, validation: dict[str, Any]) -> None:
        self._write_json(
            self.baseline_dir / "baseline_validation_summary.json",
            {
                "baseline_model": "LogisticRegression(class_weight='balanced', solver='liblinear')",
                "features": (
                    f"Morgan fingerprints radius={self.config.fingerprint_radius}, "
                    f"bits={self.config.fingerprint_bits}, include_chirality={self.config.include_chirality}"
                ),
                "random_state": self.config.random_state,
                "valid_metric_rows": int((baseline_df["status"] == "valid").sum()) if not baseline_df.empty else 0,
                **validation,
                "metrics_path": str(self.baseline_dir / "baseline_metrics.csv"),
                "predictions_path": str(self.baseline_dir / "baseline_predictions.csv"),
            },
        )

    def _write_report(
        self,
        *,
        manifest: dict[str, Any],
        sweep_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        validation: dict[str, Any],
        recommendation_reason: str,
    ) -> None:
        table_cols = [
            "split_id",
            "spectral_threshold",
            "status",
            "n_train",
            "n_test",
            "max_train_similarity_mean",
            "max_train_similarity_median",
            "max_train_similarity_max",
            "test_positive_rate",
        ]
        baseline_cols = [
            "split_id",
            "status",
            "roc_auc",
            "average_precision",
            "balanced_accuracy",
            "accuracy",
        ]
        sweep_md = self._markdown_table(sweep_df[table_cols], floatfmt=".3f")
        baseline_md = self._markdown_table(baseline_df[baseline_cols], floatfmt=".3f") if not baseline_df.empty else ""
        n_rows = manifest["dataset"]["n_rows"]
        report = f"""# Focused SPECTRA Split Construction: {self.config.dataset_path.stem}

## Scope

This run constructs SPECTRA splits only. It uses the supplied CSV as the starting dataset and does not use target-model errors or labels to define split membership.

## Dataset

- Source: `{self.config.dataset_path}`
- Rows: {n_rows}
- Input column: `{self.config.smiles_col}`
- Label column: `{self.config.label_col}` used only after split construction for diagnostics and baseline validation
- Task type: binary molecular property classification
- Positive rate: {manifest["dataset"]["positive_rate"]:.3f}

## Similarity Definition

- Selected definition: `{manifest["similarity_definition"]["selected_id"]}`
- Scientific unit: molecule
- Axis: maximum train-neighbor Morgan fingerprint Tanimoto similarity
- Parameters: radius {self.config.fingerprint_radius}, {self.config.fingerprint_bits} bits, include chirality `{self.config.include_chirality}`
- Registry evidence and selection details: `similarity_definition_selection.json`

## Computation Strategy

- Selected computation: `{manifest["similarity_computation"]["selected_id"]}`
- Exact RDKit all-pairs Tanimoto was used for split construction. Very large long-form pairwise CSVs are capped on disk; nearest-neighbor tables, quantiles, graph summaries, assignments, and diagnostics are always written.
- Selection details: `similarity_computation_selection.json`

## Split Construction

For each threshold, a graph was built with edges where Tanimoto similarity is greater than or equal to the threshold. Connected components were assigned wholly to either train or test, which guarantees that no train/test pair remains at or above that threshold. Test components were selected deterministically from molecular similarity statistics only, preferring components closest to the current threshold boundary.

## Spectral Sweep

Train-test similarity strictly decreases across valid levels: `{validation["train_test_similarity_strictly_decreases"]}`.

{sweep_md}

## Baseline Validation

Baseline: fixed logistic regression with class-balanced loss on radius-2 2048-bit Morgan fingerprints. Labels were used only for model fitting/evaluation after split membership was fixed.

Overall AUROC decreases from the easiest to hardest validated spectral level: `{validation["baseline_roc_auc_decreases_overall"]}`.

{baseline_md}

## Recommendation

Recommended split: `{manifest["recommended_split_id"]}`.

Reason: {recommendation_reason}

## Artifacts

- Manifest: `split_construction_manifest.json`
- Similarity definition: `similarity_definition_selection.json`
- Similarity computation: `similarity_computation_selection.json`
- Pairwise artifacts: `pairwise_similarity/`
- Property graph artifacts: `property_similarity_graph/`
- Split assignments: `split_assignments/*_assignments.csv`
- Similarity diagnostics: `diagnostics/train_test_similarity.csv`
- Spectral sweep: `diagnostics/spectral_parameter_sweep.csv`
- Baseline metrics: `baseline_validation/baseline_metrics.csv`
- Baseline predictions: `baseline_validation/baseline_predictions.csv`

## Blockers

No blocking data or dependency issues were encountered. RDKit parsed all SMILES.
"""
        (self.config.output_root / "split_construction_report.md").write_text(report, encoding="utf-8")

    def _markdown_table(self, df: pd.DataFrame, *, floatfmt: str) -> str:
        try:
            return df.to_markdown(index=False, floatfmt=floatfmt)
        except Exception:
            return "```\n" + df.to_csv(index=False) + "```"

    def _split_id(self, threshold: float) -> str:
        return f"spectra_tau_{threshold:.2f}".replace(".", "p")

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
