#!/usr/bin/env python3
"""Compare fresh vanilla vs SPECTRA agent generalization runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPORT_NAMES = ("report.md", "final_report.md", "README.md", "report.json", "summary.json")


def read_texts(run_dir: Path) -> str:
    chunks: list[str] = []
    for name in REPORT_NAMES:
        path = run_dir / name
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    for path in sorted(run_dir.glob("*.json")):
        if path.name not in REPORT_NAMES:
            chunks.append(path.read_text(encoding="utf-8", errors="replace")[:10000])
    return "\n".join(chunks)


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def list_files(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    ignored_parts = {
        "__pycache__",
        "caduceus_probe_venv",
        "embeddings",
    }
    files = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir)
        if any(part in ignored_parts for part in rel.parts):
            continue
        files.append(str(rel))
    return sorted(files)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    text = read_texts(run_dir)
    lower = text.lower()
    files = list_files(run_dir)
    file_set = set(files)
    expected_spectra_artifacts = {
        "question_trace.json",
        "mode_decision.json",
        "next_experiment.json",
        "blockers.json",
        "split_stats.csv",
        "performance_by_overlap.csv",
        "retraining_manifest.csv",
    }
    target_model_terms = [
        "caduceus",
        "frozen caduceus",
        "caduceus embedding",
        "automodel",
        "checkpoint",
        "model.safetensors",
    ]
    behavioral_terms = [
        "performance_by_overlap",
        "fresh probe",
        "fresh logistic",
        "fine-tun",
        "trained",
        "retrained",
        "mcc",
        "accuracy",
        "f1",
    ]
    recovery_terms = [
        "micromamba",
        "environment",
        "torch",
        "transformers",
        "dependency",
        "recovery",
    ]
    similarity_terms = [
        "similarity",
        "overlap",
        "duplicate",
        "k-mer",
        "kmer",
        "nearest",
        "spectral",
    ]
    return {
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "n_files": len(files),
        "files": files[:200],
        "files_truncated": len(files) > 200,
        "has_expected_spectra_artifact_set": expected_spectra_artifacts.issubset(file_set),
        "expected_spectra_artifacts_present": sorted(expected_spectra_artifacts.intersection(file_set)),
        "performance_by_overlap_rows": count_csv_rows(run_dir / "performance_by_overlap.csv"),
        "split_stats_rows": count_csv_rows(run_dir / "split_stats.csv"),
        "mentions_target_model": any(term in lower for term in target_model_terms),
        "mentions_behavioral_experiment": any(term in lower for term in behavioral_terms),
        "mentions_environment_recovery": any(term in lower for term in recovery_terms),
        "mentions_similarity_or_overlap": any(term in lower for term in similarity_terms),
        "mentions_spectra": "spectra" in lower,
        "mentions_caduceus_direct_probe": (
            "frozen caduceus" in lower
            or "caduceus embedding" in lower
            or "caduceus probe" in lower
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-run", type=Path, required=True)
    parser.add_argument("--spectra-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "vanilla": summarize_run(args.vanilla_run),
        "spectra": summarize_run(args.spectra_run),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
