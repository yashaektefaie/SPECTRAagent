"""Compare strict Caduceus SPECTRA agent runs before/after exploration protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def summarize(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    report = load_json(report_path) if report_path.exists() else {}
    text = ""
    report_md = run_dir / "report.md"
    if report_md.exists():
        text = report_md.read_text(encoding="utf-8", errors="replace")
    lower_text = text.lower() + "\n" + json.dumps(report).lower()
    behavioral_terms = [
        "fresh probe",
        "fresh downstream",
        "fresh model",
        "trained",
        "retrained",
        "fit",
        "performance_by_overlap",
        "model_metrics",
        "mcc",
        "auroc",
    ]
    blocker_terms = [
        "blocker",
        "blocked",
        "could not",
        "missing",
        "unavailable",
        "infeasible",
    ]
    artifact_files = sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    )
    behavioral_artifacts = [
        "performance_by_overlap.csv",
        "retraining_manifest.csv",
        "split_stats.csv",
    ]
    exploration_artifacts = [
        "question_trace.json",
        "next_experiment.json",
        "blockers.json",
        "similarity_hypothesis_scores.json",
    ]
    return {
        "run_dir": str(run_dir),
        "has_report_json": report_path.exists(),
        "has_report_md": report_md.exists(),
        "mentions_behavioral_experiment": any(term in lower_text for term in behavioral_terms),
        "mentions_concrete_blocker": any(term in lower_text for term in blocker_terms),
        "mentions_next_experiment": "next experiment" in lower_text or "next_experiment" in lower_text,
        "mentions_pre_benchmark": "pre-benchmark" in lower_text or "prebenchmark" in lower_text,
        "has_behavioral_artifacts": all((run_dir / name).exists() for name in behavioral_artifacts),
        "has_exploration_artifacts": all((run_dir / name).exists() for name in exploration_artifacts),
        "performance_curve_rows": count_csv_rows(run_dir / "performance_by_overlap.csv"),
        "split_rows": count_csv_rows(run_dir / "split_stats.csv"),
        "artifact_files": artifact_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "old_run": summarize(args.old_run),
        "new_run": summarize(args.new_run),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
