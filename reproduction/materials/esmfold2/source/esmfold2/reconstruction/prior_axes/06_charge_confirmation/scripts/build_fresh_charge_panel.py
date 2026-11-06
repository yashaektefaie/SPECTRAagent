#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
POOL = Path(
    "/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run/"
    "external_rcsb_same_bucket_expansion_20260531/rcsb_external_candidate_pool.csv"
)
PRIOR_TABLES = [
    Path("/ewsc/yektefai/spectra_runs/esmfold2_shared_axis_cameo_rcsb_20260601T173500Z/spectra_ready_dataset/combined_prospective_performance.csv"),
    Path("/ewsc/yektefai/spectra_runs/esmfold2_external_flexible_axis_expanded_20260601T170049Z/spectra_ready_dataset/candidate_features.csv"),
    Path("/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run/external_rcsb_same_bucket_expansion_20260531/spectra_ready_dataset/candidate_features.csv"),
]

OUT_DIR = ROOT / "spectra_ready_dataset"
OUT_RAW = OUT_DIR / "raw_rcsb_charge_confirmation"
OUT_PDB = OUT_RAW / "experiment_structures"
OUT_ELIGIBLE = OUT_DIR / "eligible_unscored_charge_pool.csv"
OUT_PANEL = OUT_DIR / "candidate_features.csv"
OUT_MANIFEST = ROOT / "charge_confirmation_panel_manifest.json"

PDB_URL = "https://files.rcsb.org/download/{entry_id}.pdb"
CIF_URL = "https://files.rcsb.org/download/{entry_id}.cif"
MAX_STRUCTURE_BYTES = 50_000_000
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
BASIC = set("KRH")
ACIDIC = set("DE")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def calc_abs_charge_per_residue(seq: str) -> float:
    residues = [c for c in str(seq or "").upper() if c in AA20]
    if not residues:
        return float("nan")
    charge = sum(c in BASIC for c in residues) - sum(c in ACIDIC for c in residues)
    return abs(float(charge)) / float(len(residues))


def load_prior_exclusions() -> tuple[set[str], set[str], dict[str, Any]]:
    targets: set[str] = set()
    sequences: set[str] = set()
    tables: list[dict[str, Any]] = []
    for path in PRIOR_TABLES:
        if not path.exists():
            tables.append({"path": str(path), "exists": False})
            continue
        df = pd.read_csv(path, usecols=lambda c: c in {"target_id", "sequence"})
        before_t = len(targets)
        before_s = len(sequences)
        if "target_id" in df.columns:
            targets.update(df["target_id"].dropna().astype(str))
        if "sequence" in df.columns:
            sequences.update(df["sequence"].dropna().astype(str).str.upper())
        tables.append(
            {
                "path": str(path),
                "exists": True,
                "rows": int(len(df)),
                "new_targets": int(len(targets) - before_t),
                "new_sequences": int(len(sequences) - before_s),
            }
        )
    return targets, sequences, {"tables": tables, "n_prior_targets": len(targets), "n_prior_sequences": len(sequences)}


def download_text(url: str, out_path: Path, accepted_prefixes: tuple[str, ...]) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    try:
        resp = requests.get(url, timeout=60)
    except Exception:
        return False
    if resp.status_code != 200:
        return False
    text = resp.content.decode("utf-8", errors="replace")
    if not text.startswith(accepted_prefixes):
        return False
    out_path.write_text(text)
    time.sleep(0.03)
    return True


def download_structure(entry_id: str, target_id: str) -> tuple[bool, Path | None, str]:
    pdb_path = OUT_PDB / f"{target_id}.pdb"
    if download_text(PDB_URL.format(entry_id=entry_id), pdb_path, ("HEADER", "TITLE", "COMPND", "REMARK", "ATOM")):
        if pdb_path.stat().st_size > MAX_STRUCTURE_BYTES:
            return False, pdb_path, "pdb_too_large"
        return True, pdb_path, "pdb"
    cif_path = OUT_PDB / f"{target_id}.cif"
    if download_text(CIF_URL.format(entry_id=entry_id), cif_path, ("data_", "#")):
        if cif_path.stat().st_size > MAX_STRUCTURE_BYTES:
            return False, cif_path, "cif_too_large"
        return True, cif_path, "cif"
    return False, None, "failed"


def assign_charge_levels(df: pd.DataFrame, n_levels: int = 5) -> pd.DataFrame:
    out = df.copy()
    stress = pd.to_numeric(out["calc_seq_abs_charge_per_residue"], errors="coerce")
    mask = stress.notna()
    levels = pd.qcut(stress[mask].rank(method="first"), q=n_levels, labels=False, duplicates="drop")
    out.loc[mask, "charge_level"] = levels.astype(int)
    out["charge_level"] = out["charge_level"].astype("Int64")
    out["charge_axis_feature"] = "calc_seq_abs_charge_per_residue"
    out["charge_direction"] = "high"
    return out


def level_thresholds(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for level, part in df.groupby("charge_level", dropna=True):
        rows.append(
            {
                "level": int(level),
                "n_eligible": int(len(part)),
                "entry_count": int(part["entry_id"].nunique()),
                "charge_min": float(part["calc_seq_abs_charge_per_residue"].min()),
                "charge_mean": float(part["calc_seq_abs_charge_per_residue"].mean()),
                "charge_max": float(part["calc_seq_abs_charge_per_residue"].max()),
                "seq_len_median": float(part["seq_len"].median()),
                "resolution_median": float(pd.to_numeric(part["resolution"], errors="coerce").median()),
            }
        )
    return rows


def candidate_order(candidates: pd.DataFrame, desired_lengths: np.ndarray) -> list[int]:
    remaining = set(candidates.index.tolist())
    ordered: list[int] = []
    for desired in desired_lengths:
        if not remaining:
            break
        ranked = sorted(
            remaining,
            key=lambda idx: (
                abs(float(candidates.loc[idx, "seq_len"]) - float(desired)),
                float(candidates.loc[idx, "resolution"]) if pd.notna(candidates.loc[idx, "resolution"]) else 99.0,
                str(candidates.loc[idx, "target_id"]),
            ),
        )
        idx = ranked[0]
        ordered.append(idx)
        remaining.remove(idx)
    if remaining:
        ordered.extend(
            sorted(
                remaining,
                key=lambda idx: (
                    float(candidates.loc[idx, "seq_len"]),
                    float(candidates.loc[idx, "resolution"]) if pd.notna(candidates.loc[idx, "resolution"]) else 99.0,
                    str(candidates.loc[idx, "target_id"]),
                ),
            )
        )
    return ordered


def select_panel(eligible: pd.DataFrame, target_per_level: int = 20, max_per_entry: int = 2) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    OUT_PDB.mkdir(parents=True, exist_ok=True)
    selected_rows = []
    diagnostics: list[dict[str, Any]] = []
    entry_counts: dict[str, int] = {}
    global_lengths = np.quantile(eligible["seq_len"], np.linspace(0.05, 0.95, target_per_level))
    level_order = [4, 0, 1, 2, 3]

    for level in level_order:
        part = eligible[eligible["charge_level"].eq(level)].copy()
        ordered_indices = candidate_order(part, global_lengths)
        level_rows = []
        attempts = 0
        pdb_failures = 0
        structure_size_skips = 0
        entry_cap_skips = 0
        for idx in ordered_indices:
            if len(level_rows) >= target_per_level:
                break
            rec = part.loc[idx].copy()
            entry = str(rec["entry_id"])
            if entry_counts.get(entry, 0) >= max_per_entry:
                entry_cap_skips += 1
                continue
            attempts += 1
            ok, structure_path, structure_format = download_structure(entry, str(rec["target_id"]))
            if not ok or structure_path is None:
                if structure_format.endswith("_too_large"):
                    structure_size_skips += 1
                else:
                    pdb_failures += 1
                continue
            if structure_path.stat().st_size > MAX_STRUCTURE_BYTES:
                structure_size_skips += 1
                pdb_failures += 1
                continue
            entry_counts[entry] = entry_counts.get(entry, 0) + 1
            rec["pdb_download_ok"] = True
            rec["structure_format"] = structure_format
            rec["eligible_reference"] = True
            rec["ref_chain"] = rec["auth_asym_id"]
            rec["actual_ref_chain"] = rec["auth_asym_id"]
            rec["pdb_path"] = str(structure_path.relative_to(ROOT))
            rec["cameo_target"] = rec["target_id"]
            rec["submission_date"] = rec["release_date"]
            rec["difficulty"] = np.nan
            rec["quaternary_state"] = np.nan
            rec["monomeric_model_count"] = np.nan
            rec["ref_pdb_chain"] = rec["target_id"]
            rec["ref_pdb_id"] = rec["entry_id"]
            rec["metadata_sequence_length"] = rec["seq_len"]
            rec["msa_rows"] = np.nan
            rec["msa_nonquery_rows"] = np.nan
            rec["msa_log10_nonquery_rows"] = np.nan
            rec["msa_top_nonquery_identity"] = np.nan
            rec["charge_level_name"] = f"L{int(level)}_charge_quantile"
            rec["scaling_group"] = rec["charge_level_name"]
            level_rows.append(rec)
        selected_rows.extend(level_rows)
        diagnostics.append(
            {
                "level": int(level),
                "eligible_before_download": int(len(part)),
                "download_attempts": int(attempts),
                "pdb_failures": int(pdb_failures),
                "structure_size_skips": int(structure_size_skips),
                "entry_cap_skips": int(entry_cap_skips),
                "selected": int(len(level_rows)),
                "selected_entries": int(pd.DataFrame(level_rows)["entry_id"].nunique()) if level_rows else 0,
            }
        )
    panel = pd.DataFrame(selected_rows)
    panel = panel.sort_values(["charge_level", "calc_seq_abs_charge_per_residue", "target_id"]).reset_index(drop=True)
    return panel, diagnostics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PDB.mkdir(parents=True, exist_ok=True)
    prior_targets, prior_sequences, prior_report = load_prior_exclusions()

    pool = pd.read_csv(POOL)
    pool["sequence_upper"] = pool["sequence"].astype(str).str.upper()
    pool["sequence_hash"] = pool["sequence_upper"].map(sha256_text)
    pool["calc_seq_abs_charge_per_residue"] = pool["sequence_upper"].map(calc_abs_charge_per_residue)

    eligible = pool.copy()
    eligible = eligible[eligible["seq_len"].between(120, 300)].copy()
    eligible = eligible[pd.to_numeric(eligible["resolution"], errors="coerce").fillna(99) <= 4.0].copy()
    eligible = eligible[eligible["auth_asym_id"].astype(str).str.len().eq(1)].copy()
    eligible = eligible[eligible["aa20_fraction"].ge(1.0)].copy()
    eligible = eligible.drop_duplicates("sequence_upper").copy()
    eligible = eligible[~eligible["target_id"].astype(str).isin(prior_targets)].copy()
    eligible = eligible[~eligible["sequence_upper"].isin(prior_sequences)].copy()
    eligible = eligible[pd.to_numeric(eligible["calc_seq_abs_charge_per_residue"], errors="coerce").notna()].copy()
    eligible = eligible.sort_values(["calc_seq_abs_charge_per_residue", "seq_len", "target_id"]).reset_index(drop=True)
    eligible = assign_charge_levels(eligible, n_levels=5)
    eligible.to_csv(OUT_ELIGIBLE, index=False)

    panel, selection_diagnostics = select_panel(eligible, target_per_level=20, max_per_entry=2)
    panel.to_csv(OUT_PANEL, index=False)

    panel_level_summary = (
        panel.groupby(["charge_level", "charge_level_name"])
        .agg(
            n=("target_id", "count"),
            entries=("entry_id", "nunique"),
            charge_min=("calc_seq_abs_charge_per_residue", "min"),
            charge_mean=("calc_seq_abs_charge_per_residue", "mean"),
            charge_max=("calc_seq_abs_charge_per_residue", "max"),
            seq_len_median=("seq_len", "median"),
            resolution_median=("resolution", "median"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    manifest = {
        "axis": {
            "axis_id": "fresh_external_calc_seq_abs_charge_per_residue_q5",
            "feature": "calc_seq_abs_charge_per_residue",
            "direction": "high",
            "definition": "absolute net sequence charge per residue from AA sequence only; higher charge level is the predicted harder tail",
            "frozen_before_target_scoring": True,
            "level_rule": "five quantile bands from eligible unscored RCSB pool using rank(method='first') on calc_seq_abs_charge_per_residue",
            "eligible_level_stats": level_thresholds(eligible),
        },
        "filters": [
            "RCSB candidate pool inherited from prior external candidate manifest",
            "exclude all prior SPECTRA CAMEO/RCSB target_ids and exact sequences used in the charge-axis discovery evidence",
            "protein AA20-only chains",
            "120 <= sequence length <= 300",
            "experimental resolution <= 4.0 when present",
            "one-character auth_asym_id",
            "unique exact sequence before band assignment",
            "max two selected chains per PDB entry across the final panel",
            "downloaded reference file size <= 50 MB",
        ],
        "source_pool": str(POOL),
        "source_pool_rows": int(len(pool)),
        "eligible_unscored_rows": int(len(eligible)),
        "prior_exclusions": prior_report,
        "target_per_level": 20,
        "max_selected_chains_per_entry": 2,
        "max_structure_mb": MAX_STRUCTURE_BYTES / 1_000_000,
        "panel_rows": int(len(panel)),
        "panel_entries": int(panel["entry_id"].nunique()) if len(panel) else 0,
        "panel_level_summary": panel_level_summary,
        "selection_diagnostics": selection_diagnostics,
        "artifacts": {
            "eligible_pool": str(OUT_ELIGIBLE.relative_to(ROOT)),
            "candidate_features": str(OUT_PANEL.relative_to(ROOT)),
        },
        "notes": [
            "No target-model scores, confidence values, or reference/prediction errors are used for candidate filtering or split membership.",
            "Panel construction intentionally protects the high-charge tail first because that tail is the sparse confirmation level.",
            "Length and resolution are recorded for adjustment in the final SPC summary.",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
