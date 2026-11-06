#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_viral_disulfide_axis_pilot_20260602")
PREV = Path("/ewsc/yektefai/spectra_runs/esmfold2_all_scored_evidence_loop_20260601T230100Z")
POOL = PREV / "spectra_ready_dataset" / "expanded_candidate_pool_features.csv"
SCORED = PREV / "spectra_ready_dataset" / "combined_evidence_with_cys_confirmation.csv"
OUT_DIR = ROOT / "spectra_ready_dataset"
RAW_DIR = OUT_DIR / "raw_rcsb_viral_disulfide_pilot"
STRUCT_DIR = RAW_DIR / "experiment_structures"
OUT_ELIGIBLE = OUT_DIR / "eligible_unscored_viral_disulfide_pool.csv"
OUT_PANEL = OUT_DIR / "viral_disulfide_pilot_candidate_features.csv"
OUT_MANIFEST = ROOT / "viral_disulfide_pilot_manifest.json"

PDB_URL = "https://files.rcsb.org/download/{entry_id}.pdb"
CIF_URL = "https://files.rcsb.org/download/{entry_id}.cif"
MAX_STRUCTURE_BYTES = 50_000_000
AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_virus_or_phage(organism: str) -> bool:
    org = str(organism or "").lower()
    return "virus" in org or "phage" in org


def disulfide_capacity_per100(sequence: str) -> float:
    seq = str(sequence or "").upper()
    if not seq:
        return math.nan
    return (seq.count("C") // 2) / (len(seq) / 100.0)


def axis_score(row: pd.Series) -> int:
    virus = int(is_virus_or_phage(str(row.get("organism", ""))))
    dense = int(float(row["viral_disulfide_capacity_per100"]) >= 1.2)
    return virus + dense


def length_bin(seq_len: float) -> str:
    seq_len = float(seq_len)
    if seq_len <= 120:
        return "L_080_120"
    if seq_len <= 180:
        return "L_121_180"
    if seq_len <= 260:
        return "L_181_260"
    if seq_len <= 360:
        return "L_261_360"
    return "L_361_500"


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
    pdb_path = STRUCT_DIR / f"{target_id}.pdb"
    if download_text(PDB_URL.format(entry_id=entry_id), pdb_path, ("HEADER", "TITLE", "COMPND", "REMARK", "ATOM")):
        if pdb_path.stat().st_size <= MAX_STRUCTURE_BYTES:
            return True, pdb_path, "pdb"
    cif_path = STRUCT_DIR / f"{target_id}.cif"
    if download_text(CIF_URL.format(entry_id=entry_id), cif_path, ("data_", "#")):
        if cif_path.stat().st_size <= MAX_STRUCTURE_BYTES:
            return True, cif_path, "cif"
    return False, None, "failed"


def prepare_eligible() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    scored = pd.read_csv(SCORED)
    scored_targets = set(scored["target_id"].dropna().astype(str))
    scored_sequences = set(scored["sequence_hash"].dropna().astype(str))

    eligible = pool.copy()
    eligible["seq_len"] = pd.to_numeric(eligible["seq_len"], errors="coerce")
    eligible["resolution"] = pd.to_numeric(eligible["resolution"], errors="coerce")
    eligible = eligible[~eligible["target_id"].astype(str).isin(scored_targets)].copy()
    eligible = eligible[~eligible["sequence_hash"].astype(str).isin(scored_sequences)].copy()
    eligible = eligible[eligible["seq_len"].between(80, 500)].copy()
    eligible = eligible[eligible["resolution"].le(4.0)].copy()
    eligible = eligible[eligible["sequence"].astype(str).str.upper().map(lambda s: set(s).issubset(AA20))].copy()
    eligible["viral_axis_virus_or_phage"] = eligible["organism"].map(is_virus_or_phage).astype(int)
    eligible["viral_disulfide_capacity_per100"] = eligible["sequence"].map(disulfide_capacity_per100)
    eligible["viral_axis_disulfide_dense"] = (eligible["viral_disulfide_capacity_per100"] >= 1.2).astype(int)
    eligible["viral_disulfide_score"] = eligible.apply(axis_score, axis=1).astype(int)
    eligible["viral_disulfide_level_name"] = eligible["viral_disulfide_score"].map(
        {
            0: "L0_neither_viral_nor_disulfide_dense",
            1: "L1_viral_or_disulfide_dense",
            2: "L2_viral_and_disulfide_dense",
        }
    )
    eligible["viral_length_bin"] = eligible["seq_len"].map(length_bin)
    eligible.to_csv(OUT_ELIGIBLE, index=False)
    return eligible


def select_panel(eligible: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    selected: list[pd.Series] = []
    diagnostics: list[dict] = []
    entry_counts: dict[str, int] = {}

    level2 = eligible[eligible["viral_disulfide_score"].eq(2)].copy()
    level2 = level2.sort_values(["organism", "seq_len", "target_id"])
    target_counts = {2: len(level2), 1: min(14, max(10, len(level2) * 2)), 0: min(14, max(10, len(level2) * 2))}

    for level in [2, 1, 0]:
        part = eligible[eligible["viral_disulfide_score"].eq(level)].copy()
        target_n = target_counts[level]
        if level in {0, 1} and not level2.empty:
            # Match controls to the top-stress length/method/resolution profile.
            target_len = float(level2["seq_len"].median())
            target_res = float(level2["resolution"].median())
            target_em_frac = float(level2["method"].astype(str).str.contains("ELECTRON", case=False, regex=True).mean())
            part["_rank"] = (
                (part["seq_len"].astype(float) - target_len).abs() / 120.0
                + (part["resolution"].astype(float) - target_res).abs() / 2.0
                + (part["method"].astype(str).str.contains("ELECTRON", case=False, regex=True).astype(float) - target_em_frac).abs()
                + part["target_id"].astype(str).rank(method="first") / 10000.0
            )
            part = part.sort_values("_rank")
        else:
            part["_rank"] = part["target_id"].astype(str).rank(method="first") / 10000.0
            part = part.sort_values("_rank")

        failures = 0
        entry_cap_skips = 0
        chosen = 0
        for _, rec in part.iterrows():
            if chosen >= target_n:
                break
            entry = str(rec["entry_id"])
            if entry_counts.get(entry, 0) >= 1:
                entry_cap_skips += 1
                continue
            ok, path, fmt = download_structure(entry, str(rec["target_id"]))
            if not ok or path is None:
                failures += 1
                continue
            entry_counts[entry] = entry_counts.get(entry, 0) + 1
            out = rec.copy()
            out["pdb_download_ok"] = True
            out["structure_format"] = fmt
            out["eligible_reference"] = True
            out["ref_chain"] = rec["auth_asym_id"]
            out["actual_ref_chain"] = rec["auth_asym_id"]
            out["pdb_path"] = str(path)
            out["cameo_target"] = rec["target_id"]
            out["submission_date"] = rec.get("release_date")
            out["difficulty"] = pd.NA
            out["quaternary_state"] = pd.NA
            out["monomeric_model_count"] = pd.NA
            out["ref_pdb_chain"] = rec["target_id"]
            out["ref_pdb_id"] = rec["entry_id"]
            out["metadata_sequence_length"] = rec["seq_len"]
            out["msa_rows"] = pd.NA
            out["msa_nonquery_rows"] = pd.NA
            out["msa_log10_nonquery_rows"] = pd.NA
            out["msa_top_nonquery_identity"] = pd.NA
            out["scaling_group"] = out["viral_disulfide_level_name"]
            selected.append(out)
            chosen += 1
        diagnostics.append(
            {
                "level": int(level),
                "eligible": int(len(part)),
                "target_n": int(target_n),
                "selected": int(chosen),
                "download_failures": int(failures),
                "entry_cap_skips": int(entry_cap_skips),
            }
        )
    panel = pd.DataFrame(selected)
    if not panel.empty:
        panel = panel.sort_values(["viral_disulfide_score", "seq_len", "target_id"]).reset_index(drop=True)
    return panel, diagnostics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eligible = prepare_eligible()
    panel, diagnostics = select_panel(eligible)
    panel.to_csv(OUT_PANEL, index=False)
    level_summary = []
    for level, part in panel.groupby("viral_disulfide_score"):
        level_summary.append(
            {
                "level": int(level),
                "n": int(len(part)),
                "unique_entries": int(part["entry_id"].nunique()),
                "virus_n": int(part["viral_axis_virus_or_phage"].sum()),
                "disulfide_dense_n": int(part["viral_axis_disulfide_dense"].sum()),
                "disulfide_capacity_mean": float(part["viral_disulfide_capacity_per100"].mean()),
                "seq_len_median": float(part["seq_len"].median()),
                "resolution_median": float(part["resolution"].median()),
                "em_n": int(part["method"].astype(str).str.contains("ELECTRON", case=False, regex=True).sum()),
                "xray_n": int(part["method"].astype(str).str.contains("X-RAY", case=False, regex=True).sum()),
                "organisms": part["organism"].value_counts().head(10).to_dict(),
            }
        )
    manifest = {
        "created_at_utc": now(),
        "axis_id": "viral_disulfide_score_pilot",
        "axis_definition": {
            "score": "virus_or_phage + disulfide_dense",
            "virus_or_phage": "organism string contains virus or phage",
            "disulfide_dense": "floor(cysteine_count / 2) per 100 aa >= 1.2",
            "levels": {
                "0": "neither viral/phage nor disulfide-dense",
                "1": "exactly one of viral/phage or disulfide-dense",
                "2": "both viral/phage and disulfide-dense",
            },
            "unit": "RCSB protein chain",
            "status": "frozen before pilot target-model scoring",
        },
        "panel_summary": {
            "n_panel": int(len(panel)),
            "level_summary": level_summary,
            "eligible_counts": {str(k): int(v) for k, v in eligible["viral_disulfide_score"].value_counts().sort_index().items()},
            "diagnostics": diagnostics,
        },
        "artifacts": {
            "eligible_pool": str(OUT_ELIGIBLE),
            "panel": str(OUT_PANEL),
        },
        "validity_note": "This is a bounded pilot because only seven unused level-2 candidates were available in the current broad pool.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["panel_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
