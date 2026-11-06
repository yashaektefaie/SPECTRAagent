#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_envelope_fusion_axis_confirmation_20260602")
SOURCE_CANDIDATES = Path("/ewsc/yektefai/spectra_runs/esmfold2_viral_disulfide_axis_pilot_20260602/spectra_ready_dataset/rcsb_envelope_fusion_candidates.csv")
PREV_COMBINED = Path("/ewsc/yektefai/spectra_runs/esmfold2_all_scored_evidence_loop_20260601T230100Z/spectra_ready_dataset/combined_evidence_with_cys_confirmation.csv")
PREV_PILOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_viral_disulfide_axis_pilot_20260602/target_model_results.viral_disulfide_pilot.target_level.csv")
OUT_DIR = ROOT / "spectra_ready_dataset"
STRUCT_DIR = OUT_DIR / "raw_rcsb_envelope_fusion_confirmation" / "experiment_structures"
OUT_PANEL = OUT_DIR / "envelope_fusion_confirmation_candidate_features.csv"
OUT_POOL = OUT_DIR / "eligible_envelope_fusion_pool.csv"
OUT_MANIFEST = ROOT / "envelope_fusion_confirmation_manifest.json"

ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{entry}/{entity}"
PDB_URL = "https://files.rcsb.org/download/{entry}.pdb"
CIF_URL = "https://files.rcsb.org/download/{entry}.cif"
MAX_STRUCTURE_BYTES = 50_000_000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ffloat(value):
    try:
        if value is None or str(value).strip() == "" or str(value).lower() == "nan":
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def get_json(url: str) -> dict:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    time.sleep(0.02)
    return resp.json()


def first_chain_for_entity(entry: str, entity: str) -> str:
    data = get_json(ENTITY_URL.format(entry=entry, entity=entity))
    strand = data.get("entity_poly", {}).get("pdbx_strand_id") or ""
    chains = [x.strip() for x in str(strand).split(",") if x.strip()]
    if not chains:
        raise ValueError(f"no chain id for {entry}_{entity}")
    return chains[0]


def download_text(url: str, out_path: Path, accepted_prefixes: tuple[str, ...]) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        return False
    text = resp.content.decode("utf-8", errors="replace")
    if not text.startswith(accepted_prefixes):
        return False
    out_path.write_text(text)
    time.sleep(0.03)
    return True


def download_structure(entry: str, target_id: str) -> tuple[bool, Path | None, str]:
    # Prefer mmCIF here because viral assemblies often use auth chain ids that
    # are safer in mmCIF than PDB text.
    cif_path = STRUCT_DIR / f"{target_id}.cif"
    if download_text(CIF_URL.format(entry=entry), cif_path, ("data_", "#")) and cif_path.stat().st_size <= MAX_STRUCTURE_BYTES:
        return True, cif_path, "cif"
    pdb_path = STRUCT_DIR / f"{target_id}.pdb"
    if download_text(PDB_URL.format(entry=entry), pdb_path, ("HEADER", "TITLE", "COMPND", "REMARK", "ATOM")) and pdb_path.stat().st_size <= MAX_STRUCTURE_BYTES:
        return True, pdb_path, "pdb"
    return False, None, "failed"


def load_scored_hashes() -> tuple[set[str], set[str]]:
    hashes: set[str] = set()
    targets: set[str] = set()
    for path in [PREV_COMBINED, PREV_PILOT]:
        if not path.exists():
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("sequence_hash"):
                    hashes.add(row["sequence_hash"])
                if row.get("target_id"):
                    targets.add(row["target_id"])
    return hashes, targets


def level(row: dict) -> int:
    disulfide_dense = int((ffloat(row["disulfide_capacity_per100"]) or 0) >= 1.2)
    membrane_lowglyc = int((ffloat(row["tm_helix_proxy_calc"]) or 0) >= 1.8 and int(row["nglyc_motif_count"]) <= 2)
    return disulfide_dense + membrane_lowglyc


def select_level(rows: list[dict], n: int, max_per_org: int = 2) -> list[dict]:
    rows = sorted(rows, key=lambda r: (r["release_date"], r["organism"], r["entity_description"], r["target_id"]), reverse=True)
    selected = []
    org_counts = Counter()
    seq_seen = set()
    for row in rows:
        if row["sequence_hash"] in seq_seen:
            continue
        if org_counts[row["organism"]] >= max_per_org:
            continue
        selected.append(row)
        seq_seen.add(row["sequence_hash"])
        org_counts[row["organism"]] += 1
        if len(selected) >= n:
            break
    return selected


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    scored_hashes, scored_targets = load_scored_hashes()
    pool = []
    with SOURCE_CANDIDATES.open() as handle:
        for row in csv.DictReader(handle):
            if row["sequence_hash"] in scored_hashes or row["target_id"] in scored_targets:
                continue
            if row["release_date"][:4] < "2024":
                continue
            row = dict(row)
            row["envelope_fusion_score"] = level(row)
            row["envelope_fusion_level_name"] = {
                0: "L0_low_envelope_fusion_sequence_proxy",
                1: "L1_partial_envelope_fusion_sequence_proxy",
                2: "L2_strict_envelope_fusion_sequence_proxy",
            }[row["envelope_fusion_score"]]
            pool.append(row)

    with OUT_POOL.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pool[0].keys()))
        writer.writeheader()
        writer.writerows(pool)

    targets = {2: 16, 1: 16, 0: 16}
    selected = []
    diagnostics = []
    for lev in [2, 1, 0]:
        part = [r for r in pool if r["envelope_fusion_score"] == lev]
        chosen = select_level(part, targets[lev])
        failures = 0
        packaged = []
        for row in chosen:
            entry, entity = row["entry_id"], row["entity_id"]
            try:
                chain = first_chain_for_entity(entry, entity)
                target_id = f"{entry}_{entity}_{chain}"
                ok, path, fmt = download_structure(entry, target_id)
                if not ok or path is None:
                    failures += 1
                    continue
                out = dict(row)
                out["target_id"] = target_id
                out["auth_asym_id"] = chain
                out["asym_id"] = chain
                out["ref_chain"] = chain
                out["actual_ref_chain"] = chain
                out["pdb_download_ok"] = True
                out["structure_format"] = fmt
                out["eligible_reference"] = True
                out["pdb_path"] = str(path)
                out["cameo_target"] = target_id
                out["submission_date"] = row["release_date"]
                out["difficulty"] = ""
                out["quaternary_state"] = ""
                out["monomeric_model_count"] = ""
                out["ref_pdb_chain"] = target_id
                out["ref_pdb_id"] = entry
                out["metadata_sequence_length"] = row["seq_len"]
                out["msa_rows"] = ""
                out["msa_nonquery_rows"] = ""
                out["msa_log10_nonquery_rows"] = ""
                out["msa_top_nonquery_identity"] = ""
                out["scaling_group"] = out["envelope_fusion_level_name"]
                packaged.append(out)
            except Exception:
                failures += 1
        selected.extend(packaged)
        diagnostics.append({"level": lev, "eligible": len(part), "chosen": len(chosen), "packaged": len(packaged), "failures": failures})

    selected = sorted(selected, key=lambda r: (r["envelope_fusion_score"], r["seq_len"], r["target_id"]))
    with OUT_PANEL.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    summary = []
    for lev in [0, 1, 2]:
        part = [r for r in selected if r["envelope_fusion_score"] == lev]
        summary.append(
            {
                "level": lev,
                "n": len(part),
                "organisms": dict(Counter(r["organism"] for r in part)),
                "descriptions": dict(Counter(r["entity_description"] for r in part)),
                "seq_len_median": sorted(float(r["seq_len"]) for r in part)[len(part) // 2] if part else None,
                "resolution_median": sorted(float(r["resolution"]) for r in part)[len(part) // 2] if part else None,
            }
        )
    manifest = {
        "created_at_utc": now(),
        "axis_id": "viral_envelope_fusion_sequence_proxy_confirmation",
        "axis_definition": {
            "scope": "viral/envelope RCSB polymer entities, release year >= 2024, exact sequences and target IDs excluded from prior scored evidence",
            "score": "disulfide_dense + membrane_lowglyc_proxy",
            "disulfide_dense": "floor(cysteine_count/2) per 100 aa >= 1.2",
            "membrane_lowglyc_proxy": "tm_helix_proxy_calc >= 1.8 and N[^P][ST] motif count <= 2",
            "levels": {
                "0": "neither condition",
                "1": "exactly one condition",
                "2": "both conditions; strict envelope/fusion sequence proxy",
            },
        },
        "panel_summary": summary,
        "diagnostics": diagnostics,
        "artifacts": {"pool": str(OUT_POOL), "panel": str(OUT_PANEL)},
        "validity_note": "This is an expanded confirmation of an outcome-mined stricter axis; releases include 2024-2026 and may not be pretraining-novel.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_panel": len(selected), "summary": summary, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
