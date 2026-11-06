#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
PRIOR_EXTERNAL = Path(
    "/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run/"
    "external_rcsb_same_bucket_expansion_20260531"
)
PRIOR_ROOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_beyond_evolutionary_support_final_run")
POOL = PRIOR_EXTERNAL / "rcsb_external_candidate_pool.csv"
AXIS_JSON = PRIOR_ROOT / "axis_definitions_followup.json"
PRIOR_RESULTS = PRIOR_EXTERNAL / "target_model_results.scaled_samples.csv"

OUT_RAW = ROOT / "spectra_ready_dataset" / "raw_rcsb_external"
OUT_PDB = OUT_RAW / "experiment_structures"
OUT_PANEL = ROOT / "spectra_ready_dataset" / "candidate_features.csv"
OUT_RESULTS = ROOT / "target_model_results.scaled_samples.csv"
OUT_MANIFEST = ROOT / "expanded_external_panel_manifest.json"

PDB_URL = "https://files.rcsb.org/download/{entry_id}.pdb"


def load_high_threshold() -> float:
    axes = json.loads(AXIS_JSON.read_text())["axes"]
    primary = next(a for a in axes if a["axis_id"] == "seq_intrinsic_flexible_nonhelical_composite_7")
    return float(primary["level_stats"][-1]["min"])


def level_for_rank(rank: float, high_threshold: float) -> tuple[int, str]:
    if rank < 0.30:
        return 0, "L0_rank_lt_0.30"
    if rank < 0.50:
        return 1, "L1_rank_0.30_0.50"
    if rank < 0.65:
        return 2, "L2_rank_0.50_0.65"
    if rank < high_threshold:
        return 3, "L3_rank_0.65_to_high_threshold"
    return 4, "L4_frozen_cameo_high_tail"


def download_pdb(entry_id: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    resp = requests.get(PDB_URL.format(entry_id=entry_id), timeout=60)
    if resp.status_code != 200 or not resp.text.startswith(("HEADER", "TITLE", "COMPND", "REMARK", "ATOM")):
        return False
    out_path.write_text(resp.text)
    time.sleep(0.02)
    return True


def select_evenly(candidates: pd.DataFrame, n: int, used_entries: set[str]) -> pd.DataFrame:
    avail = candidates[~candidates["entry_id"].astype(str).isin(used_entries)].copy()
    avail = avail.sort_values(["flexible_nonhelical_composite_rank", "seq_len", "target_id"]).reset_index(drop=True)
    if len(avail) <= n:
        return avail

    chosen_indices: list[int] = []
    taken = set()
    target_positions = np.linspace(0, len(avail) - 1, n)
    for pos in target_positions:
        center = int(round(float(pos)))
        order = sorted(range(len(avail)), key=lambda i: (abs(i - center), i))
        for idx in order:
            if idx not in taken:
                taken.add(idx)
                chosen_indices.append(idx)
                break
    return avail.iloc[chosen_indices].sort_values(["flexible_nonhelical_composite_rank", "target_id"])


def main() -> None:
    OUT_PDB.mkdir(parents=True, exist_ok=True)
    high_threshold = load_high_threshold()
    pool = pd.read_csv(POOL)
    pool = pool[pool["seq_len"].between(100, 400)].copy()
    pool = pool[pd.to_numeric(pool["resolution"], errors="coerce").fillna(99) <= 4.0].copy()
    pool = pool[pool["auth_asym_id"].astype(str).str.len().eq(1)].copy()
    pool = pool.drop_duplicates("sequence").copy()
    levels = pool["flexible_nonhelical_composite_rank"].map(lambda x: level_for_rank(float(x), high_threshold))
    pool["external_axis_level"] = [x[0] for x in levels]
    pool["external_axis_level_name"] = [x[1] for x in levels]

    target_per_level = 30
    selected_parts: list[pd.DataFrame] = []
    used_entries: set[str] = set()
    level_order = [0, 4, 3, 2, 1]
    for level in level_order:
        candidates = pool[pool["external_axis_level"].eq(level)].copy()
        selected = select_evenly(candidates, target_per_level, used_entries)
        selected_parts.append(selected)
        used_entries.update(selected["entry_id"].astype(str))

    panel = pd.concat(selected_parts, ignore_index=True)
    panel = panel.sort_values(["external_axis_level", "flexible_nonhelical_composite_rank", "target_id"]).reset_index(drop=True)

    pdb_ok = []
    for _, row in panel.iterrows():
        pdb_path = OUT_PDB / f"{row['target_id']}.pdb"
        pdb_ok.append(download_pdb(str(row["entry_id"]), pdb_path))
    panel["pdb_download_ok"] = pdb_ok
    panel = panel[panel["pdb_download_ok"]].copy()

    panel["eligible_reference"] = True
    panel["ref_chain"] = panel["auth_asym_id"]
    panel["actual_ref_chain"] = panel["auth_asym_id"]
    panel["pdb_path"] = panel["target_id"].map(
        lambda x: f"spectra_ready_dataset/raw_rcsb_external/experiment_structures/{x}.pdb"
    )
    panel["cameo_target"] = panel["target_id"]
    panel["submission_date"] = panel["release_date"]
    panel["difficulty"] = np.nan
    panel["quaternary_state"] = np.nan
    panel["monomeric_model_count"] = np.nan
    panel["ref_pdb_chain"] = panel["target_id"]
    panel["ref_pdb_id"] = panel["entry_id"]
    panel["metadata_sequence_length"] = panel["seq_len"]
    panel["msa_rows"] = np.nan
    panel["msa_nonquery_rows"] = np.nan
    panel["msa_log10_nonquery_rows"] = np.nan
    panel["msa_top_nonquery_identity"] = np.nan
    panel["scaling_group"] = panel["external_axis_level_name"]

    panel.to_csv(OUT_PANEL, index=False)

    reused_targets = 0
    reused_samples = 0
    if PRIOR_RESULTS.exists():
        prior = pd.read_csv(PRIOR_RESULTS)
        keep = prior[prior["target_id"].astype(str).isin(set(panel["target_id"].astype(str)))].copy()
        if not keep.empty:
            keep.to_csv(OUT_RESULTS, index=False)
            reused_targets = int(keep["target_id"].nunique())
            reused_samples = int(len(keep))

    level_summary = (
        panel.groupby(["external_axis_level", "external_axis_level_name"])
        .agg(
            n=("target_id", "count"),
            entries=("entry_id", "nunique"),
            rank_min=("flexible_nonhelical_composite_rank", "min"),
            rank_mean=("flexible_nonhelical_composite_rank", "mean"),
            rank_max=("flexible_nonhelical_composite_rank", "max"),
            seq_len_median=("seq_len", "median"),
            resolution_median=("resolution", "median"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    manifest: dict[str, Any] = {
        "source_pool": str(POOL),
        "source_pool_rows": int(pd.read_csv(POOL).shape[0]),
        "axis": "seq_intrinsic_flexible_nonhelical_composite_7",
        "axis_feature": "flexible_nonhelical_composite_rank",
        "high_tail_threshold_frozen_from_cameo": high_threshold,
        "level_definition": [
            "L0: rank < 0.30",
            "L1: 0.30 <= rank < 0.50",
            "L2: 0.50 <= rank < 0.65",
            "L3: 0.65 <= rank < frozen high-tail threshold",
            "L4: rank >= frozen CAMEO high-tail threshold",
        ],
        "filters": [
            "RCSB initial release date >= 2025-01-01 inherited from source pool",
            "protein polymer chains, AA20-only sequence, 100 <= sequence length <= 400",
            "experimental resolution <= 4.0 when present",
            "one-character auth_asym_id",
            "unique sequence",
            "deterministic one-entry reuse avoidance across selected levels",
        ],
        "target_per_level": target_per_level,
        "panel_rows": int(len(panel)),
        "panel_entries": int(panel["entry_id"].nunique()),
        "level_summary": level_summary,
        "prior_scaled_results_reused_targets": reused_targets,
        "prior_scaled_results_reused_samples": reused_samples,
        "notes": [
            "Split membership uses only frozen prospective sequence-composition features and RCSB metadata.",
            "No ESMFold2 predictions, confidence, or reference errors are used to define levels.",
            "This is an expanded external SPC panel, not a matched-pair design.",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
