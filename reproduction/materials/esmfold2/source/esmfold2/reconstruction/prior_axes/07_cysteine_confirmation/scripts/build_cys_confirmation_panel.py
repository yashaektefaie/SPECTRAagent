#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "spectra_ready_dataset"
RAW_DIR = OUT_DIR / "raw_rcsb_cys_confirmation"
STRUCT_DIR = RAW_DIR / "experiment_structures"
POOL = OUT_DIR / "expanded_candidate_pool_features.csv"
SCORED = OUT_DIR / "expanded_all_scored_evidence_table.csv"
OUT_ELIGIBLE = OUT_DIR / "eligible_unscored_cys_pool.csv"
OUT_PANEL = OUT_DIR / "cys_confirmation_candidate_features.csv"
OUT_MANIFEST = ROOT / "cys_confirmation_panel_manifest.json"
OUT_LEDGER = ROOT / "axis_ledger.json"
OUT_STATE = ROOT / "spectra_loop_state.json"
OUT_COMMANDS = ROOT / "commands_run.json"

PDB_URL = "https://files.rcsb.org/download/{entry_id}.pdb"
CIF_URL = "https://files.rcsb.org/download/{entry_id}.cif"
MAX_STRUCTURE_BYTES = 50_000_000
AA20 = set("ACDEFGHIKLMNPQRSTVWY")

LENGTH_BIN_EDGES = [0, 120, 180, 260, 360, 500]
LENGTH_BIN_QUOTAS = {
    "L_short_le120": 5,
    "L_120_180": 1,
    "L_260_360": 4,
    "L_360_500": 10,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.load(open(path))
    return default


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def length_bin(seq_len: float) -> str | None:
    if seq_len <= 120:
        return "L_short_le120"
    if 120 < seq_len <= 180:
        return "L_120_180"
    if 260 < seq_len <= 360:
        return "L_260_360"
    if 360 < seq_len <= 500:
        return "L_360_500"
    return None


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
        if pdb_path.stat().st_size > MAX_STRUCTURE_BYTES:
            return False, pdb_path, "pdb_too_large"
        return True, pdb_path, "pdb"
    cif_path = STRUCT_DIR / f"{target_id}.cif"
    if download_text(CIF_URL.format(entry_id=entry_id), cif_path, ("data_", "#")):
        if cif_path.stat().st_size > MAX_STRUCTURE_BYTES:
            return False, cif_path, "cif_too_large"
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
    eligible["cys_fraction"] = pd.to_numeric(eligible["cys_fraction"], errors="coerce")
    eligible = eligible[~eligible["target_id"].astype(str).isin(scored_targets)].copy()
    eligible = eligible[~eligible["sequence_hash"].astype(str).isin(scored_sequences)].copy()
    eligible = eligible[eligible["seq_len"].between(80, 500)].copy()
    eligible = eligible[eligible["resolution"].le(4.0)].copy()
    eligible = eligible[eligible["cys_fraction"].notna()].copy()
    eligible = eligible[eligible["sequence"].astype(str).str.upper().map(lambda s: set(s).issubset(AA20))].copy()
    eligible["cys_level"] = pd.qcut(
        eligible["cys_fraction"].rank(method="first"),
        q=5,
        labels=False,
        duplicates="drop",
    ).astype(int)
    eligible["cys_level_name"] = eligible["cys_level"].map(lambda x: f"L{int(x)}_cys_quantile")
    eligible["cys_axis_feature"] = "cys_fraction"
    eligible["cys_direction"] = "high"
    eligible["cys_length_bin"] = eligible["seq_len"].map(length_bin)
    eligible = eligible[eligible["cys_length_bin"].notna()].copy()
    eligible.to_csv(OUT_ELIGIBLE, index=False)
    return eligible


def select_panel(eligible: pd.DataFrame, max_per_entry: int = 2) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    selected_rows: list[pd.Series] = []
    diagnostics = []
    entry_counts: dict[str, int] = {}
    method_targets = {"ELECTRON MICROSCOPY": 0.70, "X-RAY DIFFRACTION": 0.30}

    for cys_level in range(5):
        level_part = eligible[eligible["cys_level"].eq(cys_level)].copy()
        for bin_name, quota in LENGTH_BIN_QUOTAS.items():
            part = level_part[level_part["cys_length_bin"].eq(bin_name)].copy()
            if part.empty:
                diagnostics.append(
                    {
                        "cys_level": cys_level,
                        "length_bin": bin_name,
                        "eligible_before_download": 0,
                        "selected": 0,
                        "download_failures": 0,
                        "entry_cap_skips": 0,
                    }
                )
                continue
            # Prefer candidates close to the bin's length midpoint and to the
            # global panel resolution median while preserving method mixture.
            midpoint = {
                "L_short_le120": 100,
                "L_120_180": 150,
                "L_260_360": 310,
                "L_360_500": 430,
            }[bin_name]
            target_resolution = float(eligible["resolution"].median())
            part["_rank"] = (
                (part["seq_len"].astype(float) - midpoint).abs() / 100.0
                + (part["resolution"].astype(float) - target_resolution).abs() / 2.0
                + part["target_id"].astype(str).rank(method="first") / 10000.0
            )
            part = part.sort_values("_rank")
            level_bin_rows = []
            failures = 0
            entry_cap_skips = 0
            for _, rec in part.iterrows():
                if len(level_bin_rows) >= quota:
                    break
                entry = str(rec["entry_id"])
                if entry_counts.get(entry, 0) >= max_per_entry:
                    entry_cap_skips += 1
                    continue
                ok, structure_path, fmt = download_structure(entry, str(rec["target_id"]))
                if not ok or structure_path is None:
                    failures += 1
                    continue
                if structure_path.stat().st_size > MAX_STRUCTURE_BYTES:
                    failures += 1
                    continue
                entry_counts[entry] = entry_counts.get(entry, 0) + 1
                out = rec.copy()
                out["pdb_download_ok"] = True
                out["structure_format"] = fmt
                out["eligible_reference"] = True
                out["ref_chain"] = rec["auth_asym_id"]
                out["actual_ref_chain"] = rec["auth_asym_id"]
                out["pdb_path"] = str(structure_path.relative_to(ROOT))
                out["cameo_target"] = rec["target_id"]
                out["submission_date"] = rec.get("release_date")
                out["difficulty"] = np.nan
                out["quaternary_state"] = np.nan
                out["monomeric_model_count"] = np.nan
                out["ref_pdb_chain"] = rec["target_id"]
                out["ref_pdb_id"] = rec["entry_id"]
                out["metadata_sequence_length"] = rec["seq_len"]
                out["msa_rows"] = np.nan
                out["msa_nonquery_rows"] = np.nan
                out["msa_log10_nonquery_rows"] = np.nan
                out["msa_top_nonquery_identity"] = np.nan
                out["scaling_group"] = out["cys_level_name"]
                level_bin_rows.append(out)
            selected_rows.extend(level_bin_rows)
            diagnostics.append(
                {
                    "cys_level": int(cys_level),
                    "length_bin": bin_name,
                    "quota": int(quota),
                    "eligible_before_download": int(len(part)),
                    "selected": int(len(level_bin_rows)),
                    "download_failures": int(failures),
                    "entry_cap_skips": int(entry_cap_skips),
                }
            )
    panel = pd.DataFrame(selected_rows)
    if not panel.empty:
        panel = panel.sort_values(["cys_level", "cys_length_bin", "seq_len", "target_id"]).reset_index(drop=True)
    return panel, diagnostics


def panel_summary(panel: pd.DataFrame, eligible: pd.DataFrame) -> dict[str, Any]:
    if panel.empty:
        return {"n_panel": 0}
    level_summary = []
    for level, part in panel.groupby("cys_level"):
        level_summary.append(
            {
                "level": int(level),
                "n": int(len(part)),
                "entries": int(part["entry_id"].nunique()),
                "cys_min": float(part["cys_fraction"].min()),
                "cys_mean": float(part["cys_fraction"].mean()),
                "cys_max": float(part["cys_fraction"].max()),
                "seq_len_median": float(part["seq_len"].median()),
                "resolution_median": float(part["resolution"].median()),
                "xray_n": int(part["method"].astype(str).str.contains("X-RAY", case=False, regex=True).sum()),
                "em_n": int(part["method"].astype(str).str.contains("ELECTRON", case=False, regex=True).sum()),
            }
        )
    level_df = pd.DataFrame(level_summary).sort_values("level")
    return {
        "n_panel": int(len(panel)),
        "unique_targets": int(panel["target_id"].nunique()),
        "unique_sequences": int(panel["sequence_hash"].nunique()),
        "unique_entries": int(panel["entry_id"].nunique()),
        "max_per_entry": int(panel["entry_id"].value_counts().max()),
        "level_summary": level_summary,
        "length_bin_counts": panel.groupby(["cys_level", "cys_length_bin"]).size().unstack(fill_value=0).to_dict(),
        "measured_cys_order_ok": bool(level_df["cys_mean"].is_monotonic_increasing),
        "min_level_n": int(level_df["n"].min()),
        "eligible_pool_level_counts": {
            str(k): int(v) for k, v in eligible["cys_level"].value_counts().sort_index().items()
        },
    }


def update_state_and_ledger(manifest: dict[str, Any]) -> None:
    state = load_json(OUT_STATE, {})
    state.update(
        {
            "created_at_utc": now(),
            "phase": "frozen_cys_confirmation_panel_ready",
            "frozen_axis": manifest["frozen_axis"],
            "artifacts": {
                **state.get("artifacts", {}),
                "cys_confirmation_manifest": str(OUT_MANIFEST),
                "cys_confirmation_candidate_features": str(OUT_PANEL),
                "eligible_unscored_cys_pool": str(OUT_ELIGIBLE),
            },
            "closure_gate": {
                "terminal_passed": False,
                "final_report_written": False,
                "reason": "High-cysteine axis is frozen and panelized but target-model confirmation has not yet been run.",
            },
        }
    )
    write_json(OUT_STATE, state)

    ledger = load_json(OUT_LEDGER, {"entries": []})
    ledger["entries"] = [e for e in ledger.get("entries", []) if e.get("axis_id") != "frozen_high_cys_confirmation"]
    ledger["entries"].append(
        {
            "axis_id": "frozen_high_cys_confirmation",
            "category": "frozen_confirmation_axis",
            "status": "panel_frozen_pending_target_scoring",
            "prospective": True,
            "outcome_informed": True,
            "frozen_before_target_scoring": True,
            "feature": "cys_fraction",
            "direction": "high",
            "manifest": str(OUT_MANIFEST),
            "candidate_features": str(OUT_PANEL),
            "panel_summary": manifest["panel_summary"],
            "claim_boundary_if_confirmed": manifest["frozen_axis"]["claim_boundary_if_confirmed"],
        }
    )
    ledger["created_at_utc"] = now()
    write_json(OUT_LEDGER, ledger)

    commands = load_json(OUT_COMMANDS, [])
    commands.append(
        {
            "command": "/ewsc/yektefai/envs/envs/pgt/bin/python scripts/build_cys_confirmation_panel.py",
            "cwd": str(ROOT),
            "purpose": "Freeze and construct a fresh high-cysteine RCSB confirmation panel before target-model scoring.",
            "status": "completed",
        }
    )
    write_json(OUT_COMMANDS, commands)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    eligible = prepare_eligible()
    panel, diagnostics = select_panel(eligible)
    panel.to_csv(OUT_PANEL, index=False)
    manifest = {
        "created_at_utc": now(),
        "frozen_axis": {
            "axis_id": "high_cys_fraction_fresh_rcsb_confirmation",
            "feature": "cys_fraction",
            "direction": "high",
            "unit": "single RCSB protein chain",
            "levels": "five quantile levels over eligible unscored RCSB pool; level increases with cysteine fraction",
            "inclusion": [
                "RCSB candidate pool from the prior external expansion manifest",
                "protein chains with 80 <= sequence length <= 500",
                "experimental resolution <= 4.0 A",
                "AA20-only sequence",
                "exact target_id and exact sequence not present in all accumulated scored evidence",
                "selected structure file downloadable from RCSB as PDB or mmCIF",
            ],
            "deduplication": "exclude exact target IDs and sequence hashes from discovery/scored evidence; cap selected chains to at most two per PDB entry",
            "sample_size_target": "20 targets per cysteine level, 100 total",
            "length_balance_rule": {
                "L_short_le120": 5,
                "L_120_180": 1,
                "L_260_360": 4,
                "L_360_500": 10,
            },
            "controls": [
                "seq_len",
                "resolution",
                "method/provenance",
                "tax_nonhuman_nonmodel_stress",
                "order_promoting_fraction",
                "calc_seq_abs_charge_per_residue",
                "fraction_charged_residues_fcr",
            ],
            "primary_metric": "CA-lDDT from best-by-pTM ESMFold2 scaled inference",
            "secondary_metrics": ["GDT-TS", "CA-RMSD", "baseline CA-lDDT"],
            "success_criteria": [
                "measured cysteine fraction increases monotonically across all five levels",
                "minimum scored level n >= 15",
                "top cysteine level CA-lDDT minus lower levels <= -0.04",
                "top two levels minus bottom two levels CA-lDDT <= -0.04 or monotone/threshold degradation is clear in the five-level curve",
                "length/resolution-adjusted cysteine stress coefficient remains negative",
                "baseline CA-lDDT top-minus-lower absolute movement <= 0.02 or is explicitly smaller than target-model movement",
                "GDT-TS or CA-RMSD changes agree in direction with CA-lDDT",
            ],
            "failure_criteria": [
                "positive reversal in the fresh panel",
                "effect < 0.02 CA-lDDT after controls",
                "effect materially explained by length, resolution, method, taxonomy, or charge/fold-composition controls",
            ],
            "heavy_step_budget": {
                "target_model": "biohub/ESMFold2 local checkpoint",
                "inference": "10 loops, 68 diffusion steps, 5 samples, best by pTM",
                "expected_wall_time": "about 15-30 minutes for 100 chains based on the prior 97-chain charge confirmation run",
                "timeout_or_fallback": "if interrupted/OOM, resume from partial CSV; if repeated OOM is isolated to long chains, mark failed chains and summarize scored panel only if every level retains n >= 15",
            },
            "claim_boundary_if_confirmed": "Fresh RCSB single-chain targets from May 2026 candidate pool, length 80-500 aa, resolution <=4.0 A, exact-sequence novel relative to accumulated ESMFold2 SPECTRA evidence; high sequence cysteine fraction as a prospective degradation boundary.",
        },
        "panel_summary": panel_summary(panel, eligible),
        "selection_diagnostics": diagnostics,
        "artifacts": {
            "eligible_unscored_cys_pool": str(OUT_ELIGIBLE),
            "cys_confirmation_candidate_features": str(OUT_PANEL),
        },
    }
    write_json(OUT_MANIFEST, manifest)
    update_state_and_ledger(manifest)
    print(json.dumps(manifest["panel_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
