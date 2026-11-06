#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path("/ewsc/yektefai/spectra_runs/esmfold2_envelope_fusion_axis_confirmation_20260602")
OUT_DIR = ROOT / "spectra_ready_dataset"
STRUCT_DIR = OUT_DIR / "raw_rcsb_surface_lowhelical_confirmation" / "experiment_structures"
OUT_POOL = OUT_DIR / "surface_lowhelical_candidate_pool.csv"
OUT_PANEL = OUT_DIR / "surface_lowhelical_confirmation_candidate_features.csv"
OUT_MANIFEST = ROOT / "surface_lowhelical_confirmation_manifest.json"

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{entry}/{entity}"
ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{entry}"
CIF_URL = "https://files.rcsb.org/download/{entry}.cif"
PDB_URL = "https://files.rcsb.org/download/{entry}.pdb"
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
MAX_STRUCTURE_BYTES = 50_000_000

SCORED_TABLES = [
    Path("/ewsc/yektefai/spectra_runs/esmfold2_all_scored_evidence_loop_20260601T230100Z/spectra_ready_dataset/combined_evidence_with_cys_confirmation.csv"),
    Path("/ewsc/yektefai/spectra_runs/esmfold2_viral_disulfide_axis_pilot_20260602/target_model_results.viral_disulfide_pilot.target_level.csv"),
    ROOT / "target_model_results.envelope_fusion_confirmation.target_level.csv",
]

QUERIES = [
    "viral surface glycoprotein",
    "virus surface glycoprotein",
    "viral envelope protein",
    "viral envelope glycoprotein",
    "envelope protein E virus",
    "spike glycoprotein E2",
    "alphavirus E2",
    "alphavirus E1",
    "flavivirus envelope protein",
    "orthoflavivirus envelope protein",
    "tick-borne encephalitis envelope protein",
    "yellow fever virus envelope protein",
    "Japanese encephalitis envelope protein",
    "dengue envelope protein",
    "filovirus glycoprotein GP1",
    "coronavirus spike S1",
    "paramyxovirus fusion glycoprotein",
    "viral receptor binding glycoprotein",
    "viral glycoprotein ectodomain",
    "class II fusion protein virus",
    "viral capsid protein",
    "viral protease",
    "viral helicase",
    "viral polymerase",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()


def nglyc_count(seq: str) -> int:
    return len(re.findall(r"(?=N[^P][ST])", seq))


def disulfide_capacity_per100(seq: str) -> float:
    return (seq.count("C") // 2) / (len(seq) / 100.0)


def tm_proxy(seq: str) -> float:
    kd = {
        "A": 1.8,
        "C": 2.5,
        "D": -3.5,
        "E": -3.5,
        "F": 2.8,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "K": -3.9,
        "L": 3.8,
        "M": 1.9,
        "N": -3.5,
        "P": -1.6,
        "Q": -3.5,
        "R": -4.5,
        "S": -0.8,
        "T": -0.7,
        "V": 4.2,
        "W": -0.9,
        "Y": -1.3,
    }
    vals = [kd.get(aa, 0.0) for aa in seq]
    if len(vals) < 19:
        return sum(vals) / max(1, len(vals))
    return max(sum(vals[i : i + 19]) / 19.0 for i in range(len(vals) - 18))


def max_hydrophobic_window(seq: str, width: int = 21) -> float:
    hyd = set("AILMFWYV")
    if len(seq) <= width:
        return sum(aa in hyd for aa in seq) / max(1, len(seq))
    return max(sum(aa in hyd for aa in seq[i : i + width]) / width for i in range(len(seq) - width + 1))


def contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.I) is not None


MAX_ENTITY_IDS = 1200
TARGET_POOL_COUNTS = {0: 40, 1: 30, 2: 40}


def post_search(text: str, rows: int = 200) -> list[str]:
    payload = {
        "query": {"type": "terminal", "service": "full_text", "parameters": {"value": text}},
        "return_type": "polymer_entity",
        "request_options": {"paginate": {"start": 0, "rows": rows}},
    }
    resp = requests.post(SEARCH_URL, json=payload, timeout=60)
    resp.raise_for_status()
    time.sleep(0.05)
    return [item["identifier"] for item in resp.json().get("result_set", [])]


def get_json(url: str) -> dict:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    time.sleep(0.02)
    return resp.json()


def entry_metadata(entry: str) -> dict:
    data = get_json(ENTRY_URL.format(entry=entry))
    info = data.get("rcsb_entry_info", {})
    exptl = data.get("exptl", [])
    method = ";".join(e.get("method", "") for e in exptl) if isinstance(exptl, list) else ""
    resolution = info.get("resolution_combined")
    if isinstance(resolution, list) and resolution:
        resolution = resolution[0]
    elif not isinstance(resolution, (int, float)):
        resolution = math.nan
    return {
        "method": method,
        "resolution": resolution,
        "release_date": data.get("rcsb_accession_info", {}).get("initial_release_date", ""),
        "title": data.get("struct", {}).get("title", ""),
        "keywords": " ".join(data.get("struct_keywords", {}).values()) if isinstance(data.get("struct_keywords"), dict) else "",
    }


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
    cif_path = STRUCT_DIR / f"{target_id}.cif"
    if download_text(CIF_URL.format(entry=entry), cif_path, ("data_", "#")) and cif_path.stat().st_size <= MAX_STRUCTURE_BYTES:
        return True, cif_path, "cif"
    pdb_path = STRUCT_DIR / f"{target_id}.pdb"
    if download_text(PDB_URL.format(entry=entry), pdb_path, ("HEADER", "TITLE", "COMPND", "REMARK", "ATOM")) and pdb_path.stat().st_size <= MAX_STRUCTURE_BYTES:
        return True, pdb_path, "pdb"
    return False, None, "failed"


def load_scored() -> tuple[set[str], set[str]]:
    hashes: set[str] = set()
    targets: set[str] = set()
    for path in SCORED_TABLES:
        if not path.exists():
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("sequence_hash"):
                    hashes.add(row["sequence_hash"])
                if row.get("target_id"):
                    targets.add(row["target_id"])
    return hashes, targets


def axis_features(row: dict) -> dict:
    text = " ".join(str(row.get(c, "")) for c in ["organism", "entity_description", "title", "keywords"]).lower()
    surface = contains(text, r"\b(envelope|spike|glycoprotein|fusion|gp1|gp2|gp350|protein e|e1|e2)\b")
    capsid_or_enzyme = contains(text, r"\b(capsid|protease|helicase|polymerase|nsp5|ns2b|ns3)\b")
    disulfide_dense = float(row["disulfide_capacity_per100"]) >= 1.2
    low_helical = float(row["tm_helix_proxy_calc"]) < 2.55
    if (not surface) or capsid_or_enzyme:
        level = 0
    elif disulfide_dense and low_helical:
        level = 2
    else:
        level = 1
    return {
        "surface_keyword": int(surface),
        "capsid_or_enzyme_keyword": int(capsid_or_enzyme),
        "surface_lowhelical_disulfide_dense": int(disulfide_dense),
        "surface_lowhelical_low_tm_proxy": int(low_helical),
        "surface_lowhelical_score": level,
        "surface_lowhelical_level_name": {
            0: "L0_non_surface_or_capsid_enzyme_control",
            1: "L1_surface_without_disulfide_lowhelical_combo",
            2: "L2_surface_disulfide_dense_lowhelical",
        }[level],
    }


def select_level(rows: list[dict], n: int, max_per_org: int = 2) -> list[dict]:
    rows = sorted(rows, key=lambda r: (r["release_date"], float(r["resolution"]), r["organism"], r["target_id"]), reverse=True)
    selected: list[dict] = []
    org_counts: Counter[str] = Counter()
    seq_seen: set[str] = set()
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
    scored_hashes, scored_targets = load_scored()
    query_to_ids: list[tuple[str, list[str]]] = []
    query_hits: defaultdict[str, list[str]] = defaultdict(list)
    for query in QUERIES:
        try:
            ids = post_search(query)
            query_to_ids.append((query, ids))
            for ident in ids:
                query_hits[ident].append(query)
            print(json.dumps({"query": query, "hits": len(ids)}), flush=True)
        except Exception as exc:
            print(json.dumps({"query": query, "error": str(exc)}), flush=True)

    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    max_len = max((len(ids) for _query, ids in query_to_ids), default=0)
    for idx in range(max_len):
        for _query, ids in query_to_ids:
            if idx >= len(ids):
                continue
            ident = ids[idx]
            if ident in seen_ids:
                continue
            seen_ids.add(ident)
            ordered_ids.append(ident)
            if len(ordered_ids) >= MAX_ENTITY_IDS:
                break
        if len(ordered_ids) >= MAX_ENTITY_IDS:
            break
    print(json.dumps({"unique_entity_ids_for_metadata": len(ordered_ids), "cap": MAX_ENTITY_IDS}), flush=True)

    entry_cache: dict[str, dict] = {}
    rows: list[dict] = []
    for i, ident in enumerate(ordered_ids, start=1):
        if i % 100 == 0:
            print(
                json.dumps(
                    {
                        "processed": i,
                        "eligible_rows": len(rows),
                        "level_counts": dict(Counter(int(r["surface_lowhelical_score"]) for r in rows)),
                    }
                ),
                flush=True,
            )
        counts = Counter(int(r["surface_lowhelical_score"]) for r in rows)
        if all(counts[level] >= TARGET_POOL_COUNTS[level] for level in [0, 1, 2]):
            print(json.dumps({"early_stop": "target_pool_counts_reached", "processed": i, "level_counts": dict(counts)}), flush=True)
            break
        try:
            entry, entity = ident.split("_", 1)
            entity_data = get_json(ENTITY_URL.format(entry=entry, entity=entity))
            poly = entity_data.get("entity_poly", {})
            seq = (poly.get("pdbx_seq_one_letter_code_can") or poly.get("pdbx_seq_one_letter_code") or "").replace("\n", "").upper()
            if not seq or len(seq) < 80 or len(seq) > 500 or not set(seq).issubset(AA20):
                continue
            seq_hash = sha(seq)
            if seq_hash in scored_hashes or f"{entry}_{entity}" in scored_targets:
                continue
            source_org = entity_data.get("rcsb_entity_source_organism", [])
            organism = ""
            if isinstance(source_org, list) and source_org:
                organism = source_org[0].get("ncbi_scientific_name") or source_org[0].get("scientific_name") or ""
            org_l = organism.lower()
            if "virus" not in org_l and "phage" not in org_l:
                continue
            if entry not in entry_cache:
                entry_cache[entry] = entry_metadata(entry)
            meta = entry_cache[entry]
            try:
                resolution = float(meta["resolution"])
            except Exception:
                resolution = math.nan
            if not math.isfinite(resolution) or resolution > 4.0:
                continue
            if str(meta.get("release_date", ""))[:4] < "2024":
                continue
            desc = entity_data.get("rcsb_polymer_entity", {}).get("pdbx_description") or entity_data.get("pdbx_description") or ""
            row = {
                "rcsb_entity_id": ident,
                "entry_id": entry,
                "entity_id": entity,
                "target_id": f"{entry}_{entity}",
                "sequence": seq,
                "sequence_hash": seq_hash,
                "seq_len": len(seq),
                "organism": organism,
                "method": meta["method"],
                "resolution": resolution,
                "release_date": meta["release_date"],
                "entity_description": desc,
                "title": meta["title"],
                "keywords": meta["keywords"],
                "query_hits": ";".join(query_hits[ident]),
                "cys_fraction": seq.count("C") / len(seq),
                "disulfide_capacity_per100": disulfide_capacity_per100(seq),
                "tm_helix_proxy_calc": tm_proxy(seq),
                "nglyc_motif_count": nglyc_count(seq),
                "hydrophobic_window21_max_frac_calc": max_hydrophobic_window(seq),
            }
            row.update(axis_features(row))
            rows.append(row)
        except Exception as exc:
            print(json.dumps({"identifier": ident, "error": str(exc)[:200]}), flush=True)

    with OUT_POOL.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    targets = {0: 16, 1: 16, 2: 16}
    selected: list[dict] = []
    diagnostics = []
    for level in [2, 1, 0]:
        part = [r for r in rows if int(r["surface_lowhelical_score"]) == level]
        chosen = select_level(part, targets[level])
        packaged = []
        failures = 0
        for row in chosen:
            try:
                chain = first_chain_for_entity(row["entry_id"], row["entity_id"])
                target_id = f"{row['entry_id']}_{row['entity_id']}_{chain}"
                ok, path, fmt = download_structure(row["entry_id"], target_id)
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
                out["ref_pdb_id"] = row["entry_id"]
                out["metadata_sequence_length"] = row["seq_len"]
                out["msa_rows"] = ""
                out["msa_nonquery_rows"] = ""
                out["msa_log10_nonquery_rows"] = ""
                out["msa_top_nonquery_identity"] = ""
                out["scaling_group"] = out["surface_lowhelical_level_name"]
                packaged.append(out)
            except Exception:
                failures += 1
        selected.extend(packaged)
        diagnostics.append({"level": level, "eligible": len(part), "chosen": len(chosen), "packaged": len(packaged), "failures": failures})

    selected = sorted(selected, key=lambda r: (int(r["surface_lowhelical_score"]), int(r["seq_len"]), r["target_id"]))
    with OUT_PANEL.open("w", newline="") as handle:
        if selected:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()))
            writer.writeheader()
            writer.writerows(selected)

    summary = []
    for level in [0, 1, 2]:
        part = [r for r in selected if int(r["surface_lowhelical_score"]) == level]
        summary.append(
            {
                "level": level,
                "n": len(part),
                "organisms": dict(Counter(r["organism"] for r in part)),
                "descriptions": dict(Counter(r["entity_description"] for r in part)),
                "seq_len_median": sorted(float(r["seq_len"]) for r in part)[len(part) // 2] if part else None,
                "resolution_median": sorted(float(r["resolution"]) for r in part)[len(part) // 2] if part else None,
            }
        )
    manifest = {
        "created_at_utc": now(),
        "axis_id": "viral_surface_disulfide_dense_lowhelical_confirmation",
        "axis_definition": {
            "scope": "viral RCSB polymer entities, release year >= 2024, exact sequences excluded from all prior scored evidence in this session",
            "level_0": "not a viral surface keyword target, or capsid/enzyme keyword present",
            "level_1": "viral surface keyword target, not capsid/enzyme, but not both disulfide-dense and low-helical/TM-proxy",
            "level_2": "viral surface keyword target, not capsid/enzyme, disulfide_capacity_per100 >= 1.2, and tm_helix_proxy_calc < 2.55",
            "surface_keywords": "envelope|spike|glycoprotein|fusion|gp1|gp2|gp350|protein e",
            "capsid_enzyme_keywords": "capsid|protease|helicase|polymerase|nsp5|ns2b|ns3",
        },
        "query_count": len(QUERIES),
        "pool_rows": len(rows),
        "pool_level_counts": dict(Counter(int(r["surface_lowhelical_score"]) for r in rows)),
        "panel_summary": summary,
        "diagnostics": diagnostics,
        "artifacts": {"pool": str(OUT_POOL), "panel": str(OUT_PANEL)},
        "validity_note": "This is frozen confirmation of an outcome-informed candidate axis discovered from the envelope/fusion confirmation panel.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pool_rows": len(rows), "panel_n": len(selected), "summary": summary, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
