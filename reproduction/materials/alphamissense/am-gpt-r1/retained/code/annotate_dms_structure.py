#!/usr/bin/env python3
"""Add frozen ProteinGym AF2 local pLDDT/disorder annotations to DMS rows."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data/processed/proteingym_dms_human_scored.parquet"
df = pd.read_parquet(path)
archive = ROOT / "data/raw/ProteinGym_v1.3_AF2_structures.zip"
scores = {}
missing = []
with zipfile.ZipFile(archive) as zf:
    names = set(zf.namelist())
    for entry in sorted(df.entry_name.unique()):
        member = f"ProteinGym_AF2_structures/{entry}.pdb"
        if member not in names:
            missing.append(entry)
            continue
        residue_scores = {}
        with zf.open(member) as handle:
            for raw in io.TextIOWrapper(handle):
                if not raw.startswith("ATOM") or raw[12:16].strip() != "CA":
                    continue
                try:
                    pos = int(raw[22:26])
                    plddt = float(raw[60:66])
                except ValueError:
                    continue
                residue_scores.setdefault(pos, plddt)
        scores[entry] = residue_scores
df["af2_plddt"] = [scores.get(e, {}).get(int(p), np.nan)
                    for e, p in zip(df.entry_name, df.aa_pos)]
df["af2_low_confidence_lt50"] = df.af2_plddt < 50
df["af2_confident_ge70"] = df.af2_plddt >= 70
df.to_parquet(path, index=False)
for part in ["discovery", "confirmation"]:
    df[df.protein_partition.eq(part)].to_parquet(
        ROOT / f"data/processed/proteingym_dms_{part}_scored.parquet", index=False)
summary = {
    "rows": len(df), "rows_with_plddt": int(df.af2_plddt.notna().sum()),
    "proteins": df.entry_name.nunique(),
    "proteins_with_structure": len(scores), "missing_structure_entries": missing,
}
(ROOT / "artifacts/tables/dms_structure_qc.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
