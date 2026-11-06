#!/usr/bin/env python3
"""Fetch accession mappings for the pinned ProteinGym human DMS entry names."""
from __future__ import annotations

import csv
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ref = pd.read_csv(ROOT / "model/ProteinGym/reference_files/DMS_substitutions.csv")
ids = sorted(ref.loc[ref.taxon.eq("Human"), "UniProt_ID"].unique())
rows = []
urls = []
for start in range(0, len(ids), 20):
    batch = ids[start:start + 20]
    query = " OR ".join(f"id:{x}" for x in batch)
    params = urllib.parse.urlencode({"query": query, "format": "tsv", "fields": "accession,id"})
    url = "https://rest.uniprot.org/uniprotkb/search?" + params
    urls.append(url)
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode()
    rows.extend(csv.DictReader(io.StringIO(text), delimiter="\t"))
    time.sleep(0.2)
out = pd.DataFrame(rows).drop_duplicates()
out.to_csv(ROOT / "metadata/proteingym_human_uniprot_mapping.tsv", sep="\t", index=False)
(ROOT / "provenance/uniprot_mapping_request.json").write_text(json.dumps({
    "retrieved_at_utc": "2026-07-14",
    "endpoint": "https://rest.uniprot.org/uniprotkb/search",
    "fields": ["accession", "id"],
    "requested_entry_names": ids,
    "request_urls": urls,
    "returned_rows": len(out),
}, indent=2) + "\n")
print(out.to_string(index=False))
