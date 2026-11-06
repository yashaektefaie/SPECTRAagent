#!/usr/bin/env python3
"""Fetch RCSB metadata for SPECTRA ESMFold audit.

Outputs:
  data/post_cutoff_entries.txt   : entry IDs with initial release > 2022-03-01
  data/test_pool_entities.tsv    : test-window polymer entities (entry, entity,
                                   auth chains, length, resolution)
"""
import json, urllib.request, sys

API = "https://search.rcsb.org/rcsbsearch/v2/query"

def query(payload):
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            body = getattr(e, "read", lambda: b"")()
            print(f"  retry {attempt}: {e} {body[:300]}", file=sys.stderr)
    raise RuntimeError("query failed")

# ---- 1. All entries released after the pre-cutoff (for corpus exclusion) ----
post_ids = []
page = 0
while True:
    payload = {
        "query": {"type": "terminal", "service": "text",
                  "parameters": {"attribute": "rcsb_accession_info.initial_release_date",
                                 "operator": "greater",
                                 "value": "2022-03-01T00:00:00Z"}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": page * 10000, "rows": 10000},
                            "results_content_type": ["experimental"]},
    }
    res = query(payload)
    ids = [x["identifier"] for x in res.get("result_set", [])]
    post_ids.extend(ids)
    total = res.get("total_count", 0)
    print(f"post-cutoff page {page}: {len(ids)} (total {total})", file=sys.stderr)
    if len(post_ids) >= total or not ids:
        break
    page += 1
with open("data/post_cutoff_entries.txt", "w") as f:
    f.write("\n".join(sorted(set(post_ids))) + "\n")
print(f"post-cutoff entries: {len(set(post_ids))}", file=sys.stderr)

# ---- 2. Test pool: entries released in the frozen test window ----
payload = {
    "query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text",
         "parameters": {"attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "greater", "value": "2022-12-31T23:59:59Z"}},
        {"type": "terminal", "service": "text",
         "parameters": {"attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "less", "value": "2023-07-01T00:00:00Z"}},
        {"type": "terminal", "service": "text",
         "parameters": {"attribute": "exptl.method",
                        "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
        {"type": "terminal", "service": "text",
         "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal", "value": 3.0}},
        {"type": "terminal", "service": "text",
         "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater", "value": 0}},
    ]},
    "return_type": "entry",
    "request_options": {"paginate": {"start": 0, "rows": 10000},
                        "results_content_type": ["experimental"]},
}
rows = []
page = 0
while True:
    payload["request_options"]["paginate"]["start"] = page * 10000
    res = query(payload)
    got = [x["identifier"] for x in res.get("result_set", [])]
    rows.extend(got)
    total = res.get("total_count", 0)
    print(f"test-window page {page}: {len(got)} (total {total})", file=sys.stderr)
    if len(rows) >= total or not got:
        break
    page += 1
with open("data/test_pool_entries_raw.txt", "w") as f:
    f.write("\n".join(sorted(set(rows))) + "\n")
print(f"wrote {len(set(rows))} test-window entry ids", file=sys.stderr)
