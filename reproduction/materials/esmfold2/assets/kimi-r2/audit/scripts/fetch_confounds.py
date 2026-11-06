#!/usr/bin/env python
"""Fetch entry-level confound metadata for all full-plan entries:
protein entity count, total polymer entity count, chain instance count."""
import csv, json, os, time, requests

entries = set()
for r in csv.DictReader(open("audit/data/sample_plan_full.csv")):
    entries.add(r["entity"].split("_")[0])
out = {}
S = requests.Session()
for e in sorted(entries):
    p = f"audit/data/entry_meta/{e}.json"
    os.makedirs("audit/data/entry_meta", exist_ok=True)
    if not os.path.exists(p):
        for attempt in range(3):
            try:
                r = S.get(f"https://data.rcsb.org/rest/v1/core/entry/{e}", timeout=30)
                if r.status_code == 200:
                    json.dump(r.json(), open(p, "w"))
                    break
            except Exception:
                pass
            time.sleep(2 * (attempt + 1))
    d = json.load(open(p))
    i = d["rcsb_entry_info"]
    out[e] = {
        "poly_ent": i["polymer_entity_count"],
        "prot_ent": i.get("polymer_entity_count_protein"),
        "res": (i.get("resolution_combined") or [None])[0],
        "title": d["struct"]["title"][:100],
    }
json.dump(out, open("audit/data/entry_confounds.json", "w"), indent=1)
print("entries:", len(out))
