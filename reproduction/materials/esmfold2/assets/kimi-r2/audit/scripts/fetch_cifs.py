#!/usr/bin/env python
"""Download mmCIFs for sampled test entries and for the pool templates needed by
the template-copy reference. Source: https://files.rcsb.org/download/{ENTRY}.cif
"""
import csv, json, os, time, requests, sys

OUT = "audit/data/cif"
os.makedirs(OUT, exist_ok=True)
plan = sys.argv[1] if len(sys.argv) > 1 else "audit/data/sample_plan.csv"

axis = {r["entity"]: r for r in csv.DictReader(open("audit/data/axis_table.csv"))}
entries = set()
templates = set()
for row in csv.DictReader(open(plan)):
    e = row["entity"]
    entries.add(axis[e]["entry"])
    hit = axis[e]["hit"]
    if hit:
        templates.add(hit.split("_")[0])

S = requests.Session()
def fetch(entry):
    p = f"{OUT}/{entry}.cif"
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return True
    for attempt in range(3):
        try:
            r = S.get(f"https://files.rcsb.org/download/{entry}.cif", timeout=60)
            if r.status_code == 200:
                open(p, "wb").write(r.content)
                return True
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    print("FAIL", entry, flush=True)
    return False

ok = sum(fetch(e) for e in sorted(entries))
print(f"test entries: {ok}/{len(entries)}", flush=True)
ok2 = sum(fetch(t) for t in sorted(templates))
print(f"templates: {ok2}/{len(templates)}", flush=True)
