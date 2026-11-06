#!/usr/bin/env python3
"""Download mmCIF files for confirmation-set test entries and template entries."""
import csv, os, subprocess, sys, concurrent.futures as cf

os.makedirs("structures/test", exist_ok=True)
os.makedirs("structures/template", exist_ok=True)

rows = list(csv.DictReader(open("results/confirmation_set.tsv"), delimiter="\t"))
test_entries = sorted({r["unit"].split("_")[0] for r in rows})
templ_entries = sorted({r["template"].split("_")[0] for r in rows if r["template"] != "NONE"})

def fetch(args):
    entry, sub = args
    out = f"structures/{sub}/{entry}.cif"
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return (entry, sub, "cached")
    url = f"https://files.rcsb.org/download/{entry}.cif"
    r = subprocess.run(["curl", "-sfL", "--retry", "3", "-o", out, url],
                       capture_output=True)
    if r.returncode != 0 or not os.path.exists(out):
        return (entry, sub, "FAIL")
    return (entry, sub, "ok")

jobs = [(e, "test") for e in test_entries] + [(e, "template") for e in templ_entries]
fails = []
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    for i, (entry, sub, status) in enumerate(ex.map(fetch, jobs)):
        if status == "FAIL":
            fails.append((entry, sub))
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(jobs)}", file=sys.stderr)
print(f"done: {len(jobs)} files, {len(fails)} failures")
for f_ in fails:
    print("FAIL", f_)
