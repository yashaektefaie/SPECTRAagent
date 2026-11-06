#!/usr/bin/env python3
"""Download mmCIF for all template entries of the expanded set."""
import os, subprocess, sys, concurrent.futures as cf

os.makedirs("structures/template", exist_ok=True)
entries = open("data/template_entries_all.txt").read().split()

def fetch(entry):
    out = f"structures/template/{entry}.cif"
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return (entry, "cached")
    r = subprocess.run(["curl", "-sfL", "--retry", "3", "-o", out,
                        f"https://files.rcsb.org/download/{entry}.cif"],
                       capture_output=True)
    return (entry, "ok" if r.returncode == 0 and os.path.exists(out) else "FAIL")

fails = []
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    for i, (e, s) in enumerate(ex.map(fetch, entries)):
        if s == "FAIL":
            fails.append(e)
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(entries)}", file=sys.stderr)
print(f"done, {len(fails)} failures")
for e in fails:
    print("FAIL", e)
