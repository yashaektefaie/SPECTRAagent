#!/usr/bin/env python3
"""Step-1 bookkeeping recheck (no new science): recompute the frozen-file confounds
   - distinct proteins per bin
   - single-protein max residue fraction (q0 & insert hard bins)
   - internal-vs-terminal fractions (coarse-hard / q0 / insert / deep-and-gappy)
over the SCORED panel by EXCLUDING UBC9 (P63279). UBC9 is in the frozen file (158 positions)
but was dropped at scoring (OFFSET_FAIL, 0 scored; 158-res AM canonical vs 159-res ProteinGym),
so the published confounds (computed over the frozen file) are over 73 assays while the scored
curve is over 72. This reconciles them.

Replicates terminal_check.py predicates + (accession,target_pos) dedup EXACTLY, then runs each
bin twice: OLD = all 73 (incl. UBC9), NEW = 72 (excl. P63279). READ-ONLY on inputs; writes one
NEW txt artifact. No Spearman / no scores recomputed.
"""
import csv
from collections import defaultdict

BASE = "/path/to/audit-home/spectra_alphamissense"
OUT  = f"{BASE}/stepB"
PG   = f"{BASE}/proteingym"
TERM = 20
EXCLUDE_ACC = "P63279"  # UBC9_HUMAN

# MSA span per assay (terminal = within TERM of MSA span ends) -- identical to terminal_check.py
span = {}
with open(f"{PG}/kept_assays_verified.tsv") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        def gi(x):
            try:    return int(float(r.get(x, "")))
            except: return None
        span[r["DMS_id"]] = (gi("MSA_start"), gi("MSA_end"))

rows = []
with open(f"{OUT}/axis3_pathC_percolumn_FROZEN.tsv") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        rows.append(r)

def classify(assay, pos):
    s, e = span.get(assay, (None, None))
    if s is None or e is None: return "unk"
    if (pos - s) < TERM or (e - pos) < TERM: return "terminal"
    return "internal"

BINS = [
    ("COARSE HARD (insert + occ<=0.9)",
     lambda r: r["low_cov_insert"] == "1" or (r["low_cov_insert"] == "0" and float(r["occupancy"]) <= 0.9)),
    ("OCC q0 aligned (occ<=0.945)",
     lambda r: r["low_cov_insert"] == "0" and float(r["occupancy"]) <= 0.945),
    ("INSERT-STATE only",
     lambda r: r["low_cov_insert"] == "1"),
    ("DEEP(Neff>=200) & occ<0.7",
     lambda r: r["low_cov_insert"] == "0" and float(r["occupancy"]) < 0.7 and float(r["neff_theta02"]) >= 200),
]

def compute(predicate, exclude):
    seen = set(); ti = defaultdict(int); per_prot = defaultdict(int)
    for r in rows:
        if exclude and r["accession"] == EXCLUDE_ACC: continue
        if not predicate(r): continue
        k = (r["accession"], int(r["target_pos"]))
        if k in seen: continue
        seen.add(k)
        ti[classify(r["assay"], int(r["target_pos"]))] += 1
        per_prot[r["accession"]] += 1
    tot = sum(ti.values())
    return dict(
        n=tot,
        nprot=len(per_prot),
        top=(max(per_prot.values()) / tot) if tot else 0.0,
        internal=(ti["internal"] / tot) if tot else 0.0,
        terminal=(ti["terminal"] / tot) if tot else 0.0,
        unk=(ti["unk"] / tot) if tot else 0.0,
    )

lines = []
def P(s): lines.append(s); print(s)

P("Step-1 confound recheck -- OLD (73, incl UBC9/P63279) vs NEW (72, excl UBC9)")
P("residue unit = distinct (accession,target_pos); terminal = within %d res of MSA span ends" % TERM)
P("method mirrors terminal_check.py exactly; no Spearman / no scores touched")
P("=" * 78)
for name, pred in BINS:
    o = compute(pred, False); n = compute(pred, True)
    P("")
    P(name)
    P("  %-30s %14s %16s" % ("metric", "OLD (73)", "NEW (72)"))
    P("  %-30s %14d %16d"   % ("n_resid (distinct prot-pos)", o["n"], n["n"]))
    P("  %-30s %14d %16d"   % ("distinct proteins", o["nprot"], n["nprot"]))
    P("  %-30s %13.1f%% %15.1f%%" % ("single-protein max frac", 100*o["top"], 100*n["top"]))
    P("  %-30s %13.1f%% %15.1f%%" % ("internal fraction", 100*o["internal"], 100*n["internal"]))
    P("  %-30s %13.1f%% %15.1f%%" % ("terminal fraction", 100*o["terminal"], 100*n["terminal"]))

txt = "\n".join(lines) + "\n"
open(f"{OUT}/confound_recheck_72.txt", "w").write(txt)
P("")
P("[written] stepB/confound_recheck_72.txt")
