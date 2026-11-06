#!/usr/bin/env python3
"""
SPECTRA SPC: AlphaMissense pathogenicity performance vs evolutionary support
(MSA depth) on held-out ClinVar missense.

Axis (prospective, no target-model errors): per-protein MSA depth proxy =
number of distinct UniRef50 homologs at E<1e-3, query-cov>=0.5.
Lower MSA depth = lower pretraining/evolutionary support = lower similarity
to the well-supported regime -> SPECTRA expects degradation.

Fixed baseline (label-free, axis-independent): BLOSUM62 substitution score of
the amino-acid change. Comparing AM's curve to BLOSUM's isolates AM-specific
degradation from intrinsic task difficulty.
"""
import csv, json, math, collections
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

WORK = "/path/to/audit-home/spectra_alphamissense/work"
ART  = "/path/to/audit-home/spectra_alphamissense/artifacts"

# ---- BLOSUM62 (upper-tri symmetric), standard 20 AA ----
_B62 = """
A 4;R -1 5;N -2 0 6;D -2 -2 1 6;C 0 -3 -3 -3 9;Q -1 1 0 0 -3 5;E -1 0 0 2 -4 2 5;
G 0 -2 0 -1 -3 -2 -2 6;H -2 0 1 -1 -3 0 0 -2 8;I -1 -3 -3 -3 -1 -3 -3 -4 -3 4;
L -1 -2 -3 -4 -1 -2 -3 -4 -3 2 4;K -1 2 0 -1 -3 1 1 -2 -1 -3 -2 5;
M -1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5;F -2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6;
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7;S 1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4;
T 0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5;W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11;
Y -2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7;
V 0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4
"""
order = []
B62 = {}
for entry in _B62.replace("\n"," ").split(";"):
    toks = entry.split()
    if not toks: continue
    aa = toks[0]; vals = list(map(int, toks[1:]))
    order.append(aa)
    for j, v in enumerate(vals):
        B62[(aa, order[j])] = v
        B62[(order[j], aa)] = v
def blosum(a, b):
    return B62.get((a, b))

# ---- 1. MSA depth per protein ----
print("[1] computing MSA depth from msa_hits.m8 ...", flush=True)
depth = collections.defaultdict(set)
with open(f"{WORK}/msa_hits.m8") as fh:
    for line in fh:
        q, t = line.split("\t", 2)[:2]
        depth[q].add(t)
n_hom = {q: len(s) for q, s in depth.items()}
print(f"  proteins with hits: {len(n_hom)}", flush=True)

# ---- 2. load benchmark, attach depth + blosum ----
print("[2] loading benchmark ...", flush=True)
rows = []
with open(f"{WORK}/clinvar_am_benchmark.tsv") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        up = r["uniprot_id"]
        pv = r["protein_variant"]
        ref_aa, alt_aa = pv[0], pv[-1]
        if ref_aa not in order or alt_aa not in order:
            continue
        nh = n_hom.get(up, 0)
        rows.append({
            "uniprot": up,
            "label": int(r["label"]),
            "am": float(r["am_pathogenicity"]),
            "blosum": -blosum(ref_aa, alt_aa),  # negate: higher = more disruptive ~ pathogenic
            "stars": int(r["stars"]),
            "depth": nh,
            "logdepth": math.log10(1 + nh),
        })
print(f"  usable variants: {len(rows)}", flush=True)
# drop proteins with zero homologs (search failures / no query seq)
rows = [r for r in rows if r["depth"] > 0]
print(f"  variants in proteins with >=1 homolog: {len(rows)}", flush=True)

# ---- 3. assign protein-level MSA-depth deciles (SPC levels) ----
# bin by PROTEIN so a protein's variants share one level (avoids leakage of
# depth being correlated within protein); use protein-weighted quantiles.
prot_depth = {}
for r in rows:
    prot_depth[r["uniprot"]] = r["logdepth"]
prots = sorted(prot_depth, key=lambda p: prot_depth[p])
logs = np.array([prot_depth[p] for p in prots])
N_LEVELS = 10
# protein-count-balanced quantile edges on logdepth
qs = np.quantile(logs, np.linspace(0, 1, N_LEVELS + 1))
qs[0] -= 1e-9; qs[-1] += 1e-9
def level_of(ld):
    # level 0 = LOWEST depth (lowest similarity), level N-1 = highest
    return int(np.clip(np.searchsorted(qs, ld, side="right") - 1, 0, N_LEVELS - 1))
prot_level = {p: level_of(prot_depth[p]) for p in prots}
for r in rows:
    r["level"] = prot_level[r["uniprot"]]

# ---- 4. per-level metrics + bootstrap CIs over proteins ----
print("[3] per-level metrics ...", flush=True)
rng = np.random.default_rng(0)
by_level = collections.defaultdict(list)
for r in rows:
    by_level[r["level"]].append(r)

def auroc(items, key):
    y = [it["label"] for it in items]
    s = [it[key] for it in items]
    if len(set(y)) < 2: return None
    return roc_auc_score(y, s)

def boot_protein_auroc(items, key, B=500):
    # cluster bootstrap over proteins
    pmap = collections.defaultdict(list)
    for it in items: pmap[it["uniprot"]].append(it)
    plist = list(pmap)
    vals = []
    for _ in range(B):
        samp = rng.choice(len(plist), len(plist), replace=True)
        pool = []
        for idx in samp: pool.extend(pmap[plist[idx]])
        a = auroc(pool, key)
        if a is not None: vals.append(a)
    if not vals: return (None, None)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

levels_out = []
for lv in range(N_LEVELS):
    items = by_level[lv]
    nprot = len(set(it["uniprot"] for it in items))
    npos = sum(it["label"] for it in items)
    nneg = len(items) - npos
    depths = [it["depth"] for it in items]
    rec = {
        "level": lv,
        "label_desc": "lowest_MSA_depth" if lv == 0 else ("highest_MSA_depth" if lv == N_LEVELS-1 else ""),
        "n_variants": len(items),
        "n_proteins": nprot,
        "n_pathogenic": npos,
        "n_benign": nneg,
        "prevalence": round(npos / len(items), 4),
        "median_depth": int(np.median(depths)),
        "median_logdepth": round(float(np.median([it["logdepth"] for it in items])), 4),
        "am_auroc": auroc(items, "am"),
        "am_auprc": average_precision_score([it["label"] for it in items], [it["am"] for it in items]) if npos and nneg else None,
        "blosum_auroc": auroc(items, "blosum"),
    }
    lo, hi = boot_protein_auroc(items, "am")
    rec["am_auroc_ci95"] = [lo, hi]
    levels_out.append(rec)
    print(f"  L{lv}: nvar={rec['n_variants']:6d} nprot={nprot:5d} prev={rec['prevalence']:.3f} "
          f"medDepth={rec['median_depth']:5d} AM_auROC={rec['am_auroc']:.4f} "
          f"BLOSUM_auROC={rec['blosum_auroc']:.4f}", flush=True)

# overall
overall = {
    "n_variants": len(rows),
    "n_proteins": len(set(r["uniprot"] for r in rows)),
    "am_auroc_overall": auroc(rows, "am"),
    "blosum_auroc_overall": auroc(rows, "blosum"),
}
print("  overall:", overall, flush=True)

# ---- 5. monotonicity / similarity-gradient check ----
med_depths = [rec["median_depth"] for rec in levels_out]
am_aurocs = [rec["am_auroc"] for rec in levels_out]
# Spearman of level (depth order) vs AM auROC
from scipy.stats import spearmanr
rho_depth_auroc, p_depth = spearmanr(med_depths, am_aurocs)
# verify depth strictly increases with level (similarity gradient is real)
depth_monotonic = all(med_depths[i] < med_depths[i+1] for i in range(len(med_depths)-1))

summary = {
    "axis": "MSA_depth_uniref50_homolog_count",
    "axis_direction": "level 0 = lowest depth (lowest similarity/support)",
    "n_levels": N_LEVELS,
    "overall": overall,
    "levels": levels_out,
    "median_depth_monotonic_increasing": bool(depth_monotonic),
    "spearman_depth_vs_AM_auroc": {"rho": float(rho_depth_auroc), "p": float(p_depth)},
    "am_auroc_lowest_vs_highest": [am_aurocs[0], am_aurocs[-1]],
    "blosum_auroc_lowest_vs_highest": [levels_out[0]["blosum_auroc"], levels_out[-1]["blosum_auroc"]],
}
with open(f"{ART}/spc_msa_depth.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"[4] wrote {ART}/spc_msa_depth.json", flush=True)
print(json.dumps({k: summary[k] for k in ["median_depth_monotonic_increasing","spearman_depth_vs_AM_auroc","am_auroc_lowest_vs_highest","blosum_auroc_lowest_vs_highest"]}, indent=2))
