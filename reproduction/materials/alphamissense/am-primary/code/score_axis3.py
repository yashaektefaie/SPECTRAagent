#!/usr/bin/env python3
"""SPECTRA Axis-3 AM scoring (Path A, occupancy axis).
Join AM pathogenicity -> DMS single-mutant fitness for the 73-assay panel, then compute
per-assay |Spearman(AM, DMS)| WITHIN each occupancy bin and aggregate across assays.
Bins are FROZEN (stepB/axis3_pathC_percolumn_FROZEN.tsv) -- no model output in bin defs.
Accuracy metric = |Spearman| (AM predicts pathogenicity = anti-fitness; magnitude = accuracy,
sign aligned within-assay across bins). New files only; no overwrites. NO figure."""
import csv, gzip, statistics
from collections import defaultdict
import numpy as np

BASE="/path/to/audit-home/spectra_alphamissense"
PG=f"{BASE}/proteingym"; STEPB=f"{BASE}/stepB"
AMFILE=f"{BASE}/data/AlphaMissense_aa_substitutions.tsv.gz"
MIN_VAR=20  # min single-mutant variants per (assay,bin) for a stable Spearman

# ---------- tie-aware Spearman (numpy only) ----------
def rankavg(a):
    a=np.asarray(a,float); n=len(a)
    order=np.argsort(a,kind="mergesort")
    ranks=np.empty(n); ranks[order]=np.arange(n,dtype=float)
    sa=a[order]; i=0
    while i<n:
        j=i
        while j+1<n and sa[j+1]==sa[i]: j+=1
        if j>i: ranks[order[i:j+1]]=(i+j)/2.0
        i=j+1
    return ranks
def spearman(x,y):
    if len(x)<3: return float("nan")
    rx=rankavg(x); ry=rankavg(y); rx=rx-rx.mean(); ry=ry-ry.mean()
    den=np.sqrt((rx*rx).sum()*(ry*ry).sum())
    return float((rx*ry).sum()/den) if den>0 else float("nan")

# ---------- 1. panel ----------
assays={}; accs=set()
with open(f"{PG}/kept_assays_verified.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        acc=r["resolved_accession"].strip()
        assays[r["DMS_id"]]={"acc":acc,"tseq":r["target_seq"].strip(),
                             "dms":r["DMS_filename"].strip(),"gene":r.get("UniProt_ID","").strip()}
        accs.add(acc)
print(f"panel: {len(assays)} assays, {len(accs)} distinct accessions")

# ---------- 2. AM lookup (stream gz, keep only our accessions) ----------
am=defaultdict(dict); amwt=defaultdict(dict)
nrows=0
with gzip.open(AMFILE,"rt") as fh:
    for line in fh:
        if line[0]=="#": continue
        if line[:10]=="uniprot_id": continue
        p=line.rstrip("\n").split("\t")
        if len(p)<3: continue
        acc=p[0]
        if acc not in accs: continue
        var=p[1]; wt=var[0]; mut=var[-1]
        try: pos=int(var[1:-1])
        except: continue
        am[acc][(pos,mut)]=float(p[2]); amwt[acc][pos]=wt; nrows+=1
print(f"AM rows kept for our accessions: {nrows}; accessions with AM data: {len(am)}/{len(accs)}")
missing_am=[a for a in accs if a not in am]
if missing_am: print(f"  accessions with NO AM data: {missing_am}")

def build_canon(acc):
    d=amwt.get(acc,{})
    if not d: return ""
    L=max(d)
    return "".join(d.get(i,"X") for i in range(1,L+1))

# ---------- 3. join AM <- DMS single mutants ----------
records=[]            # (assay, acc, pos, dms, am)
cov=[]                # per-assay coverage diagnostics
for aid,info in assays.items():
    acc=info["acc"]; tseq=info["tseq"]
    canon=build_canon(acc)
    offset=canon.find(tseq) if (canon and tseq) else -1
    n_single=n_scored=n_offfail=n_wtmis=n_nolookup=0
    try:
        fh=open(f"{PG}/dms_final_73/{info['dms']}")
    except FileNotFoundError:
        cov.append((aid,acc,0,0,-1,0,0,"FILE_MISSING")); continue
    for r in csv.DictReader(fh):
        mut=r["mutant"].strip()
        if ":" in mut: continue
        n_single+=1
        wt=mut[0]; mt=mut[-1]
        try: pos=int(mut[1:-1]); dms=float(r["DMS_score"])
        except: continue
        if offset<0: n_offfail+=1; continue
        cpos=offset+pos
        if amwt[acc].get(cpos)!=wt: n_wtmis+=1; continue
        s=am[acc].get((cpos,mt))
        if s is None: n_nolookup+=1; continue
        n_scored+=1; records.append((aid,acc,pos,dms,s))
    fh.close()
    status="OK" if offset>=0 else "OFFSET_FAIL"
    cov.append((aid,acc,n_single,n_scored,n_offfail,n_wtmis,n_nolookup,status))

tot_single=sum(c[2] for c in cov); tot_scored=sum(c[3] for c in cov)
print(f"\nJOIN: total single mutants={tot_single}  AM-scored={tot_scored} ({tot_scored/tot_single:.1%})")

# ---------- 4. frozen bins ----------
binmap={}
with open(f"{STEPB}/axis3_pathC_percolumn_FROZEN.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        ins=r["low_cov_insert"]=="1"
        occ=r["occupancy"].strip(); neff=r["neff_theta02"].strip()
        binmap[(r["assay"],int(r["target_pos"]))]={
            "occ":(float(occ) if occ else float("nan")),"insert":ins,
            "neff":(float(neff) if neff else float("nan"))}

n_scored_binned=sum(1 for (aid,acc,pos,dms,s) in records if (aid,pos) in binmap)
print(f"AM-scored variants WITH a frozen bin: {n_scored_binned}/{tot_scored}")

# ---------- 5. aggregation engine ----------
def aggregate(selector):
    per={}
    for (aid,acc,pos,dms,s) in records:
        b=binmap.get((aid,pos))
        if b is None or not selector(b): continue
        per.setdefault(aid,[]).append((s,dms))
    rows=[]
    for aid,pairs in per.items():
        if len(pairs)<MIN_VAR: continue
        a=[p[0] for p in pairs]; d=[p[1] for p in pairs]
        rho=spearman(a,d)
        if rho==rho: rows.append((aid,assays[aid]["acc"],len(pairs),abs(rho),rho))
    if not rows:
        return {"n_assays":0,"n_prot":0,"n_var":0,"wmean":float("nan"),
                "umean":float("nan"),"median":float("nan"),"rows":[]}
    tv=sum(r[2] for r in rows)
    return {"n_assays":len(rows),"n_prot":len(set(r[1] for r in rows)),"n_var":tv,
            "wmean":sum(r[2]*r[3] for r in rows)/tv,
            "umean":statistics.mean(r[3] for r in rows),
            "median":statistics.median(r[3] for r in rows),"rows":rows}

# selectors
def s_coarse_hard(b): return b["insert"] or (b["occ"]<=0.9)
def s_coarse_easy(b): return (not b["insert"]) and b["occ"]>0.9
QE=[0.945,0.987,0.998,0.999]
def qbin(occ):
    return min(sum(1 for e in QE if occ>e),4)
def s_q(k):
    return lambda b: (not b["insert"]) and qbin(b["occ"])==k
def s_insert(b): return b["insert"]
def s_deep_lowocc(thr):
    return lambda b: (not b["insert"]) and b["neff"]>=200 and b["occ"]<thr
def s_deep_highocc(b): return (not b["insert"]) and b["neff"]>=200 and b["occ"]>=0.9

coarse_hard=aggregate(s_coarse_hard); coarse_easy=aggregate(s_coarse_easy)
qres={k:aggregate(s_q(k)) for k in range(5)}
insert_res=aggregate(s_insert)
d05=aggregate(s_deep_lowocc(0.5)); d07=aggregate(s_deep_lowocc(0.7)); dhi=aggregate(s_deep_highocc)
overall=aggregate(lambda b: True)

# ---------- 6. write report ----------
def fmt(d): return f"{d['wmean']:.3f}" if d['n_assays'] else "n/a"
def line(lbl,d):
    if not d["n_assays"]: return f"| {lbl} | 0 | 0 | 0 | n/a | n/a | n/a |"
    return (f"| {lbl} | {d['n_assays']} | {d['n_prot']} | {d['n_var']} | "
            f"{d['wmean']:.3f} | {d['umean']:.3f} | {d['median']:.3f} |")

gap=(coarse_easy["wmean"]-coarse_hard["wmean"]) if (coarse_hard["n_assays"] and coarse_easy["n_assays"]) else float("nan")

out=[]
out.append("# Step B — Axis-3 AM scoring results (Path A, occupancy axis)\n")
out.append("**Date:** 2026-06-10. AM AlphaMissense_aa_substitutions.tsv.gz joined to ProteinGym v1.3 "
           "DMS single mutants (73 assays). Bins FROZEN before scoring "
           "(`stepB/axis3_pathC_percolumn_FROZEN.tsv`). Accuracy metric = **|Spearman(AM, DMS_score)|** "
           "per (assay,bin); aggregated across assays. `wmean` = variant-count-weighted mean |rho|; "
           f"`umean` = unweighted mean; `median` = median. Min variants/(assay,bin) = {MIN_VAR}.\n")
out.append("## 0. Join coverage\n")
out.append(f"- Total single mutants (no-colon) across 73 assays: **{tot_single}**")
out.append(f"- AM-scored: **{tot_scored}** ({tot_scored/tot_single:.1%})")
out.append(f"- AM-scored AND within a frozen occupancy bin (usable for SPC): **{n_scored_binned}**")
if missing_am: out.append(f"- Accessions with NO AM data: {missing_am}")
out.append(f"- Overall |Spearman| (all binned variants, no split): wmean={overall['wmean']:.3f} "
           f"over {overall['n_assays']} assays / {overall['n_prot']} proteins / {overall['n_var']} variants\n")
# poor-coverage assays
poor=sorted([c for c in cov if c[2]>0 and c[3]/c[2]<0.90], key=lambda c:c[3]/c[2])
out.append("### Assays with <90% AM join coverage")
if poor:
    out.append("| assay | acc | n_single | n_scored | cov | offfail | wt_mismatch | no_AM_var | status |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for c in poor:
        out.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[3]/c[2]:.0%} | {c[4]} | {c[5]} | {c[6]} | {c[7]} |")
else:
    out.append("None — all 73 assays ≥90% AM join coverage.")
out.append("")

out.append("## 1. Coarse headline — hard vs easy (occupancy)\n")
out.append("Hard = insert-state OR aligned occ≤0.9 ; Easy = aligned occ>0.9.\n")
out.append("| bin | n_assays | n_proteins | n_variants | wmean \\|rho\\| | umean | median |")
out.append("|---|---|---|---|---|---|---|")
out.append(line("HARD (insert + occ≤0.9)",coarse_hard))
out.append(line("EASY (occ>0.9)",coarse_easy))
out.append("")
out.append(f"**Hard-vs-easy gap (easy_wmean − hard_wmean) = {gap:+.3f}**  "
           f"(positive = AM less accurate on hard columns).\n")

out.append("## 2. Quintile resolution — 5 occupancy bins (aligned) + insert-state\n")
out.append("| occ bin | range | n_assays | n_proteins | n_variants | wmean \\|rho\\| | umean | median |")
out.append("|---|---|---|---|---|---|---|---|")
rngs=["occ≤0.945","0.945–0.987","0.987–0.998","0.998–0.999","occ>0.999 (~1.0)"]
for k in range(5):
    d=qres[k]
    if d["n_assays"]:
        out.append(f"| q{k} | {rngs[k]} | {d['n_assays']} | {d['n_prot']} | {d['n_var']} | "
                   f"{d['wmean']:.3f} | {d['umean']:.3f} | {d['median']:.3f} |")
    else:
        out.append(f"| q{k} | {rngs[k]} | 0 | 0 | 0 | n/a | n/a | n/a |")
out.append(line("insert-state (hardest)",insert_res))
out.append("")

out.append("## 3. Interaction — deep (Neff≥200) AND low-occupancy vs deep high-occupancy\n")
out.append("Tests 'hard-to-align ≠ shallow': among columns with abundant homologs, does gappiness still degrade AM?\n")
out.append("| cell | n_assays | n_proteins | n_variants | wmean \\|rho\\| | umean | median |")
out.append("|---|---|---|---|---|---|---|")
out.append(line("DEEP & occ<0.5",d05))
out.append(line("DEEP & occ<0.7",d07))
out.append(line("DEEP & occ≥0.9 (reference)",dhi))
out.append("")
if dhi["n_assays"] and d07["n_assays"]:
    out.append(f"**Interaction gap (deep-high − deep&occ<0.7) = {dhi['wmean']-d07['wmean']:+.3f}**\n")

# per-assay dump for the coarse split (transparency)
out.append("## 4. Per-assay |rho| for the coarse split (transparency)\n")
out.append("| assay | acc | n_hard | rho_hard | n_easy | rho_easy |")
out.append("|---|---|---|---|---|---|")
hd={r[0]:(r[2],r[4]) for r in coarse_hard["rows"]}
ed={r[0]:(r[2],r[4]) for r in coarse_easy["rows"]}
for aid in sorted(set(hd)|set(ed)):
    nh,rh=hd.get(aid,(0,float("nan"))); ne,re=ed.get(aid,(0,float("nan")))
    out.append(f"| {aid} | {assays[aid]['acc']} | {nh} | "
               f"{rh:.3f} | {ne} | {re:.3f} |")

with open(f"{STEPB}/STEP_B_SCORING_RESULTS.md","w") as fh:
    fh.write("\n".join(out)+"\n")

# also a machine-readable per-assay-bin TSV
with open(f"{STEPB}/axis3_SPC_perassay_bins.tsv","w") as fh:
    fh.write("bin_label\tassay\tacc\tn_variants\tabs_rho\tsigned_rho\n")
    for lbl,d in [("coarse_hard",coarse_hard),("coarse_easy",coarse_easy),
                  ("q0",qres[0]),("q1",qres[1]),("q2",qres[2]),("q3",qres[3]),("q4",qres[4]),
                  ("insert",insert_res),("deep_occ<0.5",d05),("deep_occ<0.7",d07),
                  ("deep_occ>=0.9",dhi)]:
        for r in d["rows"]:
            fh.write(f"{lbl}\t{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.4f}\t{r[4]:.4f}\n")

# console summary
print("\n===== SUMMARY =====")
print(f"COARSE  hard: wmean|rho|={fmt(coarse_hard)} (na={coarse_hard['n_assays']},np={coarse_hard['n_prot']},nv={coarse_hard['n_var']})")
print(f"COARSE  easy: wmean|rho|={fmt(coarse_easy)} (na={coarse_easy['n_assays']},np={coarse_easy['n_prot']},nv={coarse_easy['n_var']})")
print(f"GAP (easy-hard) = {gap:+.3f}")
for k in range(5):
    d=qres[k]; print(f"q{k} {rngs[k]:18s}: wmean|rho|={fmt(d)} (na={d['n_assays']},np={d['n_prot']},nv={d['n_var']})")
print(f"insert      : wmean|rho|={fmt(insert_res)} (na={insert_res['n_assays']},np={insert_res['n_prot']},nv={insert_res['n_var']})")
print(f"DEEP&occ<0.5: wmean|rho|={fmt(d05)} (na={d05['n_assays']},np={d05['n_prot']},nv={d05['n_var']})")
print(f"DEEP&occ<0.7: wmean|rho|={fmt(d07)} (na={d07['n_assays']},np={d07['n_prot']},nv={d07['n_var']})")
print(f"DEEP&occ>=.9: wmean|rho|={fmt(dhi)} (na={dhi['n_assays']},np={dhi['n_prot']},nv={dhi['n_var']})")
print("\nWROTE stepB/STEP_B_SCORING_RESULTS.md and stepB/axis3_SPC_perassay_bins.tsv")
print("DONE")
