#!/usr/bin/env python3
"""Score AM vs AM-INDEPENDENT baselines (site-independent PSSM log-odds, BLOSUM62) on the SAME
frozen occupancy bins, plus a DMS-signal control (within-assay DMS spread per bin). Answers:
does AM degrade MORE steeply than baselines (AM-specific), or do all degrade together
(intrinsic position difficulty)? And is DMS signal collapsing at gappy columns?
Metric = per-assay |Spearman(pred, DMS)|, variant-weighted across assays, 95% bootstrap CI over assays.
New files only."""
import csv, gzip, math
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from Bio.Align import substitution_matrices

BASE="/path/to/audit-home/spectra_alphamissense"; PG=f"{BASE}/proteingym"; STEPB=f"{BASE}/stepB"
AMFILE=f"{BASE}/data/AlphaMissense_aa_substitutions.tsv.gz"
MIN_VAR=20; AA="ACDEFGHIKLMNPQRSTVWY"; BETA=0.5
B62=substitution_matrices.load("BLOSUM62")
def blosum(w,m):
    try: return float(B62[(w,m)])
    except:
        try: return float(B62[(m,w)])
        except: return None

# ---------- panel ----------
assays={}; accs=set()
with open(f"{PG}/kept_assays_verified.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        acc=r["resolved_accession"].strip()
        assays[r["DMS_id"]]={"acc":acc,"tseq":r["target_seq"].strip(),"dms":r["DMS_filename"].strip()}
        accs.add(acc)

# ---------- AM lookup ----------
am=defaultdict(dict); amwt=defaultdict(dict)
with gzip.open(AMFILE,"rt") as fh:
    for line in fh:
        if line[0]=="#" or line[:10]=="uniprot_id": continue
        p=line.rstrip("\n").split("\t")
        if len(p)<3 or p[0] not in accs: continue
        v=p[1];
        try: pos=int(v[1:-1])
        except: continue
        am[p[0]][(pos,v[-1])]=float(p[2]); amwt[p[0]][pos]=v[0]
def canon(acc):
    d=amwt.get(acc,{});
    return "".join(d.get(i,"X") for i in range(1,max(d)+1)) if d else ""
print(f"AM accessions loaded: {len(am)}/{len(accs)}",flush=True)

# ---------- PSSM lookup ----------
pssm={}   # (assay,pos) -> dict aa->count, plus tot
with open(f"{STEPB}/pssm_counts.tsv") as fh:
    rd=csv.reader(fh,delimiter="\t"); hdr=next(rd)
    aacols=hdr[3:]
    for row in rd:
        assay=row[0]; pos=int(row[1])
        cnt={aacols[i]:float(row[3+i]) for i in range(len(aacols))}
        pssm[(assay,pos)]=cnt
def pssm_score(assay,pos,w,m):
    c=pssm.get((assay,pos))
    if c is None or w not in c or m not in c: return None
    return math.log((c[m]+BETA)/(c[w]+BETA))
print(f"PSSM positions loaded: {len(pssm)}",flush=True)

# ---------- frozen bins ----------
binmap={}
with open(f"{STEPB}/axis3_pathC_percolumn_FROZEN.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        ins=r["low_cov_insert"]=="1"; occ=r["occupancy"].strip(); neff=r["neff_theta02"].strip()
        binmap[(r["assay"],int(r["target_pos"]))]={"occ":float(occ) if occ else float("nan"),
            "insert":ins,"neff":float(neff) if neff else float("nan")}

# ---------- join ----------
rec=[]  # (assay,acc,pos,dms,am,blosum,pssm)
for aid,info in assays.items():
    acc=info["acc"]; cn=canon(acc); off=cn.find(info["tseq"]) if (cn and info["tseq"]) else -1
    with open(f"{PG}/dms_final_73/{info['dms']}") as fh:
        for r in csv.DictReader(fh):
            mut=r["mutant"].strip()
            if ":" in mut: continue
            w=mut[0]; m=mut[-1]
            try: pos=int(mut[1:-1]); dms=float(r["DMS_score"])
            except: continue
            if off<0: continue
            cpos=off+pos
            if amwt[acc].get(cpos)!=w: continue
            a=am[acc].get((cpos,m))
            if a is None: continue
            rec.append((aid,acc,pos,dms,a,blosum(w,m),pssm_score(aid,pos,w,m)))
print(f"joined variants: {len(rec)}",flush=True)

# predictor extractors (index into rec tuple)
PRED={"AM":4,"BLOSUM62":5,"PSSM":6}

# ---------- bins ----------
QE=[0.945,0.987,0.998,0.999]
def qbin(o): return min(sum(1 for e in QE if o>e),4)
BINSELS={
 "insert":lambda b: b["insert"],
 "q0":lambda b:(not b["insert"]) and qbin(b["occ"])==0,
 "q1":lambda b:(not b["insert"]) and qbin(b["occ"])==1,
 "q2":lambda b:(not b["insert"]) and qbin(b["occ"])==2,
 "q3":lambda b:(not b["insert"]) and qbin(b["occ"])==3,
 "q4":lambda b:(not b["insert"]) and qbin(b["occ"])==4,
 "deep_occ>=0.9":lambda b:(not b["insert"]) and b["neff"]>=200 and b["occ"]>=0.9,
 "deep_occ<0.7":lambda b:(not b["insert"]) and b["neff"]>=200 and b["occ"]<0.7,
}

def agg(pred_idx, binsel, B=5000, seed=42):
    per=defaultdict(list)
    for t in rec:
        v=t[pred_idx]
        if v is None: continue
        b=binmap.get((t[0],t[2]))
        if b is None or not binsel(b): continue
        per[t[0]].append((v,t[3]))
    rows=[]
    for aid,pairs in per.items():
        if len(pairs)<MIN_VAR: continue
        x=[p[0] for p in pairs]; y=[p[1] for p in pairs]
        rho=spearmanr(x,y).correlation
        if rho==rho: rows.append((aid,assays[aid]["acc"],len(pairs),abs(rho)))
    if not rows: return None
    nv=np.array([r[2] for r in rows],float); ar=np.array([r[3] for r in rows])
    wmean=float((nv*ar).sum()/nv.sum())
    rng=np.random.default_rng(seed); n=len(rows); idx=rng.integers(0,n,size=(B,n))
    bm=(nv[idx]*ar[idx]).sum(1)/nv[idx].sum(1); lo,hi=np.percentile(bm,[2.5,97.5])
    return dict(wmean=wmean,lo=float(lo),hi=float(hi),na=n,nprot=len(set(r[1] for r in rows)),nv=int(nv.sum()))

# ---------- DMS signal control: within-assay DMS spread per bin (normalized) ----------
ABINS=["insert","q0","q1","q2","q3","q4"]
dms_by=defaultdict(lambda: defaultdict(list))  # assay -> bin -> [dms]
alld=defaultdict(list)
for t in rec:
    b=binmap.get((t[0],t[2]))
    if b is None: continue
    for bl in ABINS:
        if BINSELS[bl](b): dms_by[t[0]][bl].append(t[3]); break
    alld[t[0]].append(t[3])
sig={}
for bl in ABINS:
    rels=[]; stds=[]
    for aid in dms_by:
        v=dms_by[aid].get(bl,[])
        if len(v)<MIN_VAR: continue
        s=float(np.std(v)); o=float(np.std(alld[aid]))
        stds.append(s)
        if o>0: rels.append(s/o)
    sig[bl]=dict(med_std=float(np.median(stds)) if stds else float("nan"),
                 mean_rel=float(np.mean(rels)) if rels else float("nan"),na=len(stds))

# ---------- write ----------
with open(f"{STEPB}/axis3_baselines_binstats.tsv","w") as fh:
    fh.write("predictor\tbin\twmean_absrho\tci2.5\tci97.5\tn_assays\tn_proteins\tn_variants\n")
    for pname,pidx in PRED.items():
        for bl in ["insert","q0","q1","q2","q3","q4","deep_occ>=0.9","deep_occ<0.7"]:
            s=agg(pidx,BINSELS[bl])
            if s: fh.write(f"{pname}\t{bl}\t{s['wmean']:.4f}\t{s['lo']:.4f}\t{s['hi']:.4f}\t{s['na']}\t{s['nprot']}\t{s['nv']}\n")
            else: fh.write(f"{pname}\t{bl}\tNA\tNA\tNA\t0\t0\t0\n")
with open(f"{STEPB}/axis3_dms_signal_perbin.tsv","w") as fh:
    fh.write("bin\tmedian_within_assay_dms_std\tmean_relative_spread\tn_assays\n")
    for bl in ABINS:
        s=sig[bl]; fh.write(f"{bl}\t{s['med_std']:.4f}\t{s['mean_rel']:.4f}\t{s['na']}\n")

# ---------- console ----------
print("\n===== AM vs BASELINES (wmean |rho| per bin) =====")
print(f"{'bin':14s}{'AM':>20}{'PSSM':>20}{'BLOSUM62':>20}")
def cell(pidx,bl):
    s=agg(pidx,BINSELS[bl]);
    return f"{s['wmean']:.3f}[{s['lo']:.2f}-{s['hi']:.2f}]n{s['na']}" if s else "--"
for bl in ["insert","q0","q1","q2","q3","q4","deep_occ>=0.9","deep_occ<0.7"]:
    print(f"{bl:14s}{cell(4,bl):>20}{cell(6,bl):>20}{cell(5,bl):>20}")
print("\n===== DMS SIGNAL CONTROL (within-assay spread per bin) =====")
print(f"{'bin':10s}{'median_std':>14}{'mean_rel_spread':>18}{'n':>5}")
for bl in ABINS:
    s=sig[bl]; print(f"{bl:10s}{s['med_std']:>14.3f}{s['mean_rel']:>18.3f}{s['na']:>5}")
print("\nWROTE axis3_baselines_binstats.tsv, axis3_dms_signal_perbin.tsv")
print("DONE")
