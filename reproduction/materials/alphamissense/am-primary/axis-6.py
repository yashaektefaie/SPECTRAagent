# Original notebook cell 3
from pathlib import Path
import os
PRIMARY_REPLAY_DIR=Path.cwd()
os.chdir(PRIMARY_REPLAY_DIR.parent/'upstream_checks')

# Original notebook cell 5
from pathlib import Path
import json,hashlib,platform,subprocess,sys,tarfile
import numpy as np,pandas as pd
from IPython.display import display
ROOT=Path.cwd();BUNDLE=ROOT.parent;INPUT=BUNDLE/'am-primary/inputs';EXPECTED=BUNDLE/'am-primary/expected'
manifest=json.loads((BUNDLE/'am-opus-r2/INPUT_MANIFEST.json').read_text())
archive=BUNDLE/'am-opus-r2/inputs/raw_references.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==next(x['sha256'] for x in manifest if x['path']=='inputs/raw_references.tar.gz')
expected={x['filename']:x['sha256'] for x in json.loads((ROOT/'opus_r2_msa_member_hashes.json').read_text())}
checked={}
with tarfile.open(archive,'r|gz') as t:
    for m in t:
        name=Path(m.name).name
        if m.isfile() and m.name.startswith('msa/') and name in expected:
            h=hashlib.sha256()
            with t.extractfile(m) as f:
                for part in iter(lambda:f.read(16777216),b''):h.update(part)
            assert h.hexdigest()==expected[name],name;checked[name]=h.hexdigest()
assert len(checked)==66
assert hashlib.sha256((ROOT/'bin/mmseqs').read_bytes()).hexdigest()=='c58e5b422092544aac95646828eb563cee32ca609c260c50a195bb560d24f2e6'
if platform.system()=='Linux' and platform.machine() in ['x86_64','AMD64']:
    subprocess.run([sys.executable,str(ROOT/'run_upstream.py'),'--all','--resume'],check=True)
    RESULT=ROOT/'rebuilt/results';events=json.loads((ROOT/'rebuilt/events.json').read_text())
    print('Executed upstream reconstruction on Linux.')
else:
    RESULT=ROOT/'results';events=json.loads((ROOT/'events.json').read_text())
    print('This platform uses the recorded Linux execution arrays; it does not execute the Linux MMseqs binary here.')
done={x['msa']:x for x in events if x['kind']=='features_complete'};assert len(done)==66
for name,item in done.items():assert hashlib.sha256((RESULT/(name+'.npz')).read_bytes()).hexdigest()==item['output_sha256'],name
print('All 66 raw alignments and completed array outputs verified.')
display(pd.DataFrame([json.loads((ROOT/'EXECUTION_ENVIRONMENT.json').read_text())]))

# Original notebook cell 7
from materialize_features import materialize
OUT=ROOT/'reconstructed'
record=materialize(BUNDLE,RESULT,OUT);assert record['axis_rows']==17536 and record['pssm_rows']==15478
old_axis=pd.read_csv(INPUT/'axis3_pathC_percolumn_FROZEN.tsv',sep='\t');new_axis=pd.read_csv(OUT/'axis3_pathC_percolumn_FROZEN.tsv',sep='\t')
old_counts=pd.read_csv(INPUT/'pssm_counts.tsv',sep='\t');new_counts=pd.read_csv(OUT/'pssm_counts.tsv',sep='\t')
for field in ['assay','accession','gene','assay_family','target_pos','occupancy','low_cov_insert','depth_bin','n_variants']:
    pd.testing.assert_series_equal(old_axis[field],new_axis[field],check_exact=True)
axis_changed=(old_axis.ne(new_axis)&~(old_axis.isna()&new_axis.isna())).any(axis=1)
counts_changed=(old_counts.ne(new_counts)&~(old_counts.isna()&new_counts.isna())).any(axis=1)
comparison=dict(axis_rows=len(old_axis),pssm_rows=len(old_counts),axis_rows_changed=int(axis_changed.sum()),count_rows_changed=int(counts_changed.sum()),all_occupancy_flags_levels_exact=True,
    affected_assays=old_axis.loc[axis_changed,'assay'].unique().tolist(),max_depth_change=float((old_axis.neff_theta02-new_axis.neff_theta02).abs().max()),max_amino_acid_count_change=float((old_counts.iloc[:,3:]-new_counts.iloc[:,3:]).abs().to_numpy().max()))
assert comparison['axis_rows_changed']==comparison['count_rows_changed']==37
assert all('OTU7A' in a for a in comparison['affected_assays'])
(OUT/'feature_comparison.json').write_text(json.dumps(comparison,indent=2));display(pd.DataFrame([comparison]))
old_axis.loc[axis_changed].assign(new_effective_depth=new_axis.loc[axis_changed,'neff_theta02'].to_numpy()).to_csv(OUT/'differing_positions.csv',index=False)

# Original notebook cell 9
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

BASE=str(BUNDLE/"am-primary")
PG=str(INPUT/"proteingym"); STEPB=str(OUT)
AMFILE=str(INPUT/"official_scores_64_proteins.tsv.gz")
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

# Original notebook cell 10
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

BASE=str(BUNDLE/"am-primary"); PG=str(INPUT/"proteingym"); STEPB=str(OUT)
AMFILE=str(INPUT/"official_scores_64_proteins.tsv.gz")
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

# Original notebook cell 11
assert len(rec)==270572
old=pd.read_csv(EXPECTED/'axis3_baselines_binstats.tsv',sep='\t');new=pd.read_csv(OUT/'axis3_baselines_binstats.tsv',sep='\t')
changes=[]
for i,row in old.iterrows():
    for field in ['wmean_absrho','ci2.5','ci97.5']:
        a=row[field];b=new.loc[i,field]
        if pd.notna(a) and abs(a-b)>0:changes.append(dict(predictor=row.predictor,bin=row.bin,statistic=field,original=float(a),rebuilt=float(b),difference=float(b-a)))
change_table=pd.DataFrame(changes,columns=['predictor','bin','statistic','original','rebuilt','difference']);change_table.to_csv(OUT/'downstream_statistic_changes.csv',index=False)
for filename in ['axis3_SPC_perassay_bins.tsv','axis3_dms_signal_perbin.tsv']:
    a=pd.read_csv(EXPECTED/filename,sep='\t');b=pd.read_csv(OUT/filename,sep='\t');pd.testing.assert_frame_equal(a,b,check_exact=True)
assert all(x['predictor']=='PSSM' for x in changes)
summary=dict(rebuilt_scored_variants=len(rec),AM_per_assay_outputs_unchanged=True,DMS_dynamic_range_unchanged=True,changed_summary_statistics=len(changes),maximum_summary_change=max([abs(x['difference']) for x in changes],default=0.0),changed_predictors=sorted({x['predictor'] for x in changes}),independent_confirmation=False)
(OUT/'DOWNSTREAM_VALIDATION.json').write_text(json.dumps(summary,indent=2));display(pd.DataFrame([summary]));display(change_table)
# Export the altered baseline's individual values, including exact assay positions.
old_individual=pd.read_parquet(BUNDLE/'am-primary/recomputed/individual_predictions_measurements_levels.parquet')
old_individual['rebuilt_PSSM']=[pssm_score(r.assay,r.target_pos,r.mutant[0],r.mutant[-1]) for r in old_individual.itertuples()]
old_individual.loc[old_individual.PSSM.notna()&old_individual.rebuilt_PSSM.notna()&old_individual.PSSM.ne(old_individual.rebuilt_PSSM),['assay','acc','mutant','target_pos','DMS_score','PSSM','rebuilt_PSSM']].to_csv(OUT/'changed_individual_pssm_scores.csv.gz',index=False)

# Original notebook cell 13
os.chdir(PRIMARY_REPLAY_DIR)
print('The archived-feature track below is preserved separately from the rebuilt-feature comparison above.')

# Original notebook cell 14
from pathlib import Path
import json,hashlib,shutil,platform
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=Path.cwd();INPUT=ROOT/'inputs';EXPECTED=ROOT/'expected';OUT=ROOT/'recomputed';OUT.mkdir(exist_ok=True)
for item in json.loads((ROOT/'INPUT_MANIFEST.json').read_text()):
    p=ROOT/item['path'];assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
source=json.loads((INPUT/'official_score_subset_provenance.json').read_text())
assert source['source_sha256']=='bc18d954a9aa6b9efab8e40d6d1912e32b774906cd3846a1f51066588e9c176d'
assert source['selected_rows']==955738 and source['selected_proteins']==64
for name in ['axis3_pathC_percolumn_FROZEN.tsv','pssm_counts.tsv']:
    shutil.copyfile(INPUT/name,OUT/name)
print('All retained inputs and code verified; Python',platform.python_version())
display(pd.DataFrame([source]))

# Original notebook cell 16
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

BASE=str(ROOT)
PG=str(INPUT/"proteingym"); STEPB=str(OUT)
AMFILE=str(INPUT/"official_scores_64_proteins.tsv.gz")
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

# Original notebook cell 17
assert tot_single==273135 and tot_scored==270572 and n_scored_binned==270572
original= pd.read_csv(EXPECTED/'axis3_SPC_perassay_bins.tsv',sep='\t')
replayed= pd.read_csv(OUT/'axis3_SPC_perassay_bins.tsv',sep='\t')
keys=['bin_label','assay','acc'];original=original.sort_values(keys).reset_index(drop=True);replayed=replayed.sort_values(keys).reset_index(drop=True)
pd.testing.assert_frame_equal(original,replayed,check_exact=True)
coverage=pd.DataFrame(cov,columns=['assay','acc','n_single','n_scored','offset_fail','wt_mismatch','no_AM_variant','status'])
coverage.to_csv(OUT/'join_coverage.csv',index=False)
print('All 587 original assay/bin results reproduced exactly at the saved four-decimal precision.')
display(coverage[coverage.status.ne('OK')])
display(replayed.groupby('bin_label').agg(assays=('assay','nunique'),variants=('n_variants','sum')))

# Original notebook cell 19
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

BASE=str(ROOT); PG=str(INPUT/"proteingym"); STEPB=str(OUT)
AMFILE=str(INPUT/"official_scores_64_proteins.tsv.gz")
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

# Original notebook cell 20
for filename in ['axis3_baselines_binstats.tsv','axis3_dms_signal_perbin.tsv']:
    old=pd.read_csv(EXPECTED/filename,sep='\t');new=pd.read_csv(OUT/filename,sep='\t')
    pd.testing.assert_frame_equal(old,new,check_exact=True)
    print(filename,': all saved statistics and confidence intervals reproduced exactly.')
curve=pd.read_csv(OUT/'axis3_baselines_binstats.tsv',sep='\t')
# Export individual predictor/response pairs, preserving original CSV float parsing.
# The original rec tuples omit substitution identity; recover it through the same exact join.
import csv
individual=[]
for aid,info in assays.items():
    acc=info['acc'];cn=canon(acc);off=cn.find(info['tseq']) if cn and info['tseq'] else -1
    with open(INPUT/'proteingym/dms_final_73'/info['dms']) as f:
        for raw in csv.DictReader(f):
            mutation=raw['mutant'].strip()
            if ':' in mutation or off<0:continue
            pos=int(mutation[1:-1]);wt=mutation[0];mut=mutation[-1];cpos=off+pos
            if amwt[acc].get(cpos)!=wt or (cpos,mut) not in am[acc]:continue
            b=binmap[(aid,pos)]
            individual.append(dict(assay=aid.strip(),acc=acc,mutant=mutation,target_pos=pos,canonical_pos=cpos,offset=off,
                DMS_score=float(raw['DMS_score']),AM=am[acc][(cpos,mut)],BLOSUM62=blosum(wt,mut),PSSM=pssm_score(aid,pos,wt,mut),
                occupancy=b['occ'],effective_depth=b['neff'],insert=b['insert'],
                occupancy_bin='insert' if b['insert'] else 'q'+str(qbin(b['occ'])),
                cohort='original_scored_panel; no independent confirmation split'))
measurements=pd.DataFrame(individual);assert len(measurements)==len(rec)==270572
measurements.to_parquet(OUT/'individual_predictions_measurements_levels.parquet',index=False)
measurements[['assay','acc','mutant','target_pos','canonical_pos','occupancy_bin','cohort']].to_csv(OUT/'sample_membership.csv.gz',index=False)
# Every plotted group statistic is recalculated from the original individual tuples.
per=[]
for pname,pidx in PRED.items():
    for label,selector in BINSELS.items():
        groups=defaultdict(list)
        for t in rec:
            if t[pidx] is not None and selector(binmap[(t[0],t[2])]):groups[t[0]].append(t)
        for aid,g in groups.items():
            if len(g)<MIN_VAR:continue
            rho=spearmanr([t[pidx] for t in g],[t[3] for t in g]).statistic
            if np.isfinite(rho):per.append(dict(predictor=pname,bin=label,assay=aid.strip(),acc=assays[aid]['acc'],n_variants=len(g),abs_rho=abs(rho),signed_rho=rho))
per=pd.DataFrame(per);per.to_csv(OUT/'individual_assay_bin_performance.csv',index=False)
display(measurements.head());display(curve)

# Original notebook cell 21
fig,axs=plt.subplots(1,2,figsize=(11,3.7),constrained_layout=True)
order=['insert','q0','q1','q2','q3','q4']
for predictor,color in [('AM','#9b2d30'),('PSSM','#4a708b'),('BLOSUM62','#677f59')]:
    sub=curve[(curve.predictor==predictor)&curve.bin.isin(order)].set_index('bin').reindex(order)
    y=sub.wmean_absrho.to_numpy();lo=sub['ci2.5'].to_numpy();hi=sub['ci97.5'].to_numpy()
    axs[0].errorbar(range(6),y,yerr=[y-lo,hi-y],marker='o',capsize=3,label=predictor,color=color)
axs[0].set_xticks(range(6),['Insert','Q1','Q2','Q3','Q4','Q5']);axs[0].set_xlabel('Original occupancy groups (low → high)');axs[0].set_ylabel('Variant-weighted mean |Spearman|');axs[0].legend(frameon=False)
spread=pd.read_csv(OUT/'axis3_dms_signal_perbin.tsv',sep='\t').set_index('bin').reindex(order)
axs[1].plot(range(6),spread.mean_relative_spread,'o-',color='#555555');axs[1].axhline(1,color='grey',ls=':',lw=1)
axs[1].set_xticks(range(6),['Insert','Q1','Q2','Q3','Q4','Q5']);axs[1].set_ylabel('Mean within-assay relative DMS spread');axs[1].set_xlabel('Original occupancy groups')
for ax in axs:ax.spines[['top','right']].set_visible(False)
fig.savefig(OUT/'occupancy_and_dynamic_range.png',dpi=180);plt.show()

# Original notebook cell 23
from scipy.stats import binomtest
paired=[]
for name,high,low in [('coarse','coarse_easy','coarse_hard'),('occupancy','q4','q0'),('high_depth','deep_occ>=0.9','deep_occ<0.7')]:
    tab=replayed[replayed.bin_label.isin([high,low])].pivot(index='assay',columns='bin_label',values='abs_rho').dropna()
    diff=tab[high]-tab[low];positive=int((diff>0).sum());n=len(diff)
    paired.append(dict(comparison=name,n_pairs=n,n_positive=positive,p_two_sided=binomtest(positive,n,.5).pvalue,mean_difference=diff.mean(),median_difference=diff.median()))
paired=pd.DataFrame(paired);paired.to_csv(OUT/'paired_sign_checks.csv',index=False)
assert int(paired.loc[paired.comparison.eq('occupancy'),'n_positive'].iloc[0])==44
assert int(paired.loc[paired.comparison.eq('high_depth'),'n_positive'].iloc[0])==24
frozen=pd.read_csv(INPUT/'axis3_pathC_percolumn_FROZEN.tsv',sep='\t')
# Reproduce original protein-position gate unit, without counting repeat assays twice.
distinct=frozen.drop_duplicates(['accession','target_pos'])
levels=distinct[distinct.low_cov_insert.eq(0)].groupby('depth_bin').agg(positions=('target_pos','size'),proteins=('accession','nunique'))
levels.to_csv(OUT/'absolute_depth_group_support.csv')
print('Paired within-assay checks (same cohort):');display(paired)
print('Original absolute depth groups:');display(levels)
assert levels.loc[0,'proteins']==1 and levels.loc[1,'proteins']==2
print('No independent confirmation cohort is retained or asserted. The upstream notebook documents the executed alignment reconstruction and isolated OTU7A difference.')

# Original notebook cell 25
outputs=[]
for p in sorted(OUT.glob('*')):
    if p.is_file():outputs.append(dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(OUT/'OUTPUT_MANIFEST.json').write_text(json.dumps(outputs,indent=2))
print(len(outputs),'executed replay outputs saved and hashed.')
