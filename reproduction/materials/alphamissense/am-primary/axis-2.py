# Original notebook cell 2
from pathlib import Path
import hashlib,json,csv,platform
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=Path.cwd();INPUT=ROOT/'inputs/clinvar';OUT=ROOT/'recomputed/clinvar_pop';OUT.mkdir(exist_ok=True,parents=True)
for item in json.loads((INPUT/'INPUT_MANIFEST.json').read_text()):
    p=INPUT/item['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
assert hashlib.sha256((INPUT/'clinvar_am_benchmark.tsv').read_bytes()).hexdigest()=='783a79533ae1aa4ae4b7d99615f78959fd134aa4913666407720c4bc7f6ad481'
print('Exact individual benchmark and axis inputs verified; Python',platform.python_version())

# Original notebook cell 4
#!/usr/bin/env python3
"""SPECTRA SPC: AlphaMissense vs population-variation support (gnomAD).

Axis (prospective, leakage-free wrt AM): per-gene observed-missense saturation
= obs_mis / possible_mis from gnomAD v2.1.1 constraint. Higher = more observed
population variation = more of the weak-supervision signal AM was fine-tuned on.
Lower = sparse population support -> SPECTRA hypothesis: AM degrades.

Confounder tracked: oe_mis (obs/exp) = selective constraint, which correlates
with pathogenicity prevalence. Reported per level. Primary metric auROC is
prevalence-robust. BLOSUM62 fixed baseline included.
"""
import csv, json, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

WORK=str(INPUT)
DATA=str(INPUT)
ART=str(OUT)

# BLOSUM62 (for baseline)
_B62="""A 4;R -1 5;N -2 0 6;D -2 -2 1 6;C 0 -3 -3 -3 9;Q -1 1 0 0 -3 5;E -1 0 0 2 -4 2 5;G 0 -2 0 -1 -3 -2 -2 6;H -2 0 1 -1 -3 0 0 -2 8;I -1 -3 -3 -3 -1 -3 -3 -4 -3 4;L -1 -2 -3 -4 -1 -2 -3 -4 -3 2 4;K -1 2 0 -1 -3 1 1 -2 -1 -3 -2 5;M -1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5;F -2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6;P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7;S 1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4;T 0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5;W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11;Y -2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7;V 0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4"""
order=[];B62={}
for e in _B62.split(";"):
    t=e.split();aa=t[0];vals=list(map(int,t[1:]));order.append(aa)
    for j,v in enumerate(vals): B62[(aa,order[j])]=v;B62[(order[j],aa)]=v

# gnomAD per-gene support
gcol={}
with open(f"{DATA}/gnomad_constraint.tsv") as fh:
    hdr=fh.readline().rstrip("\n").split("\t")
    ix={c:i for i,c in enumerate(hdr)}
    for line in fh:
        f=line.rstrip("\n").split("\t")
        g=f[ix["gene"]]
        try:
            obs=float(f[ix["obs_mis"]]); poss=float(f[ix["possible_mis"]])
            oe=f[ix["oe_mis"]]; oe=float(oe) if oe not in ("","NA") else None
        except: continue
        if poss<=0: continue
        sat=obs/poss
        # keep the max-saturation transcript per gene symbol
        if g not in gcol or sat>gcol[g]["sat"]:
            gcol[g]={"sat":sat,"obs":obs,"oe":oe}
print(f"gnomAD genes: {len(gcol)}")

# benchmark
rows=[]
matched=0
with open(f"{WORK}/clinvar_am_benchmark.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        pv=r["protein_variant"];ra,aa=pv[0],pv[-1]
        if ra not in order or aa not in order: continue
        g=r["gene"]
        gi=gcol.get(g)
        if gi is None: continue
        matched+=1
        rows.append({"gene":g,"label":int(r["label"]),"am":float(r["am_pathogenicity"]),
                     "blosum":-B62[(ra,aa)],"sat":gi["sat"],"oe":gi["oe"]})
print(f"variants matched to gnomAD gene: {matched}")

# protein/gene-balanced deciles on saturation
gene_sat={r["gene"]:r["sat"] for r in rows}
genes=sorted(gene_sat,key=lambda g:gene_sat[g])
vals=np.array([gene_sat[g] for g in genes])
N=10
qs=np.quantile(vals,np.linspace(0,1,N+1)); qs[0]-=1e-9; qs[-1]+=1e-9
def lvl(s): return int(np.clip(np.searchsorted(qs,s,side="right")-1,0,N-1))
glvl={g:lvl(gene_sat[g]) for g in genes}
for r in rows: r["level"]=glvl[r["gene"]]

rng=np.random.default_rng(2)
def auroc(items,key="am"):
    y=[i["label"] for i in items];s=[i[key] for i in items]
    return roc_auc_score(y,s) if len(set(y))>1 else None
def boot(items,B=400):
    pmap=collections.defaultdict(list)
    for it in items: pmap[it["gene"]].append(it)
    pl=list(pmap);out=[]
    for _ in range(B):
        idx=rng.choice(len(pl),len(pl),replace=True);pool=[]
        for i in idx: pool.extend(pmap[pl[i]])
        a=auroc(pool)
        if a is not None: out.append(a)
    return (round(float(np.percentile(out,2.5)),4),round(float(np.percentile(out,97.5)),4)) if out else (None,None)

by=collections.defaultdict(list)
for r in rows: by[r["level"]].append(r)

levels=[]
print("\nlvl  nvar  ngene  prev  med_sat  med_oe   AM_auROC  CI95            BLOSUM")
for lv in range(N):
    it=by[lv];ng=len(set(i["gene"] for i in it))
    npos=sum(i["label"] for i in it);nneg=len(it)-npos
    med_sat=float(np.median([i["sat"] for i in it]))
    oes=[i["oe"] for i in it if i["oe"] is not None]
    med_oe=float(np.median(oes)) if oes else None
    a=auroc(it);b=auroc(it,"blosum");lo,hi=boot(it) if (npos and nneg) else (None,None)
    rec={"level":lv,"n_var":len(it),"n_gene":ng,"prev":round(npos/len(it),3),
         "med_saturation":round(med_sat,4),"med_oe_mis":round(med_oe,3) if med_oe else None,
         "am_auroc":round(a,4) if a else None,"am_ci95":[lo,hi],
         "blosum_auroc":round(b,4) if b else None}
    levels.append(rec)
    print(f"L{lv:<2} {len(it):6d} {ng:5d} {rec['prev']:.3f}  {med_sat:.4f}  "
          f"{(med_oe if med_oe else 0):.3f}   {rec['am_auroc']}  [{lo},{hi}]  {rec['blosum_auroc']}")

med_sats=[r["med_saturation"] for r in levels]
am=[r["am_auroc"] for r in levels]
oe=[r["med_oe_mis"] for r in levels]
rho,p=spearmanr(med_sats,am)
rho_oe,_=spearmanr(med_sats,[x for x in oe])
mono=all(med_sats[i]<med_sats[i+1] for i in range(N-1))
overall=auroc(rows)
summary={"axis":"gnomAD_observed_missense_saturation","n_levels":N,
         "axis_direction":"level 0 = lowest population support (lowest obs/possible missense)",
         "overall_am_auroc":round(overall,4),"levels":levels,
         "saturation_monotonic":bool(mono),
         "spearman_saturation_vs_AM_auroc":{"rho":round(float(rho),3),"p":round(float(p),4)},
         "am_auroc_lowest_vs_highest":[am[0],am[-1]],
         "confounder_med_oe_lowest_vs_highest":[oe[0],oe[-1]]}
json.dump(summary,open(f"{ART}/spc_pop_support.json","w"),indent=2)
print("\n",json.dumps({k:summary[k] for k in ["overall_am_auroc","saturation_monotonic","spearman_saturation_vs_AM_auroc","am_auroc_lowest_vs_highest","confounder_med_oe_lowest_vs_highest"]},indent=2))
print("wrote spc_pop_support.json")

# Original notebook cell 5
# Archived summary values are validation targets, never inputs to the calculation.
original=json.loads((ROOT/'expected/original_spc_pop_support.json').read_text())
replayed=json.loads((OUT/'spc_pop_support.json').read_text())
def compare(a,b,path='root'):
    if isinstance(a,dict):
        assert set(a)==set(b),path
        for key in a:compare(a[key],b[key],path+'.'+key)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{i}]')
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True,err_msg=path)
    else:assert a==b,(path,a,b)
compare(original,replayed)
print('Every original summary field reproduced within 1e-12, including the original clustered confidence intervals.')
display(pd.DataFrame(replayed['levels']))

# Original notebook cell 7
records=[]
with open(INPUT/'clinvar_am_benchmark.tsv') as f:
    for row_id,r in enumerate(csv.DictReader(f,delimiter='\t'),start=1):
        variant=r['protein_variant'];wt,alt=variant[0],variant[-1];gene=r['gene']
        if wt not in order or alt not in order or gene not in gcol:continue
        g=gcol[gene]
        r.update(original_source_row=row_id,axis_observed_missense_saturation=g['sat'],gnomad_observed_missense=g['obs'],gnomad_observed_expected=g['oe'],level=glvl[gene],BLOSUM62_pathogenicity=-B62[(wt,alt)],cohort='original ClinVar screen; no independent confirmation split')
        records.append(r)
individual=pd.DataFrame(records);assert len(individual)==len(rows)==186514
np.testing.assert_allclose(individual.am_pathogenicity.astype(float),[r['am'] for r in rows],atol=1e-12,rtol=0)
np.testing.assert_array_equal(individual.level,[r['level'] for r in rows])
individual.to_csv(OUT/'per_variant_scored.csv.gz',index=False)
pd.DataFrame([dict(gene=g,axis_value=gene_sat[g],level=glvl[g]) for g in genes]).to_csv(OUT/'gene_levels.csv',index=False)
display(individual.head())

# Original notebook cell 9
curve=pd.DataFrame(replayed['levels']);x=curve['med_saturation'].to_numpy();y=curve['am_auroc'].to_numpy();ci=np.asarray(curve['am_ci95'].tolist())
fig,ax=plt.subplots(figsize=(6.5,3.7),constrained_layout=True)
ax.errorbar(x,y,yerr=[y-ci[:,0],ci[:,1]-y],marker='o',capsize=3,color='#8e3c35',label='AlphaMissense')
ax.plot(x,curve['blosum_auroc'],'o--',color='#586c7b',label='BLOSUM62')
ax.set(xlabel='Median observed/possible missense variation',ylabel='AUROC');ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
fig.savefig(OUT/'curve.png',dpi=180);plt.show()
curve.to_csv(OUT/'curve_from_individual_records.csv',index=False)
print('The published negative screen is retained. No new independent confirmation is asserted.')

# Original notebook cell 10
manifest=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(OUT.glob('*')) if p.is_file() and p.name!='OUTPUT_MANIFEST.json']
(OUT/'OUTPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(len(manifest),'recomputed outputs saved and hashed.')
