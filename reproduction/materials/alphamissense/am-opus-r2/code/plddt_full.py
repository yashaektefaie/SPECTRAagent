import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
rng=np.random.default_rng(11)
df=pd.read_parquet('master_plddt.parquet').dropna(subset=['plddt'])
meta=pd.read_csv('assay_meta.csv')[['assay','coarse_selection_type']]
NMIN=50
def sp(g,col):
    if len(g)>=NMIN and g.DMS_score.nunique()>5 and g[col].nunique()>2:
        return abs(spearmanr(g[col],g.DMS_score)[0])
    return np.nan

# ---- FAIR HORSE RACE: cells = assay x coverage-sextile AND assay x plddt-sextile; both continuous means ----
def horse(binby):
    rows=[]
    for a,g in df.groupby('assay'):
        g=g.copy()
        try: g['cell']=pd.qcut(g[binby],6,labels=False,duplicates='drop')
        except: continue
        for c,gg in g.groupby('cell'):
            rows.append(dict(assay=a,am=sp(gg,'am_pathogenicity'),
                cov=gg.coverage.mean(),plddt=gg.plddt.mean(),ldms=np.log(gg.DMS_score.std()+1e-9)))
    R=pd.DataFrame(rows).dropna()
    R=R[np.isfinite(R.ldms)]
    def ols(cols):
        X=R.copy()
        for c in ['am']+cols: X[c+'_c']=X[c]-X.groupby('assay')[c].transform('mean')
        M=np.column_stack([np.ones(len(X))]+[X[c+'_c'].values for c in cols]); y=X['am_c'].values
        b,_,_,_=np.linalg.lstsq(M,y,rcond=None); r=y-M@b
        s2=(r@r)/(len(X)-M.shape[1]); se=np.sqrt(np.diag(s2*np.linalg.inv(M.T@M)))
        return {c:(b[i+1],b[i+1]/se[i+1]) for i,c in enumerate(cols)},len(X)
    o,n=ols(['cov','plddt','ldms'])
    print(f"  cells binned by {binby:8s} (n={n}): cov beta(t)={o['cov'][0]:+.4f}({o['cov'][1]:.2f})  plddt beta(t)={o['plddt'][0]:+.5f}({o['plddt'][1]:.2f})")
print("=== FAIR HORSE RACE (both continuous, assay FE, ctrl logDMSsd) ===")
horse('coverage'); horse('plddt')

# ---- pLDDT axis full characterization ----
def plv(x): return 0 if x<50 else (1 if x<70 else (2 if x<90 else 3))
df['pl']=df.plddt.apply(plv)
cells={}
for (a,l),g in df.groupby(['assay','pl']):
    cells[(a,l)]=dict(am=sp(g,'am_pathogenicity'),bl=sp(g,'blosum'),ps=sp(g,'pssm'))
assays=df.assay.unique()
def drop(alist,key,hi=3,lo=0):
    ds=[cells[(a,hi)][key]-cells[(a,lo)][key] for a in alist
        if (a,hi) in cells and (a,lo) in cells and cells[(a,hi)][key]==cells[(a,hi)][key] and cells[(a,lo)][key]==cells[(a,lo)][key]]
    return np.array(ds)
print("\n=== pLDDT axis: high(>=90) - disorder(<50) drop, cluster-bootstrap 95% CI ===")
for key,nm in [('am','AM'),('bl','BLOSUM'),('ps','PSSM')]:
    d=drop(assays,key); bs=[np.mean(drop(rng.choice(assays,len(assays),True),key)) for _ in range(2000)]
    lo,hi=np.percentile(bs,[2.5,97.5])
    print(f"  {nm:7s} drop={d.mean():+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] n={len(d)} up={(d>0).sum()}")
# differential degradation (AM-specificity)
print("\n=== AM-specificity on pLDDT axis (paired differential degradation) ===")
common=[a for a in assays if all((a,l) in cells and cells[(a,l)]['am']==cells[(a,l)]['am'] and cells[(a,l)]['bl']==cells[(a,l)]['bl'] and cells[(a,l)]['ps']==cells[(a,l)]['ps'] for l in [0,3])]
da=np.array([cells[(a,3)]['am']-cells[(a,0)]['am'] for a in common])
db=np.array([cells[(a,3)]['bl']-cells[(a,0)]['bl'] for a in common])
dp=np.array([cells[(a,3)]['ps']-cells[(a,0)]['ps'] for a in common])
print(f"  n={len(common)}: AM drop-BLOSUM drop = {np.mean(da-db):+.3f} (median {np.median(da-db):+.3f}) p={wilcoxon(da-db).pvalue:.3f}")
print(f"           AM drop-PSSM drop   = {np.mean(da-dp):+.3f} (median {np.median(da-dp):+.3f}) p={wilcoxon(da-dp).pvalue:.3f}")

# ---- held-out confirmation by protein ----
accs=np.array(sorted(df.acc.unique())); r2=np.random.default_rng(7); r2.shuffle(accs)
for name,sset_ in [('DISCOVERY',set(accs[:len(accs)//2])),('CONFIRM',set(accs[len(accs)//2:]))]:
    aa=list(df[df.acc.isin(sset_)].assay.unique())
    d=drop(aa,'am'); print(f"  HELD-OUT {name:9s}: AM disorder-drop={d.mean():+.3f} n={len(d)} up={(d>0).sum()}")
