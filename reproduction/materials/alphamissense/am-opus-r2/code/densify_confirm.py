import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
rng=np.random.default_rng(42)
df=pd.read_parquet('master.parquet')
meta=pd.read_csv('assay_meta.csv')[['assay','coarse_selection_type']]
df=df.merge(meta,on='assay',how='left')

# ---- Densified coverage levels (6 ordered bins by absolute coverage) ----
edges=[-1e-9,0.10,0.50,0.80,0.95,0.99,1.01]
labs=['B0_<.10','B1_.10-.50','B2_.50-.80','B3_.80-.95','B4_.95-.99','B5_>=.99']
df['cb']=pd.cut(df.coverage,edges,labels=labs)
NMIN=50
def sp(g,col):
    if len(g)>=NMIN and g.DMS_score.nunique()>5 and g[col].nunique()>2:
        return abs(spearmanr(g[col],g.DMS_score)[0])
    return np.nan
rows=[]
for (a,b),g in df.groupby(['assay','cb'],observed=True):
    rows.append(dict(assay=a,cb=b,n=len(g),cov=g.coverage.mean(),
        am=sp(g,'am_pathogenicity'),bl=sp(g,'blosum'),ps=sp(g,'pssm')))
D=pd.DataFrame(rows)

print("=== DENSIFIED SPC (mean per-assay accuracy per coverage bin) ===")
print("bin           covmean ncells variants     AM  BLOSUM   PSSM")
for bb in labs:
    s=D[D.cb==bb]
    cov=s["cov"].mean(); nc=int(s["am"].notna().sum()); nv=int(s["n"].sum())
    am=s["am"].mean(); bl=s["bl"].mean(); ps=s["ps"].mean()
    print("%-12s %7.3f %6d %8d %6.3f %7.3f %6.3f"%(bb,cov,nc,nv,am,bl,ps))

# ---- Cluster bootstrap CIs over assays for L1(>=.95) vs L3(<.50) drop (from frozen 3-level) ----
def lvl(c): return 'H' if c>=0.95 else ('L' if c<0.50 else 'M')
df['lv']=df.coverage.apply(lvl)
cells={}
for (a,l),g in df.groupby(['assay','lv']):
    cells[(a,l)]=dict(am=sp(g,'am_pathogenicity'),bl=sp(g,'blosum'),ps=sp(g,'pssm'))
assays=df.assay.unique()
def paired_drop(assay_list,key):
    ds=[]
    for a in assay_list:
        h=cells.get((a,'H'),{}).get(key); l=cells.get((a,'L'),{}).get(key)
        if h==h and l==l and h is not None and l is not None: ds.append(h-l)
    return np.mean(ds) if ds else np.nan, len(ds)
def boot_ci(key):
    obs,n=paired_drop(assays,key)
    bs=[]
    for _ in range(2000):
        samp=rng.choice(assays,len(assays),replace=True)
        m,_=paired_drop(samp,key); bs.append(m)
    lo,hi=np.percentile(bs,[2.5,97.5]); return obs,n,lo,hi
print("\n=== High(>=.95) - Low(<.50) accuracy drop, cluster-bootstrap 95% CI (over assays) ===")
for key,nm in [('am','AM'),('bl','BLOSUM'),('ps','PSSM')]:
    obs,n,lo,hi=boot_ci(key)
    print(f"  {nm:7s} drop={obs:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  (n={n} paired assays)")

# ---- HELD-OUT CONFIRMATION: split by protein ----
accs=np.array(sorted(df.acc.unique())); rng2=np.random.default_rng(7); rng2.shuffle(accs)
disc=set(accs[:len(accs)//2]); conf=set(accs[len(accs)//2:])
def report_split(mask_accs,name):
    sub=df[df.acc.isin(mask_accs)]
    aa=sub.assay.unique()
    d_am,n=paired_drop(aa,'am'); d_bl,_=paired_drop(aa,'bl'); d_ps,_=paired_drop(aa,'ps')
    # within-assay OLS slope (assay FE) controlling DMS_sd on this split
    R=[]
    for (a,l),g in sub.groupby(['assay','lv']):
        R.append(dict(assay=a,lv=l,am=sp(g,'am_pathogenicity'),dms_sd=g.DMS_score.std()))
    R=pd.DataFrame(R).dropna(subset=['am']); R=R[R.dms_sd>0]
    R['cr']=R.lv.map({'L':0,'M':1,'H':2}); R['ld']=np.log(R.dms_sd)
    for c in ['am','cr','ld']: R[c+'_c']=R[c]-R.groupby('assay')[c].transform('mean')
    X=np.c_[np.ones(len(R)),R.cr_c,R.ld_c]; y=R.am_c.values
    b,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    r=y-X@b; s2=(r@r)/(len(R)-3); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
    print(f"  {name:12s} nprot={len(mask_accs)} nassay={len(aa)}: AM drop(H-L)={d_am:+.3f}(n={n})  AM cov-slope(DMSctrl)={b[1]:+.4f}(t={b[1]/se[1]:.2f})")
print("\n=== HELD-OUT CONFIRMATION (protein split) ===")
report_split(disc,'DISCOVERY')
report_split(conf,'CONFIRMATION')
D.to_csv('densified_spc.csv',index=False)
