import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
df=pd.read_parquet('master.parquet')
meta=pd.read_csv('assay_meta.csv')[['assay','coarse_selection_type','seq_len']]
df=df.merge(meta,on='assay',how='left')

def level(c):
    if c>=0.95: return 'L1_high'
    if c>=0.50: return 'L2_mid'
    return 'L3_low'
df['lvl']=df.coverage.apply(level)

NMIN=50
def sp(g,col):
    if len(g)>=NMIN and g.DMS_score.nunique()>5 and g[col].nunique()>2:
        r,_=spearmanr(g[col],g.DMS_score); return abs(r)
    return np.nan

rows=[]
for (a,l),g in df.groupby(['assay','lvl']):
    rows.append(dict(assay=a,lvl=l,n=len(g),cov=g.coverage.mean(),
                     dms_sd=g.DMS_score.std(),
                     stab=(g.coarse_selection_type=='Stability').iloc[0],
                     acc_am=sp(g,'am_pathogenicity'),
                     acc_bl=sp(g,'blosum'),
                     acc_ps=sp(g,'pssm')))
R=pd.DataFrame(rows)
R.to_csv('spc_by_assay_level.csv',index=False)

print("=== LEVEL ORDERING (measured coverage decreases?) ===")
print(R.groupby('lvl').agg(n_cells=('assay','count'),
      variants=('n','sum'),cov_mean=('cov','mean'),
      dms_sd=('dms_sd','median')).to_string())

print("\n=== per-level MEAN accuracy across assays (cells with valid Spearman) ===")
for m,nm in [('acc_am','AM'),('acc_bl','BLOSUM'),('acc_ps','PSSM')]:
    g=R.groupby('lvl')[m].agg(['mean','median','count'])
    print(f'{nm:7s}', {l:(round(g.loc[l,"mean"],3),round(g.loc[l,"median"],3),int(g.loc[l,"count"])) for l in ['L1_high','L2_mid','L3_low'] if l in g.index})

# WITHIN-ASSAY PAIRED L1 vs L3
piv_am=R.pivot(index='assay',columns='lvl',values='acc_am')
piv_bl=R.pivot(index='assay',columns='lvl',values='acc_bl')
piv_ps=R.pivot(index='assay',columns='lvl',values='acc_ps')
both=piv_am[['L1_high','L3_low']].dropna()
print(f"\n=== WITHIN-ASSAY PAIRED L1(high) vs L3(low), assays with both & valid: n={len(both)} ===")
for nm,piv in [('AM',piv_am),('BLOSUM',piv_bl),('PSSM',piv_ps)]:
    p=piv[['L1_high','L3_low']].dropna()
    d=p['L1_high']-p['L3_low']
    w=wilcoxon(p['L1_high'],p['L3_low']) if len(p)>5 else None
    print(f'{nm:7s} n={len(p):2d} mean L1={p.L1_high.mean():.3f} L3={p.L3_low.mean():.3f} drop(L1-L3)={d.mean():+.3f} median_drop={d.median():+.3f} p={w.pvalue:.2e} up={ (d>0).sum()}/{len(p)}')

# AM-specificity: Delta(AM-BLOSUM) per level (paired within assay-level cell)
R['d_am_bl']=R.acc_am-R.acc_bl
R['d_am_ps']=R.acc_am-R.acc_ps
print("\n=== AM edge over baselines per level (mean of per-cell diff) ===")
print(R.groupby('lvl')[['d_am_bl','d_am_ps']].mean().to_string())
