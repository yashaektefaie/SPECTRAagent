import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
rng=np.random.default_rng(1)
df=pd.read_parquet('master_plddt.parquet')
df=df.dropna(subset=['plddt'])
NMIN=50
def sp(g,col):
    if len(g)>=NMIN and g.DMS_score.nunique()>5 and g[col].nunique()>2:
        return abs(spearmanr(g[col],g.DMS_score)[0])
    return np.nan

# pLDDT ordered levels
def plv(x): return 'P1_disorder(<50)' if x<50 else ('P2(50-70)' if x<70 else ('P3(70-90)' if x<90 else 'P4_high(>=90)'))
df['plv']=df.plddt.apply(plv)
labs=['P1_disorder(<50)','P2(50-70)','P3(70-90)','P4_high(>=90)']
rows=[]
for (a,l),g in df.groupby(['assay','plv']):
    rows.append(dict(assay=a,plv=l,n=len(g),plddt=g.plddt.mean(),
        am=sp(g,'am_pathogenicity'),bl=sp(g,'blosum'),ps=sp(g,'pssm'),dms_sd=g.DMS_score.std()))
P=pd.DataFrame(rows)
print("=== pLDDT AXIS: mean per-assay accuracy per level ===")
print("level                plddt ncells variants     AM  BLOSUM   PSSM")
for l in labs:
    s=P[P.plv==l]
    print("%-20s %5.1f %6d %8d %6.3f %7.3f %6.3f"%(l,s['plddt'].mean(),int(s['am'].notna().sum()),int(s['n'].sum()),s['am'].mean(),s['bl'].mean(),s['ps'].mean()))

# paired disorder(<50) vs high(>=90)
pa=P.pivot(index='assay',columns='plv',values='am')
J=pa[['P4_high(>=90)','P1_disorder(<50)']].dropna()
d=J['P4_high(>=90)']-J['P1_disorder(<50)']
print("\nWithin-assay paired AM: high(>=90) - disorder(<50): mean=%+.3f median=%+.3f p=%.2e n=%d up=%d"%(
    d.mean(),d.median(),wilcoxon(J['P4_high(>=90)'],J['P1_disorder(<50)']).pvalue,len(J),(d>0).sum()))
for nm,mc in [('BLOSUM','bl'),('PSSM','ps')]:
    pv=P.pivot(index='assay',columns='plv',values=mc)[['P4_high(>=90)','P1_disorder(<50)']].dropna()
    dd=pv['P4_high(>=90)']-pv['P1_disorder(<50)']
    print("  %-7s drop=%+.3f p=%.2e n=%d"%(nm,dd.mean(),wilcoxon(pv.iloc[:,0],pv.iloc[:,1]).pvalue,len(pv)))

# ---- SYMMETRIC COLLINEARITY TEST at cell level (assay FE, control DMS_sd) ----
def celltab():
    df['cvr']=df.coverage.apply(lambda c:0 if c<0.5 else (1 if c<0.95 else 2))
    df['pvr']=df.plddt.apply(lambda x:0 if x<50 else (1 if x<70 else (2 if x<90 else 3)))
    R=[]
    # use 3x cov-level cells (as before) with mean plddt
    for (a,l),g in df.groupby(['assay','cvr']):
        R.append(dict(assay=a,cell='cov%d'%l,am=sp(g,'am_pathogenicity'),
                      covr=l,plddt=g.plddt.mean(),ldms=np.log(g.DMS_score.std())))
    return pd.DataFrame(R).dropna(subset=['am','plddt','ldms'])
def ols(R,cols):
    for c in ['am']+cols: R[c+'_c']=R[c]-R.groupby('assay')[c].transform('mean')
    X=np.column_stack([np.ones(len(R))]+[R[c+'_c'].values for c in cols]); y=R['am_c'].values
    b,_,_,_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b
    s2=(r@r)/(len(R)-X.shape[1]); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
    return {c:(b[i+1],b[i+1]/se[i+1]) for i,c in enumerate(cols)}, len(R)
R=celltab()
print("\n=== SYMMETRIC test (cells=assay x coverage-tertile; assay FE; ctrl log DMS_sd) ===")
o,n=ols(R.copy(),['covr','ldms']);   print("  coverage alone:            covr beta=%+.4f (t=%.2f)  n=%d"%(*o['covr'],n))
o,n=ols(R.copy(),['plddt','ldms']);  print("  pLDDT alone:               plddt beta=%+.5f (t=%.2f) n=%d"%(*o['plddt'],n))
o,n=ols(R.copy(),['covr','plddt','ldms']); print("  BOTH:  covr beta=%+.4f (t=%.2f) ; plddt beta=%+.5f (t=%.2f)"%(*o['covr'],*o['plddt']))
