import numpy as np, pandas as pd, glob, os
from scipy.stats import spearmanr
df=pd.read_parquet('master.parquet')
meta=pd.read_csv('assay_meta.csv')[['assay','coarse_selection_type']]
df=df.merge(meta,on='assay',how='left')

# ---- extract per-residue pLDDT (CA B-factor) from AF pdbs ----
plddt={}
for f in glob.glob('af/*.pdb'):
    acc=os.path.basename(f)[:-4]; d={}
    for line in open(f):
        if line.startswith('ATOM') and line[12:16].strip()=='CA':
            resi=int(line[22:26]); b=float(line[60:66]); d[resi]=b
    plddt[acc]=d
print('AF proteins with pLDDT:',len(plddt))
# map (acc,position)->plddt ; position is UniProt residue number == target_seq index (verified)
df['plddt']=[plddt.get(a,{}).get(p,np.nan) for a,p in zip(df.acc,df.position)]
df['lv']=df.coverage.apply(lambda c:'H' if c>=0.95 else ('L' if c<0.50 else 'M'))
cov_ok=df.dropna(subset=['plddt'])
print('variants with pLDDT:',len(cov_ok),'/',len(df))
# coverage vs pLDDT correlation (per-variant)
r,_=spearmanr(cov_ok.coverage,cov_ok.plddt)
print('Spearman(coverage, pLDDT) = %.3f  (are they the same axis?)'%r)

NMIN=50
def sp(g,col):
    if len(g)>=NMIN and g.DMS_score.nunique()>5 and g[col].nunique()>2:
        return abs(spearmanr(g[col],g.DMS_score)[0])
    return np.nan
def lvl(c): return 'H' if c>=0.95 else ('L' if c<0.50 else 'M')
df['lv']=df.coverage.apply(lvl)

def ols_slope(sub, ctrl_plddt=False):
    R=[]
    for (a,l),g in sub.groupby(['assay','lv']):
        R.append(dict(assay=a,lv=l,am=sp(g,'am_pathogenicity'),
                      dms_sd=g.DMS_score.std(),plddt=g.plddt.mean()))
    R=pd.DataFrame(R).dropna(subset=['am']); R=R[R.dms_sd>0]
    R['cr']=R.lv.map({'L':0,'M':1,'H':2}); R['ld']=np.log(R.dms_sd)
    cols=['cr','ld']+(['plddt'] if ctrl_plddt else [])
    R=R.dropna(subset=cols+['am'])
    for c in ['am']+cols: R[c+'_c']=R[c]-R.groupby('assay')[c].transform('mean')
    X=np.column_stack([np.ones(len(R))]+[R[c+'_c'].values for c in cols]); y=R['am_c'].values
    b,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    r=y-X@b; s2=(r@r)/(len(R)-X.shape[1]); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
    return b[1],se[1],len(R)

print("\n=== CONFOUND: Stability phenotype ===")
b,se,n=ols_slope(df); print("  ALL assays:            AM cov-slope=%+.4f (t=%.2f, n=%d)"%(b,b/se,n))
b,se,n=ols_slope(df[df.coarse_selection_type!='Stability']); print("  EXCLUDING Stability:   AM cov-slope=%+.4f (t=%.2f, n=%d)"%(b,b/se,n))
b,se,n=ols_slope(df[df.coarse_selection_type=='Stability']); print("  ONLY Stability:        AM cov-slope=%+.4f (t=%.2f, n=%d)"%(b,b/se,n))

print("\n=== CONFOUND: disorder (pLDDT) — does coverage effect survive controlling pLDDT? ===")
b,se,n=ols_slope(cov_ok,ctrl_plddt=False); print("  cov-slope (no pLDDT ctrl):   %+.4f (t=%.2f, n=%d)"%(b,b/se,n))
b,se,n=ols_slope(cov_ok,ctrl_plddt=True);  print("  cov-slope (pLDDT controlled):%+.4f (t=%.2f, n=%d)"%(b,b/se,n))
df.to_parquet('master_plddt.parquet')
