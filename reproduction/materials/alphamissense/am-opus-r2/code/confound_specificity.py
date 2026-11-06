import numpy as np, pandas as pd
from scipy.stats import wilcoxon
R=pd.read_csv('spc_by_assay_level.csv')

def piv(m): return R.pivot(index='assay',columns='lvl',values=m)
pa,pb,pp=piv('acc_am'),piv('acc_bl'),piv('acc_ps')
J=pd.DataFrame({
 'am1':pa['L1_high'],'am3':pa['L3_low'],
 'bl1':pb['L1_high'],'bl3':pb['L3_low'],
 'ps1':pp['L1_high'],'ps3':pp['L3_low']}).dropna()
J['drop_am']=J.am1-J.am3; J['drop_bl']=J.bl1-J.bl3; J['drop_ps']=J.ps1-J.ps3
J['am_minus_bl']=J.drop_am-J.drop_bl; J['am_minus_ps']=J.drop_am-J.drop_ps
print(f"Differential degradation (paired, n={len(J)} assays with L1&L3 for all 3 methods):")
print(f"  AM drop={J.drop_am.mean():.3f}  BLOSUM drop={J.drop_bl.mean():.3f}  PSSM drop={J.drop_ps.mean():.3f}")
for c,lbl in [('am_minus_bl','AM drop - BLOSUM drop'),('am_minus_ps','AM drop - PSSM drop')]:
    w=wilcoxon(J[c]); print(f"  {lbl:22s} mean={J[c].mean():+.3f} median={J[c].median():+.3f} p={w.pvalue:.3f}")

def within_ols(m):
    dd=R.dropna(subset=[m,'dms_sd']).copy(); dd=dd[dd.dms_sd>0]
    dd['covrank']=dd['lvl'].map({'L3_low':0,'L2_mid':1,'L1_high':2}); dd['ldms']=np.log(dd.dms_sd)
    for col in [m,'covrank','ldms']:
        dd[col+'_c']=dd[col]-dd.groupby('assay')[col].transform('mean')
    X=np.c_[np.ones(len(dd)),dd['covrank_c'],dd['ldms_c']]; y=dd[m+'_c'].values
    b,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    r=y-X@b; s2=(r@r)/(len(dd)-3); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
    return b,se,len(dd)
print("\nWithin-assay OLS (assay fixed effects): acc ~ coverage_rank(0,1,2) + log(DMS_sd)")
print("  coverage_rank>0 => accuracy rises with coverage AFTER removing DMS-variance effect")
for m,nm in [('acc_am','AM'),('acc_bl','BLOSUM'),('acc_ps','PSSM')]:
    b,se,n=within_ols(m)
    print(f"  {nm:7s} n={n}: beta_cov={b[1]:+.4f} (t={b[1]/se[1]:.2f})   beta_logDMSsd={b[2]:+.4f} (t={b[2]/se[2]:.2f})")
