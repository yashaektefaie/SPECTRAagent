"""Disentangle covariate-shift NOVELTY from BLUR.
Key question: does a blur-orthogonalized novelty metric still degrade CONCH
within class? And per-class partial correlations with proper inference."""
import os, sys, json, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, rankdata, pearsonr
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(__file__))
from feats import NOVELTY_FEATS, BLUR_FEATS

ROOT="."; W=f"{ROOT}/work"
CLASSES=['ADI','BACK','DEB','LYM','MUC','MUS','NORM','STR','TUM']
nct=pd.read_csv(f"{W}/out/nct_features.csv")
cv=pd.read_csv(f"{W}/out/crcval_scored.csv")
R={}

# ---- blur-residualized (blur-free) novelty ----
# regress each color feature on the 3 blur metrics (fit on REFERENCE), residualize ref+test,
# then Mahalanobis from reference residual distribution.
Bref=sm.add_constant(nct[BLUR_FEATS].values)
Btst=sm.add_constant(cv[BLUR_FEATS].values)
res_ref=np.zeros((len(nct),len(NOVELTY_FEATS))); res_tst=np.zeros((len(cv),len(NOVELTY_FEATS)))
for j,f in enumerate(NOVELTY_FEATS):
    coef,_,_,_=np.linalg.lstsq(Bref,nct[f].values,rcond=None)
    res_ref[:,j]=nct[f].values-Bref@coef
    res_tst[:,j]=cv[f].values-Btst@coef
mu=res_ref.mean(0); VI=np.linalg.pinv(LedoitWolf().fit(res_ref).covariance_)
D=res_tst-mu
cv['novelty_bf']=np.sqrt(np.einsum('ij,jk,ik->i',D,VI,D))
R['corr_novelty_vs_noveltybf']=float(spearmanr(cv.novelty,cv.novelty_bf).correlation)
R['corr_noveltybf_vs_sharp']=float(spearmanr(cv.novelty_bf,cv.sharp if 'sharp' in cv else cv.lap_var).correlation)
cv['sharp']=cv['lap_var']

# ---- mean-only novelty (first moments, least blur-sensitive) ----
MEANF=['mean_R','mean_G','mean_B','mean_H','mean_E','mean_sat','mean_val','bg_frac']
Xr=nct[MEANF].values; Xt=cv[MEANF].values
mu2=Xr.mean(0); VI2=np.linalg.pinv(LedoitWolf().fit(Xr).covariance_); D2=Xt-mu2
cv['novelty_mean']=np.sqrt(np.einsum('ij,jk,ik->i',D2,VI2,D2))
R['corr_noveltymean_vs_sharp']=float(spearmanr(cv.novelty_mean,cv.sharp).correlation)

def pr(x): x=np.asarray(x,float); return (rankdata(x)-1)/(len(x)-1)
def within_tertile(d,col):
    lev=np.full(len(d),-1,int); d=d.reset_index(drop=True)
    for c in CLASSES:
        m=(d.label_name==c).values; v=d.loc[m,col].values
        if m.sum()<6: continue
        lev[np.where(m)[0]]=np.searchsorted(np.quantile(v,[1/3,2/3]),v)
    return lev
def macro_spc(d,lvl,ok='correct'):
    out={}
    for t in range(3):
        a=[d[(d.label_name==c)&(d[lvl]==t)][ok].mean() for c in CLASSES if ((d.label_name==c)&(d[lvl]==t)).sum()>=8]
        out[t]=float(np.mean(a))
    return out
def logit(df,terms,novcol='novelty'):
    X=pd.DataFrame(index=df.index)
    for t in terms:
        if t=='nov': X['nov']=pr(df[novcol])
        elif t=='sharp': X['sharp']=pr(df['sharp'])
        elif t=='class':
            for c in CLASSES[1:]: X['cls_'+c]=(df.label_name==c).astype(float)
    X=sm.add_constant(X); m=sm.Logit(df['correct'].values,X.values).fit(disp=0,maxiter=200)
    nm=list(X.columns)
    return {n:{'OR':float(np.exp(m.params[i])),'coef':float(m.params[i]),'p':float(m.pvalues[i])} for i,n in enumerate(nm) if n in('nov','sharp')}

# within-class SPC for each novelty variant
for tag,col in [('novelty_raw','novelty'),('novelty_blurfree','novelty_bf'),('novelty_meanonly','novelty_mean')]:
    cv['_l']=within_tertile(cv,col)
    R[f'SPC_{tag}_conch']=macro_spc(cv,'_l')
    R[f'logit_{tag}_class']=logit(cv,['nov','class'],col)
    R[f'logit_{tag}_sharp_class']=logit(cv,['nov','sharp','class'],col)

# ---- per-class partial Spearman with p-values ----
def partial_sp(x,y,z):
    rx,ry,rz=rankdata(x),rankdata(y),rankdata(z)
    def rr(a,b):
        B=np.c_[np.ones_like(b),b]; c,_,_,_=np.linalg.lstsq(B,a,rcond=None); return a-B@c
    ex,ey=rr(rx,rz),rr(ry,rz)
    if ex.std()<1e-9 or ey.std()<1e-9: return np.nan,np.nan
    r,p=pearsonr(ex,ey); return float(r),float(p)
percls={}
for c in CLASSES:
    s=cv[cv.label_name==c]
    if s.correct.nunique()<2 or len(s)<30:
        percls[c]={'n':int(len(s)),'note':'ceiling/small'}; continue
    nv_raw,p1=partial_sp(s.novelty.values,s.correct.values,s.sharp.values)
    nv_bf,p1b=partial_sp(s.novelty_bf.values,s.correct.values,s.sharp.values)
    sh,p2=partial_sp(s.sharp.values,s.correct.values,s.novelty.values)
    percls[c]={'n':int(len(s)),
               'rho_novelty_raw':float(spearmanr(s.novelty,s.correct).correlation),
               'partial_novelty_given_sharp':nv_raw,'p':p1,
               'partial_noveltyBF_given_sharp':nv_bf,'p_bf':p1b,
               'partial_sharp_given_novelty':sh,'p_sharp':p2,
               'corr_novelty_sharp':float(spearmanr(s.novelty,s.sharp).correlation)}
R['per_class_partials']=percls
# pooled (Fisher-z, weighted) for the blur-free novelty
def pooled(key):
    vals=[(np.arctanh(np.clip(percls[c][key],-.999,.999)),percls[c]['n']) for c in CLASSES if key in percls[c] and not np.isnan(percls[c][key])]
    z=sum(v*n for v,n in vals)/sum(n for _,n in vals); return float(np.tanh(z))
R['pooled_partial_noveltyBF_given_sharp']=pooled('partial_noveltyBF_given_sharp')
R['pooled_partial_novelty_given_sharp']=pooled('partial_novelty_given_sharp')
R['pooled_partial_sharp_given_novelty']=pooled('partial_sharp_given_novelty')

json.dump(R,open(f"{W}/out/results_disentangle.json","w"),indent=2,default=str)

def sp(d): return f"T1={d[0]:.3f} T2={d[1]:.3f} T3={d[2]:.3f} (drop {d[0]-d[2]:+.3f})"
print("corr(novelty_raw, novelty_blurfree):",round(R['corr_novelty_vs_noveltybf'],3))
print("corr(novelty_raw, sharp):", round(float(spearmanr(cv.novelty,cv.sharp).correlation),3),
      "| corr(novelty_blurfree, sharp):",round(R['corr_noveltybf_vs_sharp'],3),
      "| corr(novelty_meanonly, sharp):",round(R['corr_noveltymean_vs_sharp'],3))
print("\nWITHIN-CLASS macro SPC (T1 similar -> T3 novel):")
for tag in ['novelty_raw','novelty_blurfree','novelty_meanonly']:
    print(f"  {tag:18s}:", sp(R[f'SPC_{tag}_conch']))
print("\nCLASS-CONTROLLED logistic novelty OR (lower=more degradation):")
for tag in ['novelty_raw','novelty_blurfree','novelty_meanonly']:
    a=R[f'logit_{tag}_class']['nov']; b=R[f'logit_{tag}_sharp_class']['nov']
    print(f"  {tag:18s}: +class OR={a['OR']:.3f}(p={a['p']:.1e})   +sharp+class OR={b['OR']:.3f}(p={b['p']:.1e})")
print("\nPOOLED within-class partials: novelty|sharp=%.3f  noveltyBF|sharp=%.3f  sharp|novelty=%.3f"%(
    R['pooled_partial_novelty_given_sharp'],R['pooled_partial_noveltyBF_given_sharp'],R['pooled_partial_sharp_given_novelty']))
print("\nPER-CLASS partials (novel|sharp, noveltyBF|sharp, sharp|novel):")
for c in CLASSES:
    d=percls[c]
    if 'note' in d: print(f"  {c}: n={d['n']} {d['note']}"); continue
    print(f"  {c}: n={d['n']:4d}  nov|sharp={d['partial_novelty_given_sharp']:+.3f}(p{d['p']:.0e})  "
          f"novBF|sharp={d['partial_noveltyBF_given_sharp']:+.3f}(p{d['p_bf']:.0e})  "
          f"sharp|nov={d['partial_sharp_given_novelty']:+.3f}(p{d['p_sharp']:.0e})  r(nov,sharp)={d['corr_novelty_sharp']:+.2f}")
print("\nwrote out/results_disentangle.json")
