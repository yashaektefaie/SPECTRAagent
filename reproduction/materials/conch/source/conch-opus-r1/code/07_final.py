"""Final consolidation:
 - model-specificity: baseline (reference-trained) degradation for blur-free novelty AND sharpness
 - held-out confirmation of BOTH axes (blur-free novelty + sharpness), frozen on discovery
 - head-to-head incremental pseudo-R2 (novelty vs blur) on top of class
"""
import os, sys, json, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from scipy.stats import rankdata, spearmanr
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(__file__))
from feats import NOVELTY_FEATS, BLUR_FEATS
ROOT="."; W=f"{ROOT}/work"
CLASSES=['ADI','BACK','DEB','LYM','MUC','MUS','NORM','STR','TUM']
nct=pd.read_csv(f"{W}/out/nct_features.csv"); cv=pd.read_csv(f"{W}/out/crcval_scored.csv")
cv['sharp']=cv['lap_var']; R={}

# rebuild blur-free novelty (same as 06)
Bref=sm.add_constant(nct[BLUR_FEATS].values); Btst=sm.add_constant(cv[BLUR_FEATS].values)
rr=np.zeros((len(cv),len(NOVELTY_FEATS))); rf=np.zeros((len(nct),len(NOVELTY_FEATS)))
for j,f in enumerate(NOVELTY_FEATS):
    c,_,_,_=np.linalg.lstsq(Bref,nct[f].values,rcond=None); rf[:,j]=nct[f].values-Bref@c; rr[:,j]=cv[f].values-Btst@c
mu=rf.mean(0); VI=np.linalg.pinv(LedoitWolf().fit(rf).covariance_); D=rr-mu
cv['novelty_bf']=np.sqrt(np.einsum('ij,jk,ik->i',D,VI,D))

def within_tertile(d,col):
    lev=np.full(len(d),-1,int); d=d.reset_index(drop=True)
    for c in CLASSES:
        m=(d.label_name==c).values; v=d.loc[m,col].values
        if m.sum()<6: continue
        lev[np.where(m)[0]]=np.searchsorted(np.quantile(v,[1/3,2/3]),v)
    return lev
def macro_spc(d,lvl,ok='correct'):
    return {t:float(np.mean([d[(d.label_name==c)&(d[lvl]==t)][ok].mean()
            for c in CLASSES if ((d.label_name==c)&(d[lvl]==t)).sum()>=8])) for t in range(3)}
def pr(x): x=np.asarray(x,float); return (rankdata(x)-1)/(len(x)-1)
def fit(df,cols):
    X=pd.DataFrame(index=df.index)
    for t in cols:
        if t=='nov': X['nov']=pr(df['novelty_bf'])
        elif t=='sharp': X['sharp']=pr(df['sharp'])
        elif t=='class':
            for c in CLASSES[1:]: X['c_'+c]=(df.label_name==c).astype(float)
    X=sm.add_constant(X); return sm.Logit(df['correct'].values,X.values).fit(disp=0,maxiter=200)

# ---- model specificity: baseline vs CONCH for blur-free novelty and sharpness ----
for axis,col,order in [('blurfree_novelty','novelty_bf','similar->novel'),('sharpness','sharp','blurry->sharp')]:
    cv['_l']=within_tertile(cv,col)
    R[f'{axis}_SPC_conch']=macro_spc(cv,'_l','correct')
    R[f'{axis}_SPC_baseline']=macro_spc(cv,'_l','base_correct')
    R[f'{axis}_order']=order

# ---- head-to-head incremental pseudo-R2 over class baseline ----
base=fit(cv,['class']); pr2_base=base.prsquared
pr2_nov=fit(cv,['class','nov']).prsquared
pr2_sharp=fit(cv,['class','sharp']).prsquared
pr2_both=fit(cv,['class','nov','sharp']).prsquared
R['pseudoR2']={'class':float(pr2_base),'class+nov':float(pr2_nov),'class+sharp':float(pr2_sharp),
               'class+nov+sharp':float(pr2_both),
               'delta_nov_over_class':float(pr2_nov-pr2_base),'delta_sharp_over_class':float(pr2_sharp-pr2_base),
               'delta_nov_given_sharp':float(pr2_both-pr2_sharp),'delta_sharp_given_nov':float(pr2_both-pr2_nov)}

# ---- held-out confirmation of BOTH axes (freeze tertile thresholds on discovery) ----
rng=np.random.RandomState(7); idx=rng.permutation(len(cv))
disc=cv.iloc[idx[:len(cv)//2]].copy(); conf=cv.iloc[idx[len(cv)//2:]].copy()
def confirm(col):
    thr={c:(np.quantile(disc[disc.label_name==c][col].values,[1/3,2/3]) if (disc.label_name==c).sum()>=6 else None) for c in CLASSES}
    def app(d):
        lev=np.full(len(d),-1,int); d=d.reset_index(drop=True)
        for c in CLASSES:
            if thr[c] is None: continue
            m=(d.label_name==c).values; lev[np.where(m)[0]]=np.searchsorted(thr[c],d.loc[m,col].values)
        return lev
    conf['_lc']=app(conf)
    return macro_spc(conf,'_lc','correct')
R['confirm_blurfree_novelty_SPC']=confirm('novelty_bf')
R['confirm_sharpness_SPC']=confirm('sharp')
# held-out joint logit (blur-free novelty + sharp + class)
m=fit(conf if False else conf.assign(),['class','nov','sharp'])  # conf already has cols
mc=fit(conf,['class','nov','sharp']); names=['const']+[c for c in ['nov','sharp']]
# extract nov & sharp from mc by refitting with known column order
Xc=pd.DataFrame(index=conf.index); Xc['nov']=pr(conf['novelty_bf']); Xc['sharp']=pr(conf['sharp'])
for c in CLASSES[1:]: Xc['c_'+c]=(conf.label_name==c).astype(float)
Xc=sm.add_constant(Xc); mm=sm.Logit(conf['correct'].values,Xc.values).fit(disp=0,maxiter=200)
cols=list(Xc.columns)
R['confirm_joint_logit']={n:{'OR':float(np.exp(mm.params[i])),'p':float(mm.pvalues[i])} for i,n in enumerate(cols) if n in('nov','sharp')}

json.dump(R,open(f"{W}/out/results_final.json","w"),indent=2,default=str)
def s(d): return f"T1={d[0]:.3f} T2={d[1]:.3f} T3={d[2]:.3f} (Δ {d[0]-d[2]:+.3f})"
print("MODEL SPECIFICITY (within-class macro SPC, CONCH vs reference-trained baseline):")
print("  BLUR-FREE NOVELTY (similar->novel):")
print("    CONCH   :", s(R['blurfree_novelty_SPC_conch']))
print("    baseline:", s(R['blurfree_novelty_SPC_baseline']))
print("  SHARPNESS (blurry->sharp):")
print("    CONCH   :", s(R['sharpness_SPC_conch']))
print("    baseline:", s(R['sharpness_SPC_baseline']))
print("\nHEAD-TO-HEAD incremental McFadden pseudo-R2 (over class-only model):")
p=R['pseudoR2']
print(f"  class={p['class']:.4f}  +nov={p['class+nov']:.4f}  +sharp={p['class+sharp']:.4f}  +both={p['class+nov+sharp']:.4f}")
print(f"  Δ nov|class={p['delta_nov_over_class']:.4f}  Δ sharp|class={p['delta_sharp_over_class']:.4f}")
print(f"  Δ nov|class+sharp={p['delta_nov_given_sharp']:.4f}  Δ sharp|class+nov={p['delta_sharp_given_nov']:.4f}")
print("\nHELD-OUT CONFIRMATION (frozen discovery tertiles):")
print("  blur-free novelty:", s(R['confirm_blurfree_novelty_SPC']))
print("  sharpness        :", s(R['confirm_sharpness_SPC']))
print("  joint logit (held-out): nov OR=%.3f(p=%.1e)  sharp OR=%.3f(p=%.1e)"%(
    R['confirm_joint_logit']['nov']['OR'],R['confirm_joint_logit']['nov']['p'],
    R['confirm_joint_logit']['sharp']['OR'],R['confirm_joint_logit']['sharp']['p']))
print("\nwrote out/results_final.json")
