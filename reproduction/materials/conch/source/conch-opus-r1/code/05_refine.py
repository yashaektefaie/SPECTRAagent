"""Class-aware refinement of the CONCH train-novelty SPC.
- naive global axis is a tissue-type proxy (reported separately as failed axis)
- primary valid axis = within-class stain-novelty (class held constant)
- model-independent baseline within-class
- blur confound disentangled (class-controlled, both directions)
- confirmation on held-out half
Predictors rank-transformed to [0,1] for stable inference."""
import os, sys, json, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, rankdata, pearsonr
np.seterr(all='ignore')

ROOT="."; W=f"{ROOT}/work"
CLASSES=['ADI','BACK','DEB','LYM','MUC','MUS','NORM','STR','TUM']
cv=pd.read_csv(f"{W}/out/crcval_scored.csv")
R={}

def pr(x):  # percentile rank in [0,1]
    x=np.asarray(x,float); return (rankdata(x)-1)/(len(x)-1)

def within_class_tertile(d, col='novelty'):
    lev=np.full(len(d),-1,int); d=d.reset_index(drop=True)
    for c in CLASSES:
        m=(d.label_name==c).values; v=d.loc[m,col].values
        if m.sum()<6: continue
        q=np.quantile(v,[1/3,2/3]); lev[np.where(m)[0]]=np.searchsorted(q,v)
    return lev

# ---------- PRIMARY: within-class novelty tertile SPC (class held constant) ----------
cv['wlev']=within_class_tertile(cv,'novelty')
def macro_spc(d, lvlcol, okcol='correct'):
    out={}
    for t in range(3):
        accs=[]; ns=[]
        for c in CLASSES:
            s=d[(d.label_name==c)&(d[lvlcol]==t)]
            if len(s)>=8: accs.append(s[okcol].mean()); ns.append(len(s))
        out[t]={'macro_acc':float(np.mean(accs)),'n_classes':len(accs),'n_tiles':int(sum(ns))}
    return out
R['within_class_novelty_SPC_conch']=macro_spc(cv,'wlev','correct')
R['within_class_novelty_SPC_baseline']=macro_spc(cv,'wlev','base_correct')
# validity: similarity decreases (knn_ref_dist increases) across within-class tertiles
R['within_class_validity_knn']={t:float(cv[cv.wlev==t].knn_ref_dist.mean()) for t in range(3)}

# per-class novelty effect (spearman + tertile accs)
perc={}
for c in CLASSES:
    s=cv[cv.label_name==c]
    rr=spearmanr(s.novelty,s.correct)
    t_acc={t:float(s[s.wlev==t].correct.mean()) for t in range(3) if (s.wlev==t).any()}
    perc[c]={'n':int(len(s)),'spearman_rho':float(rr.correlation),'p':float(rr.pvalue),
             'tertile_acc':t_acc,
             'novelty_T1_to_T3_drop':float(t_acc.get(0,np.nan)-t_acc.get(2,np.nan))}
R['per_class_novelty']=perc

# ---------- class-controlled logistic regressions (rank predictors) ----------
def logit(df, terms):
    X=pd.DataFrame(index=df.index)
    for t in terms:
        if t=='nov': X['nov']=pr(df['novelty'])
        elif t=='sharp': X['sharp']=pr(df['sharp'])
        elif t=='class':
            for c in CLASSES[1:]: X['cls_'+c]=(df.label_name==c).astype(float)
    X=sm.add_constant(X)
    m=sm.Logit(df['correct'].values, X.values).fit(disp=0, maxiter=200)
    names=list(X.columns)
    return {n:{'coef':float(m.params[i]),'p':float(m.pvalues[i]),'OR':float(np.exp(m.params[i]))}
            for i,n in enumerate(names) if n in ('nov','sharp')}, float(m.prsquared)

cv['sharp']=cv['lap_var']
R['logit_nov_class'],_=logit(cv,['nov','class'])
R['logit_sharp_class'],_=logit(cv,['sharp','class'])
R['logit_nov_sharp_class'],_=logit(cv,['nov','sharp','class'])
R['logit_nov_only'],_=logit(cv,['nov'])
R['logit_nov_sharp_noclass'],_=logit(cv,['nov','sharp'])

# ---------- within-class partial Spearman (novelty|sharp) and (sharp|novelty) ----------
def partial_sp(x,y,z):
    rx,ry,rz=rankdata(x),rankdata(y),rankdata(z)
    def res(a,b):
        B=np.c_[np.ones_like(b),b]; c,_,_,_=np.linalg.lstsq(B,a,rcond=None); return a-B@c
    ex,ey=res(rx,rz),res(ry,rz)
    if ex.std()<1e-9 or ey.std()<1e-9: return np.nan,np.nan
    r,p=pearsonr(ex,ey); return float(r),float(p)
def pooled_partial(d, target='correct'):
    # Fisher-z weighted average of within-class partials
    rs=[]; w=[]
    nov_giv=[]; shp_giv=[]
    for c in CLASSES:
        s=d[d.label_name==c]
        if s[target].nunique()<2 or len(s)<30: continue
        a,_=partial_sp(s.novelty.values,s[target].values,s.sharp.values)  # nov|sharp
        b,_=partial_sp(s.sharp.values,s[target].values,s.novelty.values)  # sharp|nov
        if not np.isnan(a): nov_giv.append((np.arctanh(np.clip(a,-.999,.999)),len(s)))
        if not np.isnan(b): shp_giv.append((np.arctanh(np.clip(b,-.999,.999)),len(s)))
    def agg(lst):
        if not lst: return None
        z=sum(v*n for v,n in lst)/sum(n for _,n in lst); return float(np.tanh(z))
    return {'novelty_given_sharp':agg(nov_giv),'sharp_given_novelty':agg(shp_giv)}
R['within_class_partial_pooled']=pooled_partial(cv)
# also raw within-class novelty-sharp correlation
R['within_class_corr_nov_sharp']=float(np.mean([spearmanr(cv[cv.label_name==c].novelty,cv[cv.label_name==c].sharp).correlation for c in CLASSES]))

# ---------- BLUR as secondary corruption axis (class-controlled) ----------
cv['blev']=within_class_tertile(cv,'sharp')  # within-class sharpness tertiles (T0=blurriest)
R['within_class_sharp_SPC_conch']=macro_spc(cv,'blev','correct')  # T0 blurriest -> expect lower acc
R['per_class_sharp']={c:{'rho':float(spearmanr(cv[cv.label_name==c].sharp,cv[cv.label_name==c].correct).correlation),
                         'p':float(spearmanr(cv[cv.label_name==c].sharp,cv[cv.label_name==c].correct).pvalue)} for c in CLASSES}

# ---------- robustness: kNN-novelty instead of Mahalanobis ----------
cv['wlev_knn']=within_class_tertile(cv,'knn_ref_dist')
R['within_class_knnnovelty_SPC_conch']=macro_spc(cv,'wlev_knn','correct')
R['logit_knnnov_class'],_=logit(cv.assign(novelty=cv.knn_ref_dist),['nov','class'])

# ---------- CONFIRMATION: held-out half (seed 7), freeze within-class tertiles on discovery ----------
rng=np.random.RandomState(7); idx=rng.permutation(len(cv))
disc=cv.iloc[idx[:len(cv)//2]].copy(); conf=cv.iloc[idx[len(cv)//2:]].copy()
# freeze per-class novelty tertile thresholds on discovery, apply to confirmation
thr={}
for c in CLASSES:
    v=disc[disc.label_name==c].novelty.values
    thr[c]=np.quantile(v,[1/3,2/3]) if len(v)>=6 else None
def apply_thr(d):
    lev=np.full(len(d),-1,int); d=d.reset_index(drop=True)
    for c in CLASSES:
        if thr[c] is None: continue
        m=(d.label_name==c).values; lev[np.where(m)[0]]=np.searchsorted(thr[c],d.loc[m,'novelty'].values)
    return lev
conf['wlev']=apply_thr(conf); disc['wlev']=apply_thr(disc)
R['confirmation_within_class_novelty_SPC_conch']=macro_spc(conf,'wlev','correct')
R['confirmation_logit_nov_sharp_class'],_=logit(conf,['nov','sharp','class'])
R['confirmation_per_class_drop']={c:float(conf[(conf.label_name==c)&(conf.wlev==0)].correct.mean()
                                          -conf[(conf.label_name==c)&(conf.wlev==2)].correct.mean())
                                  for c in CLASSES if (conf.label_name==c).sum()>30}

json.dump(R,open(f"{W}/out/results_refined.json","w"),indent=2,default=str)

# ---------- print ----------
def sp(d): return " ".join(f"T{t+1}={d[t]['macro_acc']:.3f}(c{d[t]['n_classes']},n{d[t]['n_tiles']})" for t in range(3))
print("\n================ PRIMARY: WITHIN-CLASS NOVELTY SPC (class held constant) ================")
print("  validity knn_ref_dist by within-class tertile:", {t:round(R['within_class_validity_knn'][t],3) for t in range(3)})
print("  CONCH    macro-acc:", sp(R['within_class_novelty_SPC_conch']))
print("  BASELINE macro-acc:", sp(R['within_class_novelty_SPC_baseline']))
print("\n  per-class novelty (T1 most similar -> T3 most novel):")
for c in CLASSES:
    p=perc[c]; ta=p['tertile_acc']
    print(f"    {c}: n={p['n']:4d} rho={p['spearman_rho']:+.3f} p={p['p']:.1e}  acc T1/T2/T3="
          f"{ta.get(0,float('nan')):.3f}/{ta.get(1,float('nan')):.3f}/{ta.get(2,float('nan')):.3f}  drop={p['novelty_T1_to_T3_drop']:+.3f}")
print("\n================ CLASS-CONTROLLED LOGISTIC (rank predictors, OR over full range) ================")
def show(k):
    d=R[k]
    return "  ".join(f"{t}: OR={d[t]['OR']:.3f} (coef={d[t]['coef']:+.2f}, p={d[t]['p']:.1e})" for t in d)
print("  nov + class            :", show('logit_nov_class'))
print("  sharp + class          :", show('logit_sharp_class'))
print("  nov + sharp + class    :", show('logit_nov_sharp_class'))
print("  nov only (no class)    :", show('logit_nov_only'))
print("  nov + sharp (no class) :", show('logit_nov_sharp_noclass'))
print("\n  within-class corr(novelty,sharp):", round(R['within_class_corr_nov_sharp'],3))
print("  pooled within-class partial: novelty|sharp =", R['within_class_partial_pooled']['novelty_given_sharp'],
      " sharp|novelty =", R['within_class_partial_pooled']['sharp_given_novelty'])
print("\n================ BLUR as SECONDARY corruption axis (within-class sharpness tertiles, T1=blurriest) ================")
print("  CONCH macro-acc:", sp(R['within_class_sharp_SPC_conch']))
print("\n================ ROBUSTNESS: kNN-novelty ================")
print("  CONCH macro-acc:", sp(R['within_class_knnnovelty_SPC_conch']))
print("  logit knn-nov + class:", show('logit_knnnov_class'))
print("\n================ CONFIRMATION (held-out half, frozen discovery thresholds) ================")
print("  CONCH macro-acc:", sp(R['confirmation_within_class_novelty_SPC_conch']))
print("  logit nov+sharp+class:", show('confirmation_logit_nov_sharp_class'))
print("\nwrote out/results_refined.json")
