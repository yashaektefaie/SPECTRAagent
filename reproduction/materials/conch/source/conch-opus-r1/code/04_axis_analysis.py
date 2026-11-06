"""Core SPECTRA analysis: build frozen novelty axis, validate, baseline, CONCH SPC,
confounds (blur + class + bg), confirmation. Writes out/results.json + prints summary."""
import os, sys, json, numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(__file__))
from feats import NOVELTY_FEATS, BLUR_FEATS

ROOT = "."
W = f"{ROOT}/work"
CLASSES = ['ADI','BACK','DEB','LYM','MUC','MUS','NORM','STR','TUM']
R = {}

# ---------- load ----------
nct = pd.read_csv(f"{W}/out/nct_features.csv")
cv  = pd.read_csv(f"{W}/out/crcval_features.csv")
preds = pd.read_csv(f"{ROOT}/data/predictions.csv")
cv = cv.merge(preds[['image_path','correct','label_name','pred_label_name']],
              left_on='id', right_on='image_path', how='left')
assert cv['correct'].notna().all()
print(f"NCT ref tiles: {len(nct)}  | CRC-VAL test tiles: {len(cv)}")
R['n_nct'] = int(len(nct)); R['n_test'] = int(len(cv))
R['nct_per_class'] = nct['cls'].value_counts().to_dict()

X_ref = nct[NOVELTY_FEATS].values.astype(float)
X_tst = cv[NOVELTY_FEATS].values.astype(float)

# ---------- novelty axis (frozen): Mahalanobis from NCT reference ----------
mu = X_ref.mean(0)
lw = LedoitWolf().fit(X_ref)
VI = np.linalg.pinv(lw.covariance_)
def mahal(X):
    D = X - mu
    return np.sqrt(np.einsum('ij,jk,ik->i', D, VI, D))
cv['novelty'] = mahal(X_tst)
ref_nov = mahal(X_ref)  # reference self-distance for context

# secondary similarity measure for VALIDITY: kNN distance to ref in standardized space
sc = StandardScaler().fit(X_ref)
nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(X_ref))
knn_d, _ = nn.kneighbors(sc.transform(X_tst))
cv['knn_ref_dist'] = knn_d.mean(1)

# ---------- frozen levels: quartiles of novelty ----------
qs = np.quantile(cv['novelty'], [0.25,0.5,0.75])
def to_level(v): return int(np.searchsorted(qs, v))  # 0..3
cv['level'] = cv['novelty'].apply(to_level)
R['novelty_quartile_thresholds'] = [float(x) for x in qs]

def acc_by_level(d, col='correct'):
    return {int(l): {'n': int((d.level==l).sum()),
                     'acc': float(d.loc[d.level==l, col].mean())} for l in range(4)}

# ---------- VALIDITY: similarity must decrease across levels ----------
val = {}
for l in range(4):
    sub = cv[cv.level==l]
    val[l] = {'mean_novelty': float(sub.novelty.mean()),
              'mean_knn_ref_dist': float(sub.knn_ref_dist.mean()),
              'n': int(len(sub))}
R['validity_levels'] = val
sp_knn = spearmanr(cv.novelty, cv.knn_ref_dist)
R['validity_spearman_novelty_vs_knn'] = {'rho': float(sp_knn.correlation), 'p': float(sp_knn.pvalue)}
mono = all(val[l]['mean_knn_ref_dist'] < val[l+1]['mean_knn_ref_dist'] for l in range(3))
R['validity_monotone_decreasing_similarity'] = bool(mono)

# ---------- CONCH SPC ----------
R['conch_overall_acc'] = float(cv.correct.mean())
R['conch_acc_by_level'] = acc_by_level(cv)
sp = spearmanr(cv.novelty, cv.correct)
R['conch_spearman_novelty_vs_correct'] = {'rho': float(sp.correlation), 'p': float(sp.pvalue)}
R['conch_acc_drop_L1_L4'] = float(R['conch_acc_by_level'][0]['acc'] - R['conch_acc_by_level'][3]['acc'])

# ---------- model-independent baseline (reference-trained) ----------
ytr = nct['cls'].values
base = LogisticRegression(max_iter=2000, C=1.0, multi_class='multinomial')
base.fit(sc.transform(X_ref), ytr)
cv['base_pred'] = base.predict(sc.transform(X_tst))
cv['base_correct'] = (cv['base_pred'] == cv['label_name']).astype(int)
R['baseline_overall_acc'] = float(cv.base_correct.mean())
R['baseline_acc_by_level'] = acc_by_level(cv, 'base_correct')
spb = spearmanr(cv.novelty, cv.base_correct)
R['baseline_spearman_novelty_vs_correct'] = {'rho': float(spb.correlation), 'p': float(spb.pvalue)}
R['baseline_acc_drop_L1_L4'] = float(R['baseline_acc_by_level'][0]['acc'] - R['baseline_acc_by_level'][3]['acc'])

# ---------- CONFOUND 1: BLUR ----------
# composite blur/sharpness: higher = sharper. Use lap_var (primary). Also report tenengrad, hf_ratio.
cv['sharp'] = cv['lap_var']
# blur tertiles
bq = np.quantile(cv['sharp'], [1/3, 2/3])
cv['blur_tert'] = cv['sharp'].apply(lambda v: int(np.searchsorted(bq, v)))
def partial_spearman(x, y, z):
    rx = rankdata(x); ry = rankdata(y); rz = rankdata(z)
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        coef,_,_,_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1@coef
    ex = resid(rx, rz); ey = resid(ry, rz)
    from scipy.stats import pearsonr
    r,p = pearsonr(ex, ey)
    return float(r), float(p)
# novelty->correct controlling for sharpness
r_ns, p_ns = partial_spearman(cv.novelty.values, cv.correct.values, cv.sharp.values)
# sharpness->correct controlling for novelty
r_sn, p_sn = partial_spearman(cv.sharp.values, cv.correct.values, cv.novelty.values)
R['confound_blur'] = {
  'spearman_novelty_correct_raw': float(spearmanr(cv.novelty,cv.correct).correlation),
  'partial_spearman_novelty_correct_given_sharp': {'r': r_ns, 'p': p_ns},
  'spearman_sharp_correct_raw': float(spearmanr(cv.sharp,cv.correct).correlation),
  'partial_spearman_sharp_correct_given_novelty': {'r': r_sn, 'p': p_sn},
  'corr_novelty_sharp': float(spearmanr(cv.novelty,cv.sharp).correlation),
  # novelty SPC within each blur tertile (must persist if real)
  'novelty_acc_by_level_within_blur_tertile': {
     int(t): acc_by_level(cv[cv.blur_tert==t]) for t in range(3)},
  # blur SPC within each novelty level (is blur a separate corruption axis?)
  'blur_acc_by_tertile_within_novelty_level': {
     int(l): {int(t): {'n': int(((cv.level==l)&(cv.blur_tert==t)).sum()),
                       'acc': float(cv[(cv.level==l)&(cv.blur_tert==t)].correct.mean())}
              for t in range(3)} for l in range(4)},
}

# ---------- CONFOUND 2: CLASS ----------
# within-class novelty->correct spearman + class-balanced level accuracy + logistic w/ class control
within = {}
for c in CLASSES:
    s = cv[cv.label_name==c]
    if s.correct.nunique() > 1 and len(s) > 30:
        rr = spearmanr(s.novelty, s.correct)
        within[c] = {'n': int(len(s)), 'rho': float(rr.correlation), 'p': float(rr.pvalue),
                     'acc_L_lowhalf': float(s[s.novelty<=s.novelty.median()].correct.mean()),
                     'acc_L_highhalf': float(s[s.novelty>s.novelty.median()].correct.mean())}
    else:
        within[c] = {'n': int(len(s)), 'rho': None, 'note': 'too few/constant'}
R['confound_class_within'] = within
# class-balanced level accuracy: weight each class equally within each level
def class_balanced_acc(d):
    out={}
    for l in range(4):
        sub=d[d.level==l]; accs=[]
        for c in CLASSES:
            cc=sub[sub.label_name==c]
            if len(cc)>=10: accs.append(cc.correct.mean())
        out[l]={'macro_acc': float(np.mean(accs)) if accs else None, 'n_classes': len(accs)}
    return out
R['conch_classbalanced_acc_by_level'] = class_balanced_acc(cv)
# logistic: correct ~ z(novelty) + class dummies
znov = (cv.novelty - cv.novelty.mean())/cv.novelty.std()
Xc = pd.get_dummies(cv['label_name'], drop_first=True).astype(float)
Xfull = np.c_[znov.values, Xc.values]
lr = LogisticRegression(max_iter=2000).fit(Xfull, cv.correct.values)
R['confound_class_logit'] = {'novelty_coef_logodds_per_SD': float(lr.coef_[0,0]),
                             'note':'correct ~ z(novelty) + C(class); coef<0 => more novel, less accurate, controlling class'}
# also novelty coef WITHOUT class control for comparison
lr0 = LogisticRegression(max_iter=2000).fit(znov.values.reshape(-1,1), cv.correct.values)
R['confound_class_logit']['novelty_coef_no_class_control'] = float(lr0.coef_[0,0])
# is novelty just bg_frac? partial controlling bg
r_nb, p_nb = partial_spearman(cv.novelty.values, cv.correct.values, cv.bg_frac.values)
R['confound_bg'] = {'partial_spearman_novelty_correct_given_bgfrac': {'r': r_nb, 'p': p_nb},
                    'corr_novelty_bgfrac': float(spearmanr(cv.novelty,cv.bg_frac).correlation)}

# ---------- CONFIRMATION: frozen 50/50 split (seed 7) ----------
rng = np.random.RandomState(7)
idx = rng.permutation(len(cv))
disc = cv.iloc[idx[:len(cv)//2]].copy()
conf = cv.iloc[idx[len(cv)//2:]].copy()
# freeze thresholds on discovery, apply to confirmation
dq = np.quantile(disc['novelty'], [0.25,0.5,0.75])
for d in (disc, conf):
    d['level_d'] = d['novelty'].apply(lambda v: int(np.searchsorted(dq, v)))
def acc_lv(d):
    return {int(l): {'n':int((d.level_d==l).sum()),'acc':float(d.loc[d.level_d==l,'correct'].mean())} for l in range(4)}
R['confirmation'] = {
  'discovery_acc_by_level': acc_lv(disc),
  'confirmation_acc_by_level': acc_lv(conf),
  'confirmation_spearman': {'rho': float(spearmanr(conf.novelty,conf.correct).correlation),
                            'p': float(spearmanr(conf.novelty,conf.correct).pvalue)},
  'confirmation_acc_drop_L1_L4': float(acc_lv(conf)[0]['acc']-acc_lv(conf)[3]['acc']),
  'frozen_thresholds_from_discovery': [float(x) for x in dq],
}

# save merged per-tile table + results
cv.to_csv(f"{W}/out/crcval_scored.csv", index=False)
json.dump(R, open(f"{W}/out/results.json","w"), indent=2)

# ---------- print summary ----------
def pl(d): return " ".join(f"L{l+1}={d[l]['acc']:.3f}(n{d[l]['n']})" for l in range(4))
print("\n===== VALIDITY (similarity must DECREASE L1->L4) =====")
for l in range(4):
    print(f"  L{l+1}: mean_novelty={val[l]['mean_novelty']:.2f}  mean_kNN_ref_dist={val[l]['mean_knn_ref_dist']:.3f}  n={val[l]['n']}")
print(f"  monotone decreasing similarity: {mono}")
print("\n===== CONCH SPC =====")
print("  acc by level:", pl(R['conch_acc_by_level']))
print(f"  overall={R['conch_overall_acc']:.3f}  drop L1-L4={R['conch_acc_drop_L1_L4']:.3f}  spearman rho={sp.correlation:.3f} (p={sp.pvalue:.1e})")
print("  class-balanced (macro) acc by level:", {l:round(R['conch_classbalanced_acc_by_level'][l]['macro_acc'],3) for l in range(4)})
print("\n===== BASELINE (reference-trained logistic on color feats) =====")
print("  acc by level:", pl(R['baseline_acc_by_level']))
print(f"  overall={R['baseline_overall_acc']:.3f}  drop L1-L4={R['baseline_acc_drop_L1_L4']:.3f}  spearman rho={spb.correlation:.3f}")
print("\n===== CONFOUND: BLUR =====")
print(f"  corr(novelty,sharp)={R['confound_blur']['corr_novelty_sharp']:.3f}")
print(f"  novelty->correct: raw rho={R['confound_blur']['spearman_novelty_correct_raw']:.3f} | partial|sharp r={r_ns:.3f} (p={p_ns:.1e})")
print(f"  sharp->correct:   raw rho={R['confound_blur']['spearman_sharp_correct_raw']:.3f} | partial|novelty r={r_sn:.3f} (p={p_sn:.1e})")
print("\n===== CONFOUND: CLASS =====")
print(f"  novelty logodds/SD: no-class={R['confound_class_logit']['novelty_coef_no_class_control']:.3f}  with-class={R['confound_class_logit']['novelty_coef_logodds_per_SD']:.3f}")
print("  within-class spearman(novelty,correct):")
for c in CLASSES:
    w=within[c]
    if w.get('rho') is not None:
        print(f"    {c}: n={w['n']:4d} rho={w['rho']:+.3f} p={w['p']:.2e}  acc lo/hi={w['acc_L_lowhalf']:.3f}/{w['acc_L_highhalf']:.3f}")
    else:
        print(f"    {c}: n={w['n']:4d} {w.get('note')}")
print("\n===== CONFIRMATION (held-out half, seed 7) =====")
print("  discovery   acc by level:", pl(R['confirmation']['discovery_acc_by_level']))
print("  confirmation acc by level:", pl(R['confirmation']['confirmation_acc_by_level']))
print(f"  confirmation drop L1-L4={R['confirmation']['confirmation_acc_drop_L1_L4']:.3f}  spearman rho={R['confirmation']['confirmation_spearman']['rho']:.3f} (p={R['confirmation']['confirmation_spearman']['p']:.1e})")
print("\nwrote out/results.json and out/crcval_scored.csv")
