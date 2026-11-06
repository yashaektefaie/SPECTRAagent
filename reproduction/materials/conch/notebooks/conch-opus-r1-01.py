# Original notebook cell 1
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import display
ROOT = Path.cwd()
if not (ROOT / 'source').is_dir(): ROOT = ROOT.parent
SOURCE = ROOT / 'source'
assert SOURCE.is_dir(), 'Run from the conch package or its notebooks directory.'
def read_csv(path): return pd.read_csv(SOURCE / path)
def read_json(path): return json.loads((SOURCE / path).read_text())
def paired_interval(values, seed=0, draws=5000):
    values = np.asarray(values, float); rng=np.random.default_rng(seed)
    # Chunk draws to keep memory bounded, preserving NumPy RNG order.
    means=np.concatenate([values[rng.integers(0,len(values),(min(100,draws-i),len(values)))].mean(1) for i in range(0,draws,100)])
    return float(values.mean()), np.quantile(means,[.025,.975])
def wilson(k,n):
    z=1.959963984540054; p=k/n; a=1+z*z/n
    center=(p+z*z/(2*n))/a; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/a
    return center-h, center+h
def show_curve(table, x='level', group='model', y='accuracy', xlabel='Level', ylabel='Accuracy'):
    fig,ax=plt.subplots(figsize=(7,3.6))
    for name,g in table.groupby(group,sort=False):
        g=g.sort_values(x); ax.plot(g[x],g[y],marker='o',label=name)
        if 'lo' in g and 'hi' in g: ax.fill_between(g[x],g.lo,g.hi,alpha=.13)
    ax.set(xlabel=xlabel,ylabel=ylabel);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False);fig.tight_layout();plt.show()

# Original notebook cell 4
ref = read_csv('conch-opus-r1/nct_features.csv')
test = read_csv('conch-opus-r1/crcval_features.csv')
pred = read_csv('shared/crcval_predictions.csv')
assert ref.id.is_unique and test.id.is_unique and pred.sample_id.is_unique
# Pixel-feature IDs are original image filenames; the joins are never positional.
test = test.merge(pred, left_on='id',right_on='image_path',validate='one_to_one')
assert len(test)==7180 and len(ref)==2700
classes=['ADI','BACK','DEB','LYM','MUC','MUS','NORM','STR','TUM']
display(pd.DataFrame({'reference':ref.cls.value_counts(),'evaluation':test.label_name.value_counts()}).fillna(0).astype(int))
print('Exact sample IDs and image filenames are in the displayed table and packaged manifests.')
display(test[['sample_id','id','label_name']].head())

# Original notebook cell 7
import sys,zipfile,tarfile,io
from PIL import Image
sys.path.insert(0,str(SOURCE/'conch-opus-r1/code'))
from feats import compute_features,NOVELTY_FEATS,BLUR_FEATS
feature_names=NOVELTY_FEATS+BLUR_FEATS
# Recompute every original prospective pixel measurement from the source images.
with zipfile.ZipFile(SOURCE/'conch-opus-r1/CRC-VAL-HE-7K.zip') as archive:
    names={Path(n).name:n for n in archive.namelist() if n.lower().endswith('.tif')}
    computed=[]
    for name in test.id:
        im=np.asarray(Image.open(io.BytesIO(archive.read(names[name]))).convert('RGB'))
        computed.append(compute_features(im))
    pixel_test=pd.DataFrame(computed,index=test.index)
computed={}
with tarfile.open(SOURCE/'conch-opus-r1/nct-reference-images.tar.gz',mode='r|gz') as archive:
    for member in archive:
        if not member.isfile():continue
        name=Path(member.name).name
        im=np.asarray(Image.open(io.BytesIO(archive.extractfile(member).read())).convert('RGB'))
        computed[name]=compute_features(im)
    assert set(ref.id).issubset(computed)
    pixel_ref=pd.DataFrame([computed[name] for name in ref.id],index=ref.index)
np.testing.assert_allclose(pixel_test[feature_names],test[feature_names],rtol=2e-6,atol=1e-8)
np.testing.assert_allclose(pixel_ref[feature_names],ref[feature_names],rtol=2e-6,atol=1e-8)
print('Re-extracted and verified all',len(test)+len(ref),'original images,',len(feature_names),'features per image.')
# Downstream uses the recomputed values, not the archived feature columns.
test[feature_names]=pixel_test[feature_names];ref[feature_names]=pixel_ref[feature_names]

# Original notebook cell 8
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
features=['mean_R','mean_G','mean_B','std_R','std_G','std_B','mean_H','mean_E','std_H','std_E','mean_sat','std_sat','mean_val','std_val','colorfulness','bg_frac']
xref=ref[features].to_numpy(float); xtest=test[features].to_numpy(float)
mu=xref.mean(0); VI=np.linalg.pinv(LedoitWolf().fit(xref).covariance_)
d=xtest-mu; test['novelty']=np.sqrt(np.einsum('ij,jk,ik->i',d,VI,d))
scaler=StandardScaler().fit(xref)
nn=NearestNeighbors(n_neighbors=5).fit(scaler.transform(xref))
test['knn_ref_dist']=nn.kneighbors(scaler.transform(xtest))[0].mean(1)
test['sharp']=test.lap_var
# Original quartiles / within-class tertiles use np.searchsorted's left side.
test['global_level']=np.searchsorted(np.quantile(test.novelty,[.25,.5,.75]),test.novelty)+1
def within_levels(df,col,cut_source=None):
    cuts=cut_source if cut_source is not None else df
    levels=np.full(len(df),-1); cutpoints={}
    for label in classes:
        v=cuts.loc[cuts.label_name==label,col]
        if len(v)<6:continue
        q=np.quantile(v,[1/3,2/3]); cutpoints[label]=q.tolist()
        m=(df.label_name==label).to_numpy(); levels[m]=np.searchsorted(q,df.loc[m,col])+1
    return levels,cutpoints
test['stain_level'],stain_cutpoints=within_levels(test,'novelty')
test['sharp_level'],sharp_cutpoints=within_levels(test,'sharp')
display(pd.DataFrame(stain_cutpoints,index=['tertile1','tertile2']).T)
# Match reconstructed axis values to the original scored individual table.
archived=read_csv('conch-opus-r1/crcval_scored.csv').set_index('id')
np.testing.assert_allclose(test.novelty,archived.loc[test.id,'novelty'],rtol=2e-6,atol=1e-8)
print('Recomputed Mahalanobis distance matches the archived per-image values.')

# Original notebook cell 11
logit_cols=['logit_'+c for c in classes]
computed_pred=test[logit_cols].to_numpy().argmax(1)
assert np.array_equal(computed_pred,test.pred)
test['conch_correct']=(computed_pred==test.label).astype(int)
assert np.array_equal(test.conch_correct,test.correct)
# Refit the original fixed reference classifier using only the 2700 reference tiles.
reference_model=LogisticRegression(max_iter=2000,C=1.0).fit(scaler.transform(xref),ref.cls)
test['base_pred_replay']=reference_model.predict(scaler.transform(xtest))
test['baseline_correct']=(test.base_pred_replay==test.label_name).astype(int)
base_matches=(test.base_pred_replay.to_numpy()==archived.loc[test.id,'base_pred'].to_numpy()).mean()
print('CONCH labels recovered from stored logits:',len(test),'/',len(test))
print('Reference prediction agreement with original:',base_matches)
assert base_matches > .999, 'Reference refit differs: inspect environment before using results.'
display(test[['sample_id','label_name','pred_label_name','conch_correct','base_pred_replay','baseline_correct']].head())

# Original notebook cell 13
def curve(df,levcol,balanced=True):
    rows=[]
    for level,group in df.groupby(levcol):
        for model,col in [('CONCH','conch_correct'),('Reference','baseline_correct')]:
            strata=[g[col].to_numpy(float) for _,g in group.groupby('label_name') if len(g)>=8] if balanced else [group[col].to_numpy(float)]
            mean=float(np.mean([v.mean() for v in strata]))
            rng=np.random.default_rng(20261106)
            draws=np.zeros(2000)
            for v in strata: draws+=v[rng.integers(0,len(v),(2000,len(v)))].mean(1)/len(strata)
            lo,hi=np.quantile(draws,[.025,.975])
            rows.append(dict(level=int(level),model=model,accuracy=mean,lo=lo,hi=hi,n=len(group),classes=len(strata)))
    return pd.DataFrame(rows)

result=curve(test,'stain_level',balanced=True)
refined=read_json('conch-opus-r1/results_refined.json'); expected=[refined['within_class_novelty_SPC_conch'][str(j)]['macro_acc'] for j in range(3)]
np.testing.assert_allclose(result[result.model=='CONCH'].accuracy,expected,atol=1e-12)
display(result)
show_curve(result,ylabel='Class-balanced accuracy' if True else 'Overall accuracy')
print('Curves exactly match original stored results. Shading is a new class-stratified tile-bootstrap interval (2000 draws, seed 20261106), not an original reported interval.')

# Original notebook cell 15
import statsmodels.api as sm
# Reproduce the original 07_final.py reference-only blur residualization.
assert list(test.id)==list(archived.index), 'The original permutation requires the exact original row order.'
Bref=sm.add_constant(ref[BLUR_FEATS].to_numpy());Btst=sm.add_constant(test[BLUR_FEATS].to_numpy())
reference_residuals=np.zeros((len(ref),len(NOVELTY_FEATS)));test_residuals=np.zeros((len(test),len(NOVELTY_FEATS)))
for j,feature in enumerate(NOVELTY_FEATS):
    coef=np.linalg.lstsq(Bref,ref[feature].to_numpy(),rcond=None)[0]
    reference_residuals[:,j]=ref[feature].to_numpy()-Bref@coef
    test_residuals[:,j]=test[feature].to_numpy()-Btst@coef
center=reference_residuals.mean(0);inverse=np.linalg.pinv(LedoitWolf().fit(reference_residuals).covariance_)
residual=test_residuals-center
test['novelty_bf']=np.sqrt(np.einsum('ij,jk,ik->i',residual,inverse,residual))
test['blurfree_level'],_=within_levels(test,'novelty_bf')
final=read_json('conch-opus-r1/results_final.json')
controlled=curve(test,'blurfree_level',balanced=True)
for model,key in [('CONCH','blurfree_novelty_SPC_conch'),('Reference','blurfree_novelty_SPC_baseline')]:
    expected=[final[key][str(j)] for j in range(3)]
    np.testing.assert_allclose(controlled.loc[controlled.model==model,'accuracy'],expected,atol=1e-12)
display(controlled)
print('Original blur-residualized endpoint declines:')
for model,g in controlled.groupby('model',sort=False):print(model,g.accuracy.iloc[0]-g.accuracy.iloc[-1])
# Original half split and thresholds use the residualized property, not unadjusted novelty.
idx=np.random.RandomState(7).permutation(len(test));disc=test.iloc[idx[:len(test)//2]].copy();held=test.iloc[idx[len(test)//2:]].copy()
assert not set(disc.sample_id)&set(held.sample_id)
for name,property_name,key in [('Blur-residualized novelty','novelty_bf','confirm_blurfree_novelty_SPC'),('Sharpness','sharp','confirm_sharpness_SPC')]:
    held['replay_level'],cutpoints=within_levels(held,property_name,cut_source=disc)
    confirmation=curve(held,'replay_level',balanced=True)
    np.testing.assert_allclose(confirmation.loc[confirmation.model=='CONCH','accuracy'],[final[key][str(j)] for j in range(3)],atol=1e-12)
    print(name,'original within-cohort half-split check:');display(confirmation)
# Recompute original head-to-head class/blur controls and held-out joint regression.
def percentile_rank(v):return (stats.rankdata(v)-1)/(len(v)-1)
def original_fit(frame,terms):
    X=pd.DataFrame(index=frame.index)
    for term in terms:
        if term=='nov':X['nov']=percentile_rank(frame.novelty_bf)
        elif term=='sharp':X['sharp']=percentile_rank(frame.sharp)
        elif term=='class':
            for label in classes[1:]:X['c_'+label]=(frame.label_name==label).astype(float)
    X=sm.add_constant(X)
    return sm.Logit(frame.conch_correct.to_numpy(),X.to_numpy()).fit(disp=0,maxiter=200),list(X.columns)
prs={name:float(original_fit(test,terms)[0].prsquared) for name,terms in [('class',['class']),('class+nov',['class','nov']),('class+sharp',['class','sharp']),('class+nov+sharp',['class','nov','sharp'])]}
for name,v in prs.items():np.testing.assert_allclose(v,final['pseudoR2'][name],atol=1e-8)
display(pd.Series(prs,name='McFadden pseudo-R2').to_frame())
joint,columns=original_fit(held,['nov','sharp','class']);joint_rows=[]
for name in ['nov','sharp']:
    j=columns.index(name);odds=float(np.exp(joint.params[j]));p=float(joint.pvalues[j])
    np.testing.assert_allclose(odds,final['confirm_joint_logit'][name]['OR'],rtol=1e-5)
    np.testing.assert_allclose(p,final['confirm_joint_logit'][name]['p'],rtol=1e-5,atol=1e-14)
    joint_rows.append({'term':name,'odds_ratio':odds,'p':p})
display(pd.DataFrame(joint_rows))
print('Original full-cohort residualized curves, both half-split checks and joint controls reproduced. Diagnostic bootstrap bands are new, labeled recalculations.')
print('Full-cohort exploration preceded this within-cohort split. Reproducing it does not establish independently selected external confirmation.')
