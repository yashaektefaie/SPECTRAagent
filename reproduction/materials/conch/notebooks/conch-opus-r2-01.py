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

# Original notebook cell 3
rows=read_csv('conch-opus-r2/confirmation_per_tile.csv')
manifest=read_csv('conch-opus-r2/confirmation_manifest.csv')
reference=read_csv('conch-opus-r2/reference_manifest.csv')
plan=read_json('conch-opus-r2/analysis_plan.json')
assert manifest.sample_id.is_unique and len(manifest)==1080
assert set(rows.sample_id)==set(manifest.sample_id)
assert not set(manifest.sample_id)&set(reference.sample_id)
print('Confirmation tiles:',len(manifest),'paired outcome rows:',len(rows))
display(manifest.groupby('label_name').size().rename('n').to_frame())
display(manifest.head())

# Original notebook cell 6
import cv2
from PIL import Image
sigmas=[float(level['sigma']) for level in plan['ordered_levels']]
assert sigmas==sorted(rows.sigma.unique())
# Exact intervention used in the frozen run: same tile, blur level changes.
def blur_rgb(image,sigma):
    if sigma<=0:return image
    k=max(3,int(2*round(3*sigma)+1))
    return cv2.GaussianBlur(image,(k,k),sigma)
by_sample={row['sample_id']:row for row in read_json('conch-opus-r2/image_hashes.json')}
recomputed=[]
for item in manifest.to_dict('records'):
    path=SOURCE/'conch-opus-r2/images'/item['image_abspath']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==by_sample[item['sample_id']]['sha256']
    image=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    for sigma in sigmas:
        changed=blur_rgb(image,sigma)
        gray=cv2.cvtColor(changed,cv2.COLOR_RGB2GRAY).astype(np.float64)
        hsv=cv2.cvtColor(changed,cv2.COLOR_RGB2HSV)
        recomputed.append(dict(sample_id=item['sample_id'],sigma=sigma,sharpness=cv2.Laplacian(gray,cv2.CV_64F).var(),brightness=gray.mean(),saturation_mean=hsv[:,:,1].mean()))
axis=pd.DataFrame(recomputed).merge(rows,on=['sample_id','sigma'],validate='one_to_one',suffixes=('_replay',''))
np.testing.assert_allclose(axis.sharpness,axis.varlap,rtol=1e-10,atol=1e-8)
np.testing.assert_allclose(axis.brightness_replay,axis.brightness,rtol=1e-10,atol=1e-8)
np.testing.assert_allclose(axis.saturation_mean_replay,axis.saturation_mean,rtol=1e-10,atol=1e-8)
print('All 4,320 image interventions and pixel-feature measurements independently replayed and matched.')
display(axis.groupby('sigma')[['sharpness','brightness_replay','saturation_mean_replay']].mean())

# Original notebook cell 9
for model in ['conch','baseline']:
    score=(rows[model+'_pred']==rows.label).astype(int)
    assert np.array_equal(score,rows[model+'_correct'])
    rows[model+'_correct']=score
assert (rows.groupby('sample_id').size()==len(sigmas)).all()
assert rows.groupby('sample_id').label.nunique().eq(1).all()
print('All saved target and baseline predictions rescored against labels; no new model inference is represented here.')
display(rows[['sample_id','sigma','label','conch_pred','conch_correct','baseline_pred','baseline_correct']].head())

# Original notebook cell 11
result=[]
for sigma,group in rows.sort_values('sample_id').groupby('sigma'):
    for model in ['conch','baseline']:
        mean,ci=paired_interval(group[model+'_correct'],seed=1,draws=4000)
        result.append(dict(level=sigma,model=model,accuracy=mean,lo=ci[0],hi=ci[1],n=len(group)))
result=pd.DataFrame(result)
expected=read_csv('conch-opus-r2/spc_table.csv').set_index('sigma')
for item in result.to_dict('records'):
    model=item['model']; exp=expected.loc[item['level']]
    np.testing.assert_allclose([item['accuracy'],item['lo'],item['hi']],[exp[model+'_acc'],exp[model+'_ci_lo'],exp[model+'_ci_hi']],rtol=0,atol=1e-12)
display(result)
show_curve(result,xlabel='Gaussian blur sigma (pixels)',ylabel='Top-1 accuracy')

# Original notebook cell 13
p=rows.pivot(index='sample_id',columns='sigma',values=['conch_correct','baseline_correct']).sort_index()
conch_drop=(p['conch_correct'][min(sigmas)]-p['conch_correct'][max(sigmas)]).to_numpy()
base_drop=(p['baseline_correct'][min(sigmas)]-p['baseline_correct'][max(sigmas)]).to_numpy()
mean,ci=paired_interval(conch_drop)
did,didci=paired_interval(conch_drop-base_drop)
original=read_json('conch-opus-r2/analysis_results.json')
np.testing.assert_allclose([mean,*ci],[original['test_A_real_degradation']['conch_drop'],*original['test_A_real_degradation']['ci95']],atol=1e-12)
np.testing.assert_allclose([did,*didci],[original['test_B_did']['did_conch_minus_baseline'],*original['test_B_did']['ci95']],atol=1e-12)
print('CONCH decline:',mean,'95% paired bootstrap CI:',ci)
print('Reference decline:',base_drop.mean())
print('Excess CONCH decline:',did,'95% paired bootstrap CI:',didci)
print('Class balance and tile identity are fixed across levels. Independent image-quality measurements are above.')
print('Scientific conclusion: the blur intervention reduces accuracy more for CONCH than the fixed reference.')
print('Scope: same dataset family. Development versus confirmation uses the dataset-provided patient separation; reference versus confirmation patient separation cannot be verified from tile IDs. Cross-site independence remains unresolved. Blur is a known image-quality intervention, not a newly discovered natural shift.')
