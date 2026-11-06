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
K=SOURCE/'conch-kimi-r2/native'
manifest=pd.read_csv(K/'data/NCT-sample/nct_sample_manifest.csv')
pi=pd.read_csv(K/'artifacts/axisA_confirm/per_image_eval.csv')
train=manifest[manifest.split=='train_probe'].copy(); evaluation=manifest[manifest.split=='eval'].copy()
assert train.path.is_unique and evaluation.path.is_unique
assert not set(train.path)&set(evaluation.path)
assert set(pi.path)==set(evaluation.path) and len(pi)==3600
assert not set(train.md5)&set(evaluation.md5)
display(manifest.groupby(['split','class']).size().unstack(0))
display(evaluation[['zip_entry','class','label','split','md5']].head())

# Original notebook cell 5
from PIL import Image
# Original zeroshot_common.py intervention: bicubic downsample then upsample.
def degrade_resolution(img,s):
    if s==1:return img
    w,h=img.size
    return img.resize((max(1,w//s),max(1,h//s)),Image.Resampling.BICUBIC).resize((w,h),Image.Resampling.BICUBIC)
levels=[1,2,3,4,6,8]
assigned=pi[['path','class','label']].merge(pd.DataFrame({'level_s':levels}),how='cross')
assigned['effective_um_per_pixel']=.5*assigned.level_s
assert len(assigned)==21600
print('Each original evaluation tile receives every frozen resolution intervention:')
display(assigned.head(8))
# The fixed physical sampling unit is 0.5 um/pixel before digital downsampling.
# Axis-order source outputs are separately shown; no pixel/embedding re-evaluation is claimed.
order=pd.read_csv(K/'artifacts/axisA_confirm/axis_order_verification.csv')
import tarfile,io
from skimage.metrics import structural_similarity
images_by_name={}; expected_md5=dict(zip(evaluation.path,evaluation.md5)); seen=set(); sample_names=set(pi.path.iloc[:300])
with tarfile.open(K/'confirmation_images.tar.gz',mode='r|gz') as archive:
    for member in archive:
        if not member.isfile():continue
        name=Path(member.name).name
        if name not in expected_md5:continue
        raw=archive.extractfile(member).read()
        assert hashlib.md5(raw).hexdigest()==expected_md5[name]
        seen.add(name)
        if name in sample_names:images_by_name[name]=Image.open(io.BytesIO(raw)).convert('RGB')
    assert seen==set(expected_md5)
    images=[images_by_name[name] for name in pi.path.iloc[:300]]
    checks=[]
    original=np.load(K/'artifacts/axisA_confirm/dino_eval_s1.npy')
    original=original/np.linalg.norm(original,axis=1,keepdims=True)
    for factor in levels:
        changed=np.load(K/f'artifacts/axisA_confirm/dino_eval_s{factor}.npy')
        changed=changed/np.linalg.norm(changed,axis=1,keepdims=True)
        cosine=float((original*changed).sum(1).mean())
        ssim=float(np.mean([structural_similarity(np.asarray(im),np.asarray(degrade_resolution(im,factor)),channel_axis=2,data_range=255) for im in images]))
        expected=order[order.level_s==factor].iloc[0]
        np.testing.assert_allclose([cosine,ssim],[expected.dino_cosine_to_native,expected.ssim_to_native],rtol=1e-6,atol=1e-7)
        checks.append(dict(factor=factor,ssim_replayed=ssim,cosine_from_saved_embeddings=cosine))
display(pd.DataFrame(checks))
print('All 3600 original image MD5 hashes verified; original SSIM image-transform check replayed on the same 300 tiles; all embedding-cosine values recalculated from retained DINOv2 vectors.')

# Original notebook cell 7
models=['CONCH','R1-OpenCLIP-B32-LAION','R2-DINOv2-logreg']
y=pi.label.to_numpy()
for model in models:
    for s in levels:
        if model!='R2-DINOv2-logreg':
            probs=np.load(K/f"artifacts/axisA_confirm/probs_{model.replace('-', '_')}_s{s}.npy")
            assert probs.shape==(len(pi),9)
            assert np.array_equal(probs.argmax(1),pi[f'pred_{model}_s{s}'])
        pi[f'correct_{model}_s{s}']=(pi[f'pred_{model}_s{s}'].to_numpy()==y).astype(int)
print('All CONCH/OpenCLIP predicted labels reproduced from stored class-score arrays.')
print('All models rescored against the exact evaluation labels; DINOv2 uses retained predicted labels.')
display(pi[['path','label','pred_CONCH_s1','pred_CONCH_s8','correct_CONCH_s1','correct_CONCH_s8']].head())

# Original notebook cell 9
summary=pd.read_csv(K/'artifacts/axisA_confirm/axisA_summary.csv')
result=[]
for model in models:
    for s in levels:
        ok=pi[f'correct_{model}_s{s}'].to_numpy()
        accuracy=float(ok.mean()); balanced=float(np.mean([ok[y==c].mean() for c in np.unique(y)]))
        archived=summary[(summary.model==model)&(summary.level_s==s)].iloc[0]
        np.testing.assert_allclose([accuracy,balanced],[archived.top1_acc,archived.balanced_acc],atol=1e-12)
        # Wilson intervals newly added for display; original run used paired McNemar tests below.
        lo,hi=wilson(ok.sum(),len(ok))
        result.append(dict(level=.5*s,model=model,accuracy=balanced,lo=lo,hi=hi,n=len(ok)))
result=pd.DataFrame(result);display(result)
show_curve(result,xlabel='Effective resolution (micrometers per pixel)',ylabel='Accuracy on class-balanced tiles')
print('Shading: Wilson 95% intervals recalculated from individual outcomes for this notebook, not original audit intervals.')

# Original notebook cell 11
recorded=pd.read_csv(K/'artifacts/axisA_confirm/confirmatory_stats.csv')
comparisons=[]
for model in models:
    native=pi[f'correct_{model}_s1'].to_numpy().astype(bool)
    for s in levels[1:]:
        changed=pi[f'correct_{model}_s{s}'].to_numpy().astype(bool)
        b=int((native&~changed).sum()); c=int((~native&changed).sum())
        pvalue=stats.binomtest(min(b,c),b+c,.5).pvalue if b+c else 1.
        old=recorded[(recorded.model==model)&(recorded['compare']==f's1_vs_s{s}')].iloc[0]
        assert b==old.n_discordant_s1_better and c==old.n_discordant_sS_better
        np.testing.assert_allclose(pvalue,old.mcnemar_p,atol=1e-14)
        comparisons.append(dict(model=model,contrast=f'1 vs {s}',better_native=b,better_changed=c,p=pvalue))
display(pd.DataFrame(comparisons))
print('Native-to-coarsest declines, recomputed on identical images:')
for model in models: print(model,pi[f'correct_{model}_s1'].mean()-pi[f'correct_{model}_s8'].mean())
print('All paired McNemar counts and p values match the original confirmation statistics.')
