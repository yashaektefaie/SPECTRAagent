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
pi=pd.read_csv(K/'artifacts/axisA_confirm/per_image_eval.csv')
crc=pd.read_csv(K/'artifacts/axisA_crcval/crcval_per_image.csv')
manifest=pd.read_csv(K/'data/NCT-sample/nct_sample_manifest.csv')
evaluation=manifest[manifest.split=='eval']
assert pi.path.is_unique and crc.path.is_unique
assert len(pi)==3600 and len(crc)==7180
assert set(pi.path)==set(evaluation.path)
assert not set(pi.path)&set(crc.path)
cohorts={'NCT confirmation':pi,'CRC-VAL':crc}
display(pd.DataFrame({name:frame.groupby('label').size() for name,frame in cohorts.items()}))
print('Exact original filenames identify each sample. NCT has 400 tiles/class; CRC-VAL has unequal class counts.')

# Original notebook cell 5
assignment=pd.concat([frame[['path','label']].assign(cohort=name,level_s=1) for name,frame in cohorts.items()],ignore_index=True)
assert assignment.level_s.eq(1).all()
display(assignment.head())
print('Both cohorts use native-resolution predictions (s=1). Cohort membership is categorical; no ordered site-distance or independent site-specific freeze is established.')

# Original notebook cell 7
probs=np.load(K/'artifacts/axisA_confirm/probs_CONCH_s1.npy')
assert probs.shape==(len(pi),9)
assert np.array_equal(probs.argmax(1),pi.pred_CONCH_s1)
for name,frame in cohorts.items():
    frame['correct_native']=(frame.pred_CONCH_s1==frame.label).astype(int)
    assert set(frame.label)==set(range(9))
print('NCT labels are reproduced from original class-score arrays; CRC-VAL uses retained predicted labels because its native score arrays were not recovered. Correctness for both is rescored against original labels. No fresh inference.')

# Original notebook cell 9
site=[];draws=2000
for name,frame in cohorts.items():
    strata=[g.correct_native.to_numpy(float) for _,g in frame.groupby('label')]
    balanced=float(np.mean([v.mean() for v in strata]))
    rng=np.random.default_rng(20261106);replicates=np.zeros(draws)
    for v in strata:
        replicates+=v[rng.integers(0,len(v),(draws,len(v)))].mean(1)/len(strata)
    lo,hi=np.quantile(replicates,[.025,.975])
    site.append(dict(cohort=name,n=len(frame),balanced_accuracy=balanced,pooled_accuracy=frame.correct_native.mean(),lo=lo,hi=hi))
site=pd.DataFrame(site)
archived_nct=pd.read_csv(K/'artifacts/axisA_confirm/axisA_summary.csv')
archived_crc=pd.read_csv(K/'artifacts/axisA_crcval/crcval_summary.csv')
for index,archived in [(0,archived_nct),(1,archived_crc)]:
    original=archived[(archived.model=='CONCH')&(archived.level_s==1)].iloc[0]
    np.testing.assert_allclose(site.iloc[index].balanced_accuracy,original.balanced_acc,atol=1e-12)
display(site)
fig,ax=plt.subplots(figsize=(6,3.2))
ax.errorbar(site.cohort,site.balanced_accuracy,yerr=[site.balanced_accuracy-site.lo,site.hi-site.balanced_accuracy],fmt='o',capsize=4)
ax.set(ylabel='Balanced accuracy (macro recall)',xlabel='Original cohort, native resolution')
ax.spines[['top','right']].set_visible(False);fig.tight_layout();plt.show()
print('Balanced accuracy matches the historical 0.771 versus0.756 comparison. Pooled accuracy is separately labeled; it is not the historical metric. Intervals are new within-class tile-bootstrap diagnostics (2,000 draws, seed 20261106), not original audit intervals.')

# Original notebook cell 11
display(pd.DataFrame({name:frame.groupby('label').correct_native.mean() for name,frame in cohorts.items()}))
print('The cohorts have different class frequencies; macro recall gives each class equal weight. This does not control other collection differences or isolate a causal site effect.')
print('No frozen, independently confirmed site-specific axis or matched cross-cohort baseline comparison is established here. The native-cohort observation remains exploratory.')
