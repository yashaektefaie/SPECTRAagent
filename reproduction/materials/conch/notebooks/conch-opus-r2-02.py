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
dev=read_csv('conch-opus-r2/dev_features.csv'); display(dev[['sample_id','image_path','label_name','csv_correct']].head()); print('Development records:',len(dev))

# Original notebook cell 9
features=['sharpness_varlap','sharpness_tenengrad','highfreq_frac','brightness','contrast_std','saturation_mean','colorfulness','tissue_frac','hematoxylin_mean','eosin_mean','he_ratio','entropy']
rows=[]
for feature in features:
    x=dev[feature].to_numpy(); y=dev.csv_correct.to_numpy(); r,p=stats.pointbiserialr(y,x)
    weighted=[];weights=[]
    for _,g in dev.groupby('label_name'):
        if len(g)<30 or g.csv_correct.std()==0 or g[feature].std()==0:continue
        weighted.append(stats.pointbiserialr(g.csv_correct,g[feature]).statistic);weights.append(len(g))
    rows.append({'feature':feature,'marginal_r':r,'p':p,'within_class_r':np.average(weighted,weights=weights)})
display(pd.DataFrame(rows))
for feature in ['hematoxylin_mean','eosin_mean','saturation_mean']:
    x=dev[feature].to_numpy(); q=np.quantile(x,[0,.2,.4,.6,.8,1]); result=[]
    for j in range(5):
        mask=(x>=q[j])&((x<=q[j+1]) if j==4 else (x<q[j+1])); g=dev[mask]
        lo,hi=wilson(g.csv_correct.sum(),len(g));result.append(dict(level=j+1,model=feature,accuracy=g.csv_correct.mean(),lo=lo,hi=hi,n=len(g)))
    display(pd.DataFrame(result));show_curve(pd.DataFrame(result),xlabel='Development feature quintile')
print('No stain axis was frozen or independently confirmed. These are development screens only.')
