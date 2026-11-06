# Original notebook cell 2
from pathlib import Path
import sys,json,hashlib,io,os,tempfile,shutil,runpy,contextlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from IPython.display import display,Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
root=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'STATE_INPUTS.json').is_file()),None)
assert root is not None,'Extract the complete STATE bundle first.'
manifest=json.loads((root/'STATE_INPUTS.json').read_text())
P=root/'assets'/'state-opus-r1'
for relative,meta in manifest.items():
    if relative.startswith('state-opus-r1'+'/'):
        assert hashlib.sha256((root/'assets'/relative).read_bytes()).hexdigest()==meta['sha256'],relative
print('Verified all run input and code hashes.')

# Original notebook cell 3
def draw(data,x,y,group):
    rng=np.random.default_rng(7);rows=[]
    for keys,g in data.groupby([group,x],observed=True):
        v=pd.to_numeric(g[y],errors='coerce').dropna().values
        if not len(v):continue
        bs=np.array([rng.choice(v,len(v),True).mean() for _ in range(1000)])
        rows.append({group:keys[0],x:keys[1],'n':len(v),'mean':v.mean(),'lo':np.quantile(bs,.025),'hi':np.quantile(bs,.975)})
    t=pd.DataFrame(rows);display(t)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');levels=sorted(t[x].unique(),key=str)
    for name,g in t.groupby(group,observed=True):
        g=g.assign(xpos=g[x].map({v:i for i,v in enumerate(levels)})).sort_values('xpos')
        ax.errorbar(g.xpos,g['mean'],yerr=np.stack([g['mean']-g.lo,g.hi-g['mean']]),marker='o',capsize=2,label=str(name))
    ax.set_xticks(range(len(levels)),levels,rotation=25);ax.set(xlabel=x,ylabel=y);ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120);plt.close(fig);display(Image(data=buf.getvalue()))

# Original notebook cell 4
raw=pd.read_csv(P/'data/predictions.csv',low_memory=False)
genes=set(raw.loc[raw.dataset_family=='Replogle-Nadig','perturbation'].str.upper())
tahoe=raw[raw.dataset_family=='Tahoe-100M'].copy()
print('Reference panel genes:',len(genes),'scored records:',len(raw))
print('This reproduces the original evaluation-panel proxy for genetic training experience, not a verified actual training list.')

# Original notebook cell 5
r=raw[(raw.dataset_family=='Replogle-Nadig')&(raw.protocol=='zeroshot')]
for rep,g in r.groupby('representation'):
    pivot=g.pivot_table(index='perturbation',columns='context',values='overlap_at_N')
    discovery=pivot[['k562','rpe1']].mean(axis=1);other=pivot[['jurkat','hepg2']].mean(axis=1)
    matched=pd.concat([discovery.rename('training_like'),other.rename('other_contexts')],axis=1).dropna()
    print(rep,len(matched),'paired difference=',float((matched.other_contexts-matched.training_like).mean()))
    display(matched.head())
