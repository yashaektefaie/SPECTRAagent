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
targets=tahoe.groupby('perturbation_norm').drugmeta_targets.first().dropna().apply(lambda s:{v.strip().upper() for v in s.split(',') if v.strip()})
coverage=targets.apply(lambda s:len(s&genes)/len(s))
expected=pd.read_csv(P/'PRIMARY_axis_drug_target_coverage.csv').set_index('perturbation_norm')
np.testing.assert_allclose(coverage.loc[expected.index],expected.frac,atol=1e-12)
data=tahoe[(tahoe.protocol=='zeroshot')&(tahoe.representation=='HVG')].groupby('perturbation_norm').overlap_at_N.mean().to_frame().join(coverage.rename('coverage')).loc[expected.index]
np.testing.assert_allclose(data.overlap_at_N,expected.overlap,atol=1e-12)
data['level']=np.where(data.coverage==0,'none_covered',np.where(data.coverage==1,'all_covered','partial'));data['panel']='zeroshot/HVG'
draw(data,'level','overlap_at_N','panel')
print('Discovery correlation:',spearmanr(data.coverage,data.overlap_at_N))
for keys,g in tahoe.groupby(['protocol','representation']):
    z=g.groupby('perturbation_norm').overlap_at_N.mean().to_frame().join(coverage.rename('coverage')).dropna()
    print('Confirmation slice',keys,len(z),spearmanr(z.coverage,z.overlap_at_N))
