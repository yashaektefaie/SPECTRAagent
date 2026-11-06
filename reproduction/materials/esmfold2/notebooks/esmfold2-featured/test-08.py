# Original notebook cell 2
from pathlib import Path
import json,hashlib,importlib.util,sys,io
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from IPython.display import display,Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
root=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'ESM_INPUTS.json').is_file()),None)
assert root is not None,'Extract the full ESMFold2 bundle and open this notebook inside it.'
manifest=json.loads((root/'ESM_INPUTS.json').read_text());P=root/'source/esmfold2';A=P/'reconstruction/prior_axes'
def checked(p):
    p=Path(p);k=str(p.relative_to(root));assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[k]['sha256'],k
    return p
def read(p):return pd.read_csv(checked(p),low_memory=False)
def module(p):
    p=checked(p);sys.path.insert(0,str(p.parent));spec=importlib.util.spec_from_file_location('replay_'+p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def equal_curve(got,path,keys,columns):
    old=read(path)
    if 'selection' in old:old=old[old.selection=='best_by_ptm']
    a=got.set_index(keys).sort_index();b=old.set_index(keys).sort_index()
    assert a.index.tolist()==b.index.tolist(),'Group identities changed'
    for c in columns:np.testing.assert_allclose(a[c],b[c],rtol=1e-9,atol=1e-10,equal_nan=True,err_msg=c)
    print('Matched original group identities and statistics:',path.name)
def plot_rows(df,level,metric='ca_lddt',label='Retained evaluation'):
    # Replay diagnostic intervals: 2,000 resamples of target records, seed 7.
    # These are not new inference or an additional independent cohort.
    rng=np.random.default_rng(7);rows=[]
    for lev,g in df.groupby(level,observed=True):
        v=g[metric].dropna().to_numpy(float)
        if not len(v):continue
        b=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(2000)])
        rows.append({'level':lev,'n':len(v),'mean':v.mean(),'lo':np.quantile(b,.025),'hi':np.quantile(b,.975)})
    t=pd.DataFrame(rows);display(t)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ax.errorbar(range(len(t)),t['mean'],yerr=np.vstack([t['mean']-t.lo,t.hi-t['mean']]),marker='o',capsize=3)
    ax.set_xticks(range(len(t)),t.level.astype(str));ax.set(xlabel=level,ylabel=metric,title=label);ax.spines[['top','right']].set_visible(False)
    b=io.BytesIO();fig.savefig(b,format='png',dpi=120);plt.close(fig);display(Image(data=b.getvalue()));return t

# Original notebook cell 3
g=module(P/'scripts/global_hypothesis_audit.py')
raw=read(P/'data/primary_exact_sequence_dedup.csv')
assert raw.sequence_hash_calc.is_unique
d=g.add_features(raw)
np.testing.assert_allclose(d.disulfide_capacity_per100_calc,raw.disulfide_capacity_per100_calc,atol=1e-12)
print('Exact deduplicated sequences:',len(d));display(d[['target_id','sequence_clean','data_family','setting_label','ca_lddt']].head())
effects,curves,slices=g.evaluate_hypotheses(d,'primary_sequence')

# Original notebook cell 4
names=['high_disulfide_capacity']
got=effects[effects.axis.isin(names)];old=read(P/'tables/global_hypothesis_effects_primary_sequence.csv');old=old[old.axis.isin(names)]
a=got.set_index('axis').sort_index();b=old.set_index('axis').sort_index()
assert a.index.equals(b.index)
for c in ['top_ca_lddt_mean','lower_ca_lddt_mean']:
    np.testing.assert_allclose(a[c],b[c],atol=1e-12)
display(got);display(slices[slices.axis.isin(names)])
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for _,row in got.iterrows():ax.plot(['Complement','Matching group'],[row.lower_ca_lddt_mean,row.top_ca_lddt_mean],marker='o',label=row.axis)
ax.set_ylabel('Mean Cα-lDDT');ax.legend(fontsize=7);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120);plt.close(fig);display(Image(data=buf.getvalue()))
