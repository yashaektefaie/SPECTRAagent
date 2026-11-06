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
D=A/'04_reference_geometry_and_new_axes';m=module(D/'scripts/audit_frozen_reference_topology_axes.py');builder=module(D/'scripts/build_combined_and_score_axes.py')
d=read(D/'spectra_ready_dataset/combined_feature_outcome_table.csv')
np.testing.assert_array_equal(d.target_id.map(builder.stable_fold),d.hash_fold10)
d['target_minus_baseline_ca_lddt']=d.ca_lddt-d.baseline_ca_lddt
disc=d[(d.dataset=='cameo')&(d.hash_fold10<=5)]
cuts=m.q_thresholds(disc.rg_len13_norm.dropna(),7)
d['frozen_rg_len13_norm_q7_level']=m.apply_levels(d.rg_len13_norm,cuts)
print('Discovery cutpoints:',cuts);print(d.groupby(['dataset','hash_fold10']).size())

# Original notebook cell 4
old=read(D/'frozen_reference_topology_axis_curves.csv')
for name,mask in [('cameo_discovery_hash0_5',(d.dataset=='cameo')&(d.hash_fold10<=5)),('cameo_confirmation_hash6_9',(d.dataset=='cameo')&(d.hash_fold10>=6))]:
    sub=d[mask];curve=m.curve_for_feature(sub,'rg_len13_norm','frozen_rg_len13_norm_q7_level',name,'rg_len13_norm')
    expected=old[(old.subset==name)&(old.level_schema=='frozen_rg_len13_norm_q7_level')&(old.feature=='rg_len13_norm')].sort_values('axis_level')
    np.testing.assert_allclose(curve.sort_values('axis_level').ca_lddt_mean,expected.ca_lddt_mean,atol=1e-12)
    plot_rows(sub,'frozen_rg_len13_norm_q7_level',label=name)
    print(m.regression_axis(sub,'rg_len13_norm'));print(m.summarize(curve))
