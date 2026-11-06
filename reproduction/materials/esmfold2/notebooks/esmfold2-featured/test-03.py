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
    assert a.index.equals(b.index),'Group identities changed'
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
D=A/'03_external_flexible';m=module(D/'scripts/summarize_expanded_external_spc.py');g=module(P/'scripts/global_hypothesis_audit.py')
f=read(D/'spectra_ready_dataset/candidate_features.csv');r=read(D/'target_model_results.scaled_samples.csv')
assert f.target_id.is_unique
x=g.add_features(f.assign(sequence_clean=f.sequence.str.upper()))
for a,b in [('proline_fraction_calc','proline_fraction'),('chou_beta_minus_helix_calc','chou_beta_minus_helix'),('chou_turn_minus_helix_calc','chou_turn_minus_helix')]:
    np.testing.assert_allclose(x[a],f[b],atol=1e-10)
rank=f[['proline_fraction_prior_pct','chou_beta_minus_helix_prior_pct','chou_turn_minus_helix_prior_pct']].mean(axis=1)
np.testing.assert_allclose(rank,f.flexible_nonhelical_composite_rank,atol=1e-12)
selected=m.best_by(r,'ptm');d=f.merge(selected,on='target_id',validate='one_to_one',suffixes=('','_metric'));d=d[(d.status=='ok')&d.ca_lddt.notna()]
print('Exact confirmation targets:',len(d),'prediction samples:',len(r));display(d[['target_id','sample_index','external_axis_level','ca_lddt']].head())

# Original notebook cell 4
curve,effect=m.summarize('best_by_ptm',selected,f)
equal_curve(curve,D/'expanded_external_spc_by_level.csv',['external_axis_level'],['n','ca_lddt_mean','rank_mean'])
plot_rows(d,'external_axis_level',label='Original external confirmation')
print(json.dumps(effect,indent=2))
for policy,col in [('best_by_plddt','plddt_mean'),('oracle_best_ca_lddt','ca_lddt')]:
    c,e=m.summarize(policy,m.best_by(r,col),f);print(policy,json.dumps(e))

# Original notebook cell 5
discovery=read(A/'02_flexible_nonhelical/primary_axis_membership.csv')
assert discovery.target_id.is_unique and len(discovery)==138
level='axis_flexible_nonhelical_composite_rank_q7_level'
top=discovery[discovery[level]==6];rest=discovery[discovery[level]!=6]
print('Original discovery highest-minus-rest:',top.ca_lddt.mean()-rest.ca_lddt.mean())
print('Discovery contrast bootstrap:',m.bootstrap_diff(top.ca_lddt.to_numpy(),rest.ca_lddt.to_numpy()))
plot_rows(discovery,level,label='Original CAMEO discovery')
print('Confirmation-minus-discovery is not a paired experiment: these are separate cohorts.')
