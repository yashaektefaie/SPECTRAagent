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
D=A/'06_charge_confirmation';m=module(D/'scripts'/'summarize_charge_confirmation.py')
f=read(D/'spectra_ready_dataset'/'candidate_features.csv');r=read(D/'target_model_results.scaled_samples.csv')
assert f.target_id.is_unique
np.testing.assert_array_equal(f.sequence.str.len(),f.seq_len)
print('Exact targets:',len(f),'prediction sample rows:',len(r));display(f[['target_id','sequence','charge_level']].head())

# Original notebook cell 4
b=module(D/'scripts/build_fresh_charge_panel.py')
calc=f.sequence.str.upper().map(lambda s:abs(sum(s.count(a) for a in 'KRH')-sum(s.count(a) for a in 'DE'))/len(s))
np.testing.assert_allclose(calc,f.calc_seq_abs_charge_per_residue,atol=1e-12)
pool=read(D/'spectra_ready_dataset/eligible_unscored_charge_pool.csv');pool=b.assign_charge_levels(pool)
x=f[['target_id','charge_level']].merge(pool[['target_id','charge_level']],on='target_id',validate='one_to_one',suffixes=('_old','_new'))
assert len(x)==len(f) and (x.charge_level_old==x.charge_level_new).all()
print('Sequence-feature/level checks completed.')

# Original notebook cell 5
selected=m.best_by(r,'ptm');curve,effect=m.summarize_selection('best_by_ptm',selected,f);d=f.merge(selected,on='target_id',suffixes=('','_metric'))
equal_curve(curve,D/'analysis_outputs'/'charge_confirmation_level_curves.csv',['charge_level'],['n','ca_lddt_mean'])
plot_rows(d,'charge_level',label='Original confirmation panel')
print(json.dumps(effect,indent=2))
