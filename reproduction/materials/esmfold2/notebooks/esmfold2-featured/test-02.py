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
D=A/'01_cameo_msa_depth';m=module(D/'scripts/finalize_spectra_outputs.py');reader=module(D/'scripts/analyze_spectra_axes.py')
f=read(D/'spectra_ready_dataset/candidate_features.csv');r=reader.read_results(checked(D/'target_model_results.csv'))
assert f.target_id.is_unique
np.testing.assert_array_equal(f.sequence.str.len(),f.seq_len)
np.testing.assert_allclose(np.log10(1+f.msa_nonquery_rows),f.msa_log10_nonquery_rows,atol=1e-12)
d=f.merge(r,on='target_id',validate='one_to_one');d=d[(d.status=='ok')&d.ca_lddt.notna()].copy()
d=m.add_axis(d);d['low_msa_lt3']=(d.msa_log10_nonquery_rows<3).astype(int)
print('Candidate targets:',len(f),'evaluated:',len(d));display(d[['target_id','sequence','seq_len','msa_nonquery_rows','ca_lddt']].head())

# Original notebook cell 4
curve=m.spc_table(d,'predeclared_cameo_difficulty','cameo_difficulty_level','cameo_ease_similarity')
old=read(D/'spectral_performance_curve.csv');old=old[old.axis=='predeclared_cameo_difficulty']
if len(old):
    np.testing.assert_allclose(curve.target_ca_lddt_mean,old.sort_values('level').target_ca_lddt_mean,atol=1e-12)
display(curve)
plot_rows(d,'cameo_difficulty_level')
print('Continuous MSA-depth association:',spearmanr(d.msa_log10_nonquery_rows,d.ca_lddt))

# Original notebook cell 5
controls=[m.linear_control(d,x) for x in [['low_msa_lt3'],['low_msa_lt3','seq_len','resolution','baseline_ca_lddt'],['low_msa_lt3','seq_len','resolution','baseline_ca_lddt','difficulty']]]
old=json.loads(checked(D/'control_analysis.json').read_text())['linear_controls']
for got,expected in zip(controls,old):
    for k,v in got['coef'].items():np.testing.assert_allclose(v,expected['coef'][k],rtol=1e-8,atol=1e-10)
print(json.dumps(controls,indent=2))
display(d.groupby('low_msa_lt3')[['ca_lddt','baseline_ca_lddt']].agg(['count','mean']))
