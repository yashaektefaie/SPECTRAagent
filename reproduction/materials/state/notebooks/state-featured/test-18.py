# Original notebook cell 1
from pathlib import Path
import sys, json, hashlib, io, contextlib, inspect
import numpy as np
import pandas as pd
from IPython.display import display, Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'STATE_INPUTS.json').is_file()), None)
assert root is not None, 'Extract the complete STATE input bundle and open this notebook inside it.'
assets = root/'assets'
manifest = json.loads((root/'STATE_INPUTS.json').read_text())
def checked(relative):
    p = assets/relative
    assert hashlib.sha256(p.read_bytes()).hexdigest() == manifest[relative]['sha256'], relative
    return p
P = assets/'primary'
sys.path.insert(0, str(P/'scripts'))
import state_spectra_loop as base
base.ROOT=P; base.TARGET=checked('primary/target_model_results.csv'); base.EXT=P/'external'
raw = base.load_results()
assert raw.row_key.is_unique
print(f'{len(raw):,} scored perturbation/context/representation records. Individual cells are not the sample unit.')

# Original notebook cell 2
def compare_archive(actual, relative, keys, columns):
    expected = pd.read_csv(checked('primary/'+relative))
    a=actual.set_index(keys).sort_index(); b=expected.set_index(keys).sort_index()
    assert a.index.equals(b.index), 'Archived sample/level identities differ'
    for col in columns:
        np.testing.assert_allclose(a[col], b[col], rtol=1e-9, atol=1e-10, equal_nan=True, err_msg=col)
    print('Matched archived identities and values:', relative)

def plot_rows(df, level_col, metric='overlap_at_N', grouping=('spectra_subset','representation')):
    # New replay intervals use the original observation unit (perturbation within context).
    # They quantify sample resampling, not unseen-patient/site or fresh-inference uncertainty.
    rows=[]; rng=np.random.default_rng(7)
    for keys,g in df.groupby([*grouping,level_col],observed=True):
        v=pd.to_numeric(g[metric],errors='coerce').dropna().to_numpy(float)
        if not len(v):continue
        means=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(1000)])
        rows.append(dict(zip([*grouping,'level'],keys)) | {'n':len(v),'mean':v.mean(),'lo':np.quantile(means,.025),'hi':np.quantile(means,.975)})
    table=pd.DataFrame(rows); display(table)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    levs=sorted(table.level.unique(),key=str)
    for keys,g in table.groupby(list(grouping),observed=True):
        g=g.assign(x=g.level.map({v:i for i,v in enumerate(levs)})).sort_values('x')
        ax.errorbar(g.x,g['mean'],yerr=np.vstack([g['mean']-g.lo,g.hi-g['mean']]),marker='o',capsize=2,label=str(keys),linewidth=1)
    ax.set_xticks(range(len(levs)),[str(v) for v in levs],rotation=25,ha='right');ax.set_ylabel(metric);ax.set_xlabel('Declared axis level');ax.legend(fontsize=7)
    ax.spines[['top','right']].set_visible(False)
    b=io.BytesIO();fig.savefig(b,format='png',dpi=120);plt.close(fig);display(Image(data=b.getvalue()))
    return table

# Original notebook cell 3
few=pd.read_csv(checked('primary/spectra_ready_dataset/multi_pathway_wikipathways_fewshot_examples.csv'),low_memory=False)
zero=pd.read_csv(checked('primary/spectra_ready_dataset/multi_pathway_wikipathways_zeroshot_examples.csv'),low_memory=False)
data=pd.concat([few,zero],ignore_index=True)
level='support_level';feature='support_norm_log_mean'
print('Exact retained sample identities:',len(data),'unique scored rows:',data.row_key.nunique())
display(data[['row_key','context','perturbation','spectra_subset',feature,level]].head())

# Original notebook cell 4
d=few[(few.spectra_subset=='discovery_hepg2_k562')&(few.feature_count>0)]
cuts=base.quantile_cutpoints_for_axis(d.support_norm_log_mean)
recomputed=base.assign_support_bins(data.support_norm_log_mean,data.feature_count,cuts)
assert (recomputed==data.support_level).all()
print('Recomputed levels match every archived record; discovery cutpoints:',cuts)

# Original notebook cell 5
# These are official saved evaluation outputs. The archive does not contain the
# corresponding predicted expression vectors and ground truth matrices.
# Therefore overlap@N, DE recall and Pearson delta cannot be independently re-scored here.
assert np.allclose(data.overlap_at_N, raw.set_index('row_key').loc[data.row_key,'overlap_at_N'],equal_nan=True)
print('Every selected outcome links to the original scored row_key; fresh inference and metric re-scoring remain unavailable.')

# Original notebook cell 6
curve=base.summarize_levels(data,level,feature,['overlap_at_N'])
compare_archive(curve,'artifacts/spc_multi_pathway_wikipathways.csv',['spectra_subset','representation','level'],['n_rows','similarity_mean','overlap_at_N_mean'])
plot_rows(data,level)
