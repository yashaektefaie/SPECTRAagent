# Original notebook cell 1
from pathlib import Path
import json,importlib.util
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display,Image
import io
ROOT=Path.cwd()
while not (ROOT/'assets').is_dir() and ROOT.parent!=ROOT: ROOT=ROOT.parent
assert (ROOT/'assets').is_dir()
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def mean_curve(frame,group,columns):
    rows=[]
    for level,g in frame.groupby(group,sort=True):
        for model,col in columns.items():
            v=g[col].dropna().to_numpy();n=len(v)
            mean=v.mean() if n else np.nan
            half=stats.t.ppf(.975,n-1)*stats.sem(v) if n>1 else np.nan
            rows.append({'level':level,'model':model,'n':n,'mean':mean,'lo':mean-half,'hi':mean+half})
    return pd.DataFrame(rows)
def draw_curve(table,xlabel):
    fig,ax=plt.subplots(figsize=(7,3.5))
    for model,g in table.groupby('model',sort=False):
        ax.errorbar(g.level,g['mean'],yerr=[g['mean']-g.lo,g.hi-g['mean']],marker='o',capsize=3,label=model)
    ax.set(xlabel=xlabel,ylabel='TM-score');ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False);fig.tight_layout();buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=130);display(Image(data=buffer.getvalue()));plt.close(fig)
def welch(a,b):
    a,b=np.array(a),np.array(b);diff=a.mean()-b.mean();v1=a.var(ddof=1)/len(a);v2=b.var(ddof=1)/len(b);se=np.sqrt(v1+v2)
    df=(v1+v2)**2/(v1*v1/(len(a)-1)+v2*v2/(len(b)-1));h=stats.t.ppf(.975,df)*se
    return {'difference':diff,'lo':diff-h,'hi':diff+h,'p_two_sided':stats.ttest_ind(a,b,equal_var=False).pvalue}

# Original notebook cell 3
P=ROOT/'assets/kimi-r2/audit'
scored=pd.read_csv(P/'results/scores.csv');enriched=pd.read_csv(P/'results/per_unit_enriched.csv')
axis=pd.read_csv(P/'data/axis_table.csv');plan=pd.read_csv(P/'data/sample_plan_full.csv')
assert scored.entity.is_unique and len(scored)==160
assert set(scored.entity)==set(plan.entity)
assert scored.groupby('bin').size().eq(40).all()
assert set(scored.entity).issubset(set(axis.loc[axis.rep,'entity']))
clusters=pd.read_csv(P/'work_mmseqs/clus_cluster.tsv',sep='\t',header=None,names=['cluster','member'])
cluster_of=dict(zip(clusters.member,clusters.cluster));assert len({cluster_of[x] for x in scored.entity})==160
pilot=pd.read_csv(P/'data/sample_plan.csv');expansion=pd.read_csv(P/'data/sample_plan_expansion.csv')
assert not set(pilot.entity)&set(expansion.entity)
assert set(pilot.entity)|set(expansion.entity)==set(scored.entity)
print('Original 160 scored entities: 32 pilot + 128 expansion; 40 per level; distinct 40%-identity clusters.')
display(scored.groupby('bin').agg(n=('entity','size'),mean_identity=('ident','mean')))

# Original notebook cell 5
hits=pd.read_csv(P/'work_mmseqs/res.tsv',sep='\t',header=None,names=['query','target','fident','length','qcov','tcov','evalue','bits'])
max_identity=hits.groupby('query').fident.max()
axis['recomputed_identity']=axis.entity.map(max_identity).fillna(0)
np.testing.assert_allclose(axis.recomputed_identity,axis.ident,atol=5.01e-5,rtol=0)
def identity_bin(x):
    return 'I1_ge80' if x>=.8 else 'I2_40_80' if x>=.4 else 'I3_20_40' if x>=.2 else 'I4_lt20'
axis['recomputed_bin']=axis.recomputed_identity.map(identity_bin)
assert np.array_equal(axis.bin,axis.recomputed_bin)
scored=scored.merge(axis[['entity','recomputed_identity','recomputed_bin']],on='entity',validate='one_to_one')
assert np.array_equal(scored.bin,scored.recomputed_bin)
print('Maximum identity and level assignment recomputed from all retained original MMseqs2 hits for',len(axis),'candidate sequences.')
print('The underlying reference FASTA/search database is not supplied; the search itself is not rerun.')
display(scored.groupby('bin').recomputed_identity.agg(['min','mean','max','size']))

# Original notebook cell 7
from tmtools import tm_align
from Bio.SeqUtils import seq1
import importlib.metadata
print('Rescoring with tmtools',importlib.metadata.version('tmtools'),'; original script used standalone TMalign20220412. Agreement is checked per entity.')
def ca_coordinates(path):
    records=[s for s in path.read_text().splitlines() if s.startswith('ATOM') and s[12:16].strip()=='CA']
    xyz=np.array([[float(s[30:38]),float(s[38:46]),float(s[46:54])] for s in records])
    sequence=''.join(seq1(s[17:20],custom_map={'MSE':'M'}) for s in records)
    return xyz,sequence
replay=[]
for row in scored.itertuples():
    native,seqn=ca_coordinates(P/'results/exp'/f'{row.entity}.pdb')
    pred,seqp=ca_coordinates(P/'results/pred'/f'{row.entity}.pdb')
    tm=tm_align(pred,native,seqp,seqn).tm_norm_chain2
    bpath=P/'results/ref'/f'{row.entity}.pdb'
    baseline=np.nan
    if bpath.exists():
        coords,sequence=ca_coordinates(bpath);baseline=tm_align(coords,native,sequence,seqn).tm_norm_chain2
    replay.append({'entity':row.entity,'model_rescore':tm,'reference_rescore':baseline,'observed_ca':len(native)})
replay=pd.DataFrame(replay);scored=scored.merge(replay,on='entity',validate='one_to_one')
scored['model_abs_error']=abs(scored.model_rescore-scored.tm_model)
scored['reference_abs_error']=abs(scored.reference_rescore-scored.tm_ref)
display(scored[['model_abs_error','reference_abs_error']].agg(['max','mean']))
# Small version-dependent alignment drift is reported, never silently substituted.
print('Number matching original 5-decimal score within0.0000051:',int((scored.model_abs_error<=.0000051).sum()),'of 160 target,',int((scored.reference_abs_error<=.0000051).sum()),'of',int(scored.tm_ref.notna().sum()),'reference.')
scored['tm_model_recomputed']=scored.model_rescore.round(5)
scored['tm_ref_recomputed']=scored.reference_rescore.round(5)
np.testing.assert_allclose(scored.tm_model_recomputed,scored.tm_model,atol=1e-12,rtol=0)
np.testing.assert_allclose(scored.tm_ref_recomputed,scored.tm_ref,atol=1e-12,rtol=0,equal_nan=True)
print('All 280 coordinate-rescored metrics match original scores after original 5-decimal rounding. Curves and controls below use those recomputed values, not the archived score columns. No fresh model inference was performed.')

# Original notebook cell 9
table=mean_curve(scored,'bin',{'ESMFold2 (coordinate-rescored)':'tm_model_recomputed','Template reference (coordinate-rescored)':'tm_ref_recomputed'})
display(table);draw_curve(table,'Identity level (I1 high → I4 no detected homology)')
effect=welch(scored.loc[scored.bin=='I4_lt20','tm_model_recomputed'],scored.loc[scored.bin=='I1_ge80','tm_model_recomputed'])
print('Original lowest-minus-highest identity endpoint:',effect)
print('Coordinate-rescore endpoint:',welch(scored.loc[scored.bin=='I4_lt20','model_rescore'],scored.loc[scored.bin=='I1_ge80','model_rescore']))
assert abs(effect['difference']+.086)<.002

# Original notebook cell 11
conf=json.loads((P/'data/entry_confounds.json').read_text())
scored['obs_frac_recomputed']=scored.observed_ca/scored['len']
scored['protein_entities']=scored.entity.map(lambda e:conf[e.split('_')[0]]['prot_ent'])
original=enriched.set_index('entity')
np.testing.assert_allclose(scored.obs_frac_recomputed,original.loc[scored.entity,'obs_frac'])
assert np.array_equal(scored.protein_entities,original.loc[scored.entity,'prot_ent'])
subsets={'All':scored,'Observed density ≥0.8':scored[scored.obs_frac_recomputed>=.8],'Single protein entity':scored[scored.protein_entities==1],'Both':scored[(scored.obs_frac_recomputed>=.8)&(scored.protein_entities==1)]}
controls=[]
for name,d in subsets.items():
    x=welch(d.loc[d.bin=='I4_lt20','tm_model_recomputed'],d.loc[d.bin=='I1_ge80','tm_model_recomputed'])
    controls.append(dict(subset=name,n=len(d),**x))
display(pd.DataFrame(controls))
exp=scored[scored.entity.isin(expansion.entity)]
print('Expansion-only endpoint:',welch(exp.loc[exp.bin=='I4_lt20','tm_model_recomputed'],exp.loc[exp.bin=='I1_ge80','tm_model_recomputed']))
print('No template reference exists for I4; do not replace missing baseline scores with zero or calculate a four-level paired baseline effect.')
print('Intermediate levels are nonmonotonic; endpoint significance does not establish a uniformly decreasing curve.')
