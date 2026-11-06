# Original notebook cell 2
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

# Original notebook cell 4
P=ROOT/'assets/kimi-r1'
dev=pd.read_csv(P/'results/merged_per_unit.tsv',sep='\t')
confirm=pd.read_csv(P/'results/confirm/merged_per_unit.tsv',sep='\t')
plan=pd.read_csv(P/'results/panel_final.tsv',sep='\t')
plan_c=pd.read_csv(P/'results/confirm/panel_final.tsv',sep='\t')
assert dev.rep.is_unique and confirm.rep.is_unique
assert len(dev)==320 and len(confirm)==340
assert set(dev.rep)==set(plan.rep) and set(confirm.rep)==set(plan_c.rep)
assert not set(dev.rep)&set(confirm.rep)
clusters=pd.read_csv(P/'results/testclu_cluster.tsv',sep='\t',header=None,names=['cluster','member'])
cluster_of=dict(zip(clusters.member,clusters.cluster))
assert all(x in cluster_of for x in list(dev.rep)+list(confirm.rep))
assert not {cluster_of[x] for x in dev.rep}&{cluster_of[x] for x in confirm.rep}
print('Discovery320 and confirmation340 have disjoint original30%-identity clusters. Both are from the same prospective PDB pool.')
display(dev.groupby('levelA').size().rename('n').to_frame());display(confirm.groupby('group').size().rename('n').to_frame())

# Original notebook cell 6
# Execute only function definitions from the original source; its I/O entrypoint is not run.
import ast,types
original=(P/'scripts/04_select_panel.py').read_text();tree=ast.parse(original)
functions=ast.Module(body=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.Import,ast.ImportFrom))],type_ignores=[])
ns={};exec(compile(functions,'original_04_select_panel.py','exec'),ns)
for frame,prefix in [(dev,'results'),(confirm,'results/confirm')]:
    sequences={r:str(np.load(P/prefix/'native'/f'{r}.npz')['seq']) for r in frame.rep}
    actual=frame.rep.map(lambda r:ns['lc_fraction'](sequences[r]))
    np.testing.assert_allclose(actual,frame.lcfrac,atol=5.01e-5,rtol=0)
    frame['lc_recomputed']=actual
    frame['levelB_recomputed']=actual.map(ns['level_b'])
    frame['levelA_recomputed']=frame.max_pident.map(ns['level_a'])
    assert np.array_equal(frame['len'],frame.rep.map(lambda r:len(sequences[r])))
assert np.array_equal(dev.levelA,dev.levelA_recomputed)
assert np.array_equal(dev.levelB,dev.levelB_recomputed)
assert np.array_equal(confirm.group,np.where(confirm.max_pident<30,'orphan','nonorphan'))
print('Low-complexity fraction recomputed from full original input sequences for all660 units. Identity/orphan levels reconstructed from retained per-sequence identity values.')
print('Original MMseqs hit table/reference FASTA not included: maximum identity itself cannot be recomputed from an alignment search.')
display(dev.groupby('levelA').agg(n=('rep','size'),mean_identity=('max_pident','mean'),mean_low_complexity=('lc_recomputed','mean')))

# Original notebook cell 8
sc=load_module('original_08_score',P/'scripts/08_score.py')
errors=[]
for frame,prefix in [(dev,'results'),(confirm,'results/confirm')]:
    rescored=[]
    for row in frame.itertuples():
        native=np.load(P/prefix/'native'/f'{row.rep}.npz')
        prediction=sc.pred_ca(P/prefix/'pred'/f'{row.rep}.pdb')
        idx=native['res_idx'];coords=native['ca']
        assert len(prediction)==row.len and len(idx)==len(coords)
        tm,gts,gha,rmsd=sc.tm_gdt(prediction[idx],coords,len(idx))
        rescored.append(tm);errors.append(abs(tm-row.tm))
    np.testing.assert_allclose(rescored,frame.tm,atol=5.01e-5,rtol=0)
    # Original table rounded scores to4 decimals before aggregation; retain that behavior.
    frame['tm_recomputed']=[float(f'{x:.4f}') for x in rescored]
    assert np.array_equal(frame.tm_recomputed,frame.tm)
print('Original scoring function rescored all660 archived model structures; maximum deviation before original rounding:',max(errors))
print('Reference baseline scores remain original per-unit values. Template raw coordinate-transfer inputs were not recovered, so those scores are not independently regenerated.')

# Original notebook cell 10
table=mean_curve(dev,'levelA',{'ESMFold v1':'tm_recomputed','Template reference (retained)':'base_tm'})
display(table);draw_curve(table,'Original graded identity level (A1 high → A4 low)')
contrast=welch(dev.loc[dev.levelA=='A1','tm_recomputed'],dev.loc[dev.levelA=='A4','tm_recomputed'])
print('Graded endpoint:',contrast)
print('Original frozen minimum gap0.15 met:',contrast['difference']>=.15)

# Original notebook cell 12
comparison=mean_curve(confirm,'group',{'ESMFold v1':'tm_recomputed','Template reference (retained)':'base_tm'})
display(comparison)
non=confirm.loc[confirm.group=='nonorphan','tm_recomputed'];orp=confirm.loc[confirm.group=='orphan','tm_recomputed']
effect=welch(non,orp);effect['p_one_sided']=stats.ttest_ind(non,orp,equal_var=False,alternative='greater').pvalue
record=json.loads((P/'results/confirm/confirm_summary.json').read_text())
assert abs(effect['difference']-record['gap'])<1e-12
assert abs(effect['p_one_sided']-record['welch_p_one_sided'])<1e-12
print('Fresh-cluster orphan confirmation within ESMFold v1:',effect)
length_checks=[]
for low,high in [(40,150),(150,250),(250,401)]:
    part=confirm[confirm['len'].between(low,high,inclusive='left')]
    gap=part.loc[part.group=='nonorphan','tm_recomputed'].mean()-part.loc[part.group=='orphan','tm_recomputed'].mean()
    length_checks.append({'length_bin':f'{low}–{high-1}','n':len(part),'gap_nonorphan_minus_orphan':gap})
display(pd.DataFrame(length_checks))
print('No result in this notebook evaluates ESMFold2: the target in the original run was ESMFold v1.')
