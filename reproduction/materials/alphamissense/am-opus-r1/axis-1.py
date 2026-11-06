# Original notebook cell 3
from pathlib import Path
import hashlib, json, sys, platform
import numpy as np, pandas as pd, scipy, pyhmmer
from scipy.stats import spearmanr, wilcoxon
from Bio.Align import substitution_matrices
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
assert (ROOT / 'INPUT_MANIFEST.json').exists(), 'Run this notebook from its packaged folder.'
for item in json.loads((ROOT / 'INPUT_MANIFEST.json').read_text()):
    p = ROOT / item['path']
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], item['path']
assert pyhmmer.__version__ == '0.12.1', 'The historical feature calculation used pyhmmer 0.12.1.'
OUT = ROOT / 'recomputed'; OUT.mkdir(exist_ok=True)
print('Input hashes verified. Python', platform.python_version(), 'pyhmmer', pyhmmer.__version__, 'pandas',pd.__version__, 'scipy',scipy.__version__)

# Original notebook cell 5
pred = pd.read_csv(ROOT/'data/predictions.csv')
meta = pd.read_csv(ROOT/'data/assay_full.csv')
split = pd.read_csv(ROOT/'data/assay_split.csv')
assert not pred.duplicated(['assay','acc','position','mutant']).any()
assert pred[['am_pathogenicity','DMS_score']].notna().all().all()
assert split.groupby('acc').split.nunique().max() == 1
assert set(pred.assay) == set(split.assay) == set(meta.assay)
assert len(pred)==271201 and pred.assay.nunique()==73 and pred.acc.nunique()==64
variants = pred.merge(meta[['assay','UniProt_ID','coarse_selection_type','MSA_Neff_L']], on='assay', validate='many_to_one')
variants = variants.merge(split[['assay','split']], on='assay', validate='many_to_one')
variants[['assay','acc','position','mutant','split']].to_csv(OUT/'sample_split.csv.gz',index=False)
display(variants.groupby('split').agg(variants=('mutant','size'),assays=('assay','nunique'),proteins=('acc','nunique')))
display(variants.head())

# Original notebook cell 7
import pyhmmer, time, numpy as np
from pyhmmer.easel import Alphabet, TextSequence, SequenceFile
from pyhmmer.plan7 import Pipeline

abc = Alphabet.amino()

def load_db(path):
    t=time.time()
    with SequenceFile(path, digital=True, alphabet=abc) as sf:
        seqs = sf.read_block()
    return seqs

def per_position_features(name, seq, db, max_hits=8000, incE=1e-3):
    """Return per-query-position coverage counts and residue lists via phmmer."""
    L=len(seq)
    q = TextSequence(name=name.encode(), sequence=seq).digitize(abc)
    pli = Pipeline(abc, incE=incE)
    hits = pli.search_seq(q, db)
    cov = np.zeros(L, dtype=int)          # homologs aligned (non-gap) at each query pos
    # store residues per position for conservation (cap memory by counting)
    from collections import defaultdict, Counter
    rescount = [Counter() for _ in range(L)]
    nhits=min(len(hits), max_hits)
    for hi,hit in enumerate(hits):
        if hi>=max_hits: break
        for dom in hit.domains:
            ali = dom.alignment
            qpos = ali.hmm_from - 1   # 0-based query index pointer (match states)
            hmmseq = ali.hmm_sequence
            tgtseq = ali.target_sequence
            for hc, tc in zip(hmmseq, tgtseq):
                if hc != '.':            # match state -> consumes a query position
                    if 0 <= qpos < L and tc not in ('-','.'):
                        cov[qpos]+=1
                        rescount[qpos][tc.upper()]+=1
                    qpos+=1
    # conservation: shannon entropy per position (bits) over observed residues incl self
    ent=np.full(L, np.nan)
    for i in range(L):
        c=rescount[i]
        tot=sum(c.values())
        if tot>0:
            ps=np.array([v/tot for v in c.values()])
            ent[i]=-(ps*np.log2(ps)).sum()
    return dict(L=L, nhits=nhits, total_hits=len(hits), cov=cov, entropy=ent)


# Original notebook cell 8
unique = meta.drop_duplicates('target_seq')[['UniProt_ID','target_seq']].reset_index(drop=True)
db = load_db(str(ROOT/'data/uniprot_sprot.fasta.gz'))
assert len(db)==575503 and len(unique)==64
rows=[]
for i, protein in unique.iterrows():
    f=per_position_features(protein.UniProt_ID, protein.target_seq, db)
    for p in range(f['L']):
        rows.append(dict(UniProt_ID=protein.UniProt_ID, position=p+1, wt_seq=protein.target_seq[p],
                         cov=int(f['cov'][p]), cov_frac=float(f['cov'][p]/max(f['total_hits'],1)),
                         entropy=float(f['entropy'][p]), total_hits=f['total_hits']))
    if (i+1)%16==0: print('Proteins recomputed:',i+1)
features=pd.DataFrame(rows)
expected=pd.read_parquet(ROOT/'expected/pos_features.parquet')
check=features.merge(expected,on=['UniProt_ID','position'],suffixes=('_new','_old'),validate='one_to_one')
assert len(check)==len(features)==len(expected)==32071
for col in ['cov','cov_frac','entropy','total_hits']:
    np.testing.assert_allclose(check[col+'_new'],check[col+'_old'],rtol=1e-12,atol=1e-12,equal_nan=True)
features.to_parquet(OUT/'residue_features.parquet', index=False)
variants=variants.merge(features,on=['UniProt_ID','position'],validate='many_to_one')
variants['constraint']=-variants.entropy
print('All 32,071 residue features reproduced to numerical precision.')

# Original notebook cell 10
variants['wt']=variants.mutant.str[0]
variants['mut_aa']=variants.mutant.str[-1]
blosum=substitution_matrices.load('BLOSUM62')
variants['blosum62']=[float(blosum[w,m]) for w,m in zip(variants.wt,variants.mut_aa)]
original=pd.read_parquet(ROOT/'data/original_variant_baseline_scores.parquet')
keys=['assay','acc','position','mutant']
variants=variants.merge(original[keys+['pssm']],on=keys,validate='one_to_one')
validate=variants.merge(original[keys+['am_pathogenicity','DMS_score','blosum62','pssm']],on=keys,suffixes=('_new','_old'),validate='one_to_one')
for col in ['am_pathogenicity','DMS_score','blosum62','pssm']:
    np.testing.assert_allclose(validate[col+'_new'],validate[col+'_old'],rtol=0,atol=1e-12)
variants.to_parquet(OUT/'variant_scores_and_features.parquet',index=False)
display(variants[keys+['split','cov_frac','entropy','am_pathogenicity','DMS_score','blosum62','pssm']].head())

# Original notebook cell 12
def assign_levels(sub, feature, nlevels):
    positions=sub[['position',feature]].drop_duplicates('position').sort_values(feature).reset_index(drop=True)
    positions['level']=pd.qcut(positions[feature].rank(method='first'),nlevels,labels=False)
    return sub.merge(positions[['position','level']],on='position',how='left',validate='many_to_one')

def score_axis(feature, nlevels):
    rows=[]; memberships=[]
    for assay, group in variants.groupby('assay'):
        if group[feature].nunique()<nlevels: continue
        group=assign_levels(group,feature,nlevels)
        records=[]
        for level, cell in group.groupby('level'):
            if len(cell)<50: break
            record=dict(assay=assay,level=int(level),split=cell.split.iloc[0],acc=cell.acc.iloc[0],
                        n=len(cell),positions=cell.position.nunique(),entropy=cell.entropy.mean(),
                        cov_frac=cell.cov_frac.mean(),dms_std=cell.DMS_score.std(),
                        selection=cell.coarse_selection_type.iloc[0],total_hits=cell.total_hits.iloc[0])
            for name,col in [('AM','am_pathogenicity'),('PSSM','pssm'),('BLOSUM','blosum62')]:
                record[name]=abs(spearmanr(cell[col],cell.DMS_score).statistic)
            records.append(record)
        if len(records)==nlevels and all(np.isfinite(r['AM']) for r in records):
            rows.extend(records)
            memberships.append(group[['assay','acc','position','mutant','split','level',feature]])
    table=pd.DataFrame(rows)
    pd.concat(memberships).to_csv(OUT/f'{feature}_{nlevels}_membership.csv.gz',index=False)
    table.to_csv(OUT/f'{feature}_{nlevels}_assay_scores.csv',index=False)
    return table

def summarize(table):
    out=[]
    for method in ['AM','PSSM','BLOSUM']:
        pivot=table.pivot(index='assay',columns='level',values=method).dropna()
        values=pivot.to_numpy(); rng=np.random.default_rng(0)
        resamples=rng.integers(0,len(values),(5000,len(values)))
        boot=values[resamples].mean(axis=1)
        for i,level in enumerate(pivot.columns):
            low,high=np.quantile(boot[:,i],[.025,.975])
            out.append(dict(method=method,level=int(level),mean=values[:,i].mean(),low=low,high=high,n_assays=len(values)))
    return pd.DataFrame(out)

conservation3=score_axis('constraint',3); conservation5=score_axis('constraint',5)
coverage3=score_axis('cov_frac',3)
for n,new in [(3,conservation3),(5,conservation5)]:
    name='terciles' if n==3 else 'quintiles'
    expected=pd.read_csv(ROOT/f'expected/spc_conservation_{name}.csv').rename(columns={'lvl':'level'})
    check=new.merge(expected,on=['assay','level'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(check)==len(new)==len(expected)
    for c in ['AM','PSSM','BLOSUM','entropy','dms_std']:
        np.testing.assert_allclose(check[c+'_new'],check[c+'_old'],rtol=1e-10,atol=1e-10)
    print(name,':',new.assay.nunique(),'assays; all historical assay-level values reproduced.')

fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
for ax,table,label in zip(axes,[conservation3,conservation5,coverage3],['Conservation terciles','Conservation quintiles','Coverage terciles']):
    stats=summarize(table)
    for method,g in stats.groupby('method',sort=False):
        ax.errorbar(g.level,g['mean'],yerr=[g['mean']-g.low,g.high-g['mean']],marker='o',capsize=3,label=method)
    ax.set(xlabel='Level (low → high)',ylabel='Mean within-assay |Spearman|',title=label)
    ax.legend(frameon=False)
    stats.to_csv(OUT/(label.lower().replace(' ','_')+'_curve.csv'),index=False)
plt.show()

# Original notebook cell 14
def effect_summary(table,label):
    am=table.pivot(index='assay',columns='level',values='AM')
    bl=table.pivot(index='assay',columns='level',values='BLOSUM')
    ps=table.pivot(index='assay',columns='level',values='PSSM')
    hi=max(am.columns); lo=min(am.columns)
    rows=[]
    for name,delta in [('AM high-low',am[hi]-am[lo]),('AM drop minus BLOSUM drop',(am[hi]-am[lo])-(bl[hi]-bl[lo])),('AM drop minus PSSM drop',(am[hi]-am[lo])-(ps[hi]-ps[lo]))]:
        v=delta.dropna().to_numpy();rng=np.random.default_rng(0)
        boot=v[rng.integers(0,len(v),(5000,len(v)))].mean(axis=1)
        low,high=np.quantile(boot,[.025,.975])
        rows.append(dict(subset=label,comparison=name,n_assays=len(v),mean=v.mean(),low=low,high=high,positive=int((v>0).sum())))
    return rows

effects=[]
for label,table in [('All conservation',conservation3),('Discovery',conservation3[conservation3.split=='discovery']),('Confirmation',conservation3[conservation3.split=='confirmation']),('Excluding stability',conservation3[conservation3.selection!='Stability']),('At least 30 homologs',conservation3[conservation3.total_hits>=30]),('Coverage',coverage3)]:
    effects.extend(effect_summary(table,label))
effects=pd.DataFrame(effects);display(effects.round(6));effects.to_csv(OUT/'paired_effects.csv',index=False)
print('DMS standard deviation by conservation level:')
display(conservation3.groupby('level').dms_std.mean())
means=conservation5.groupby('level').AM.mean()
print('Conservation quintile AM means:',means.tolist())
print('Strictly non-decreasing:',bool((means.diff().dropna()>=0).all()))
assert not (means.diff().dropna()>=0).all()

# Original notebook cell 16
accuracy=[]
for assay,g in variants.groupby('assay'):
    accuracy.append(dict(assay=assay,agreement=abs(spearmanr(g.am_pathogenicity,g.DMS_score).statistic),depth=g.MSA_Neff_L.iloc[0],selection=g.coarse_selection_type.iloc[0],n=len(g)))
accuracy=pd.DataFrame(accuracy);accuracy.to_csv(OUT/'whole_protein_depth_assays.csv',index=False)
depth_results=[]
for label,g in [('All assays',accuracy),('Without stability assays',accuracy[accuracy.selection!='Stability']),('Stability only',accuracy[accuracy.selection=='Stability'])]:
    rho,p=spearmanr(g.depth,g.agreement)
    depth_results.append(dict(group=label,n_assays=len(g),rho=rho,p=p))
display(pd.DataFrame(depth_results))
pd.DataFrame(depth_results).to_csv(OUT/'whole_protein_depth_control.csv',index=False)

# Original notebook cell 18
manifest=[]
for path in sorted(OUT.iterdir()):
    if path.is_file():manifest.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
(OUT/'OUTPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print(len(manifest),'recomputed output files saved and hashed.')
