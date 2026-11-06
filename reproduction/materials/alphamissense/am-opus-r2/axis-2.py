# Original notebook cell 3
from pathlib import Path
import hashlib,json,tarfile,os,sys,platform
import numpy as np,pandas as pd
from scipy.stats import spearmanr,wilcoxon
from IPython.display import display
import matplotlib.pyplot as plt
ROOT=Path.cwd(); INPUT=ROOT/'inputs'; OUT=ROOT/'recomputed';OUT.mkdir(exist_ok=True)
for item in json.loads((ROOT/'INPUT_MANIFEST.json').read_text()):
    path=ROOT/item['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],item['path']
print('All recovered inputs and programs verified.')
with tarfile.open(INPUT/'raw_references.tar.gz') as archive:
    for member in archive.getmembers():
        assert not member.issym() and not member.islnk()
        assert str((OUT/member.name).resolve()).startswith(str(OUT.resolve())+os.sep)
    archive.extractall(OUT,filter='data')
assert len(list((OUT/'msa').glob('*.a2m')))==66
assert len(list((OUT/'af').glob('*.pdb')))==61
os.chdir(OUT)
for name in ['assay_msa_map.csv','DMS_substitutions.csv','assay_meta.csv']:
    (OUT/name).write_bytes((INPUT/name).read_bytes())

# Original notebook cell 5
pred=pd.read_csv(INPUT/'predictions.csv')
assert len(pred)==271201 and pred.assay.nunique()==73 and pred.acc.nunique()==64
assert not pred.duplicated(['assay','acc','position','mutant']).any()
accs=np.array(sorted(pred.acc.unique()));rng_split=np.random.default_rng(7);rng_split.shuffle(accs)
discovery=set(accs[:len(accs)//2]);confirmation=set(accs[len(accs)//2:])
assert not discovery & confirmation
pred['split']=np.where(pred.acc.isin(discovery),'discovery','confirmation')
pred[['assay','acc','position','mutant','split']].to_csv(OUT/'sample_split.csv.gz',index=False)
display(pred.groupby('split').agg(variants=('mutant','size'),assays=('assay','nunique'),proteins=('acc','nunique')))

# Original notebook cell 7
import numpy as np, pandas as pd, os, time, sys

AA="ACDEFGHIKLMNPQRSTVWY"
aa2i={c:i for i,c in enumerate(AA)}
GAP=set(b"-.")  # gap bytes

mp=pd.read_csv('assay_msa_map.csv')
# one MSA file may serve multiple assays; process unique files, store by file
files=mp.drop_duplicates('MSA_filename')[['MSA_filename','MSA_start','MSA_end','MSA_len','seq_len']]

os.makedirs('msa_feat',exist_ok=True)
covrows=[]
def load_matrix(path):
    # fast path: single line per seq
    with open(path,'rb') as f:
        data=f.read()
    lines=data.split(b'\n')
    seqs=[]
    cur=[]
    for ln in lines:
        if not ln: continue
        if ln[:1]==b'>':
            if cur: seqs.append(b''.join(cur)); cur=[]
        else:
            cur.append(ln.strip())
    if cur: seqs.append(b''.join(cur))
    L=len(seqs[0])
    # verify uniform
    buf=b''.join(seqs)
    if len(buf)%L!=0:
        # non-uniform: pad/skip
        seqs=[s for s in seqs if len(s)==L]
        buf=b''.join(seqs)
    arr=np.frombuffer(buf,dtype=np.uint8).reshape(-1,L)
    return arr

for _,r in files.iterrows():
    fn=r.MSA_filename; path=f'msa/{fn}'
    t0=time.time()
    arr=load_matrix(path)
    nseq,L=arr.shape
    # coverage: present = alphabetic (A-Z or a-z); gap = '-' or '.'
    isalpha=((arr>=65)&(arr<=90))|((arr>=97)&(arr<=122))
    present=isalpha  # letters (incl X)
    coverage=present.mean(axis=0)            # frac present per column
    # AA counts per column (uppercase standard AAs; fold lowercase to upper)
    upper=np.where((arr>=97)&(arr<=122), arr-32, arr)  # to upper
    cnt=np.zeros((L,20),dtype=np.float64)
    for i,c in enumerate(AA):
        cb=ord(c)
        cnt[:,i]=(upper==cb).sum(axis=0)
    np.save(f'msa_feat/{fn}.aacount.npy', cnt)
    for col in range(L):
        covrows.append((fn,col,int(nseq),float(coverage[col]),int(present[:,col].sum())))
    print(f'{fn}: nseq={nseq} L={L} in {time.time()-t0:.1f}s', flush=True)

cov=pd.DataFrame(covrows,columns=['MSA_filename','col0','nseq','coverage','n_present'])
cov.to_csv('msa_coverage_by_col.csv',index=False)
print('DONE total columns:',len(cov))

# Original notebook cell 8
new_cov=pd.read_csv(OUT/'msa_coverage_by_col.csv')
old_cov=pd.read_csv(INPUT/'msa_coverage_by_col.csv')
pd.testing.assert_frame_equal(new_cov,old_cov,check_exact=False,rtol=1e-12,atol=1e-12)
print(len(new_cov),'alignment columns reproduced.')

# Original notebook cell 10
import numpy as np, pandas as pd
from Bio.Align import substitution_matrices
AA="ACDEFGHIKLMNPQRSTVWY"; aa2i={c:i for i,c in enumerate(AA)}
blos=substitution_matrices.load("BLOSUM62")

df=pd.read_csv(INPUT/'predictions.csv')
df['wt']=df.mutant.str[0]; df['mt']=df.mutant.str[-1]
amap=pd.read_csv('assay_msa_map.csv')[['assay','MSA_filename','MSA_start']]
cov=pd.read_csv('msa_coverage_by_col.csv')  # MSA_filename,col0,nseq,coverage,n_present

df=df.merge(amap,on='assay',how='left')
df['col0']=df['position']-df['MSA_start']   # 0-based column

# attach coverage + nseq via (MSA_filename,col0)
df=df.merge(cov[['MSA_filename','col0','coverage','n_present','nseq']],on=['MSA_filename','col0'],how='left')

# PSSM score per variant from aacount matrices
pssm=np.full(len(df),np.nan)
alpha=1.0
aacount_cache={}
for fn,g in df.groupby('MSA_filename'):
    cnt=np.load(f'msa_feat/{fn}.aacount.npy')  # (L,20)
    tot=cnt.sum(axis=1,keepdims=True)+20*alpha
    freq=(cnt+alpha)/tot
    for idx,c0,wt,mt in zip(g.index,g.col0,g.wt,g.mt):
        if 0<=c0<cnt.shape[0] and wt in aa2i and mt in aa2i:
            pssm[idx]=np.log(freq[c0,aa2i[mt]]/freq[c0,aa2i[wt]])
df['pssm']=pssm

# BLOSUM62
def bl(w,m):
    try: return float(blos[w,m])
    except: return np.nan
df['blosum']=[bl(w,m) for w,m in zip(df.wt,df.mt)]

# sanity
print('rows',len(df))
print('coverage NaN:',df.coverage.isna().sum(),'(out-of-MSA-range positions)')
print('pssm NaN:',df.pssm.isna().sum())
print('blosum NaN:',df.blosum.isna().sum())
print('coverage describe:'); print(df.coverage.describe())
df.to_parquet('master.parquet')
print('saved master.parquet')

# Original notebook cell 11
new_master=pd.read_parquet(OUT/'master.parquet');old_master=pd.read_parquet(INPUT/'master.parquet')
pd.testing.assert_frame_equal(new_master,old_master,check_exact=False,rtol=1e-12,atol=1e-12)
print('All variant joins, coverage values, PSSM scores and BLOSUM62 scores reproduced.')

# Original notebook cell 13
import numpy as np, pandas as pd, glob, os
from Bio import Align
aa3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
ref=pd.read_csv('DMS_substitutions.csv').set_index('DMS_id')
df=pd.read_parquet('master.parquet'); df['wt']=df.mutant.str[0]

afseq={}; afpl={}
for f in glob.glob('af/*.pdb'):
    acc=os.path.basename(f)[:-4]; s={}; pl={}
    for line in open(f):
        if line.startswith('ATOM') and line[12:16].strip()=='CA':
            r=int(line[22:26]); s[r]=aa3.get(line[17:20].strip(),'X'); pl[r]=float(line[60:66])
    resis=sorted(s); afseq[acc]=(resis[0],''.join(s[r] for r in resis)); afpl[acc]=pl

aligner=Align.PairwiseAligner(); aligner.mode='local'; aligner.open_gap_score=-10; aligner.extend_gap_score=-1; aligner.match_score=2; aligner.mismatch_score=-1

# per-assay position -> AF resi map
plddt_col=np.full(len(df),np.nan)
diag=[]
for assay,g in df.groupby('assay'):
    acc=g.acc.iloc[0]
    if acc not in afseq: diag.append((assay,acc,'no_AF',np.nan)); continue
    tseq=ref.loc[assay,'target_seq']; start,aseq=afseq[acc]
    idx=aseq.find(tseq)
    posmap={}
    if idx>=0:
        base=start+idx
        for p in range(1,len(tseq)+1): posmap[p]=base+(p-1)
        method='sub'
    else:
        aln=aligner.align(tseq,aseq)[0]
        # aligned blocks: aln.aligned = ([ (t0,t1),...],[(a0,a1),...])
        tb,ab=aln.aligned
        for (t0,t1),(a0,a1) in zip(tb,ab):
            for k in range(t1-t0):
                posmap[t0+k+1]=start+a0+k   # target pos (1-based) -> af resi
        method='aln'
    # fill
    idxs=g.index; pls=[]; okc=0; tot=0
    for i,p,w in zip(idxs,g.position,g.wt):
        r=posmap.get(p)
        if r is not None:
            plddt_col[i]=afpl[acc].get(r,np.nan)
            afaa=aseq[r-start] if 0<=r-start<len(aseq) else None
            tot+=1; okc+= (afaa==w)
    diag.append((assay,acc,method,okc/tot if tot else np.nan))
df['plddt']=plddt_col
D=pd.DataFrame(diag,columns=['assay','acc','method','wtmatch'])
print('assays with wtmatch<0.9:', (D.wtmatch<0.9).sum())
print(D[D.wtmatch<0.9].to_string(index=False))
print('\noverall plddt non-null:',df.plddt.notna().sum(),'/',len(df))
# per-variant WT match overall
meta=pd.read_csv('assay_meta.csv')[['assay','coarse_selection_type']]
df=df.merge(meta,on='assay',how='left')
df.to_parquet('master_plddt.parquet')
D.to_csv('plddt_mapping_diag.csv',index=False)
print('saved corrected master_plddt.parquet')

# Original notebook cell 14
master=pd.read_parquet(OUT/'master_plddt.parquet')
expected=pd.read_parquet(INPUT/'master_plddt.parquet')
pd.testing.assert_frame_equal(master,expected,check_exact=False,rtol=1e-12,atol=1e-12)
master['pssm_recomputed']=master.pssm
score_delta=(master.pssm-expected.pssm).abs()
print('Largest freshly rebuilt PSSM score difference:',score_delta.max())
# Preserve the original individual scores for exact historical rank ties.
# The independently regenerated score is retained and assessed separately below.
master['pssm']=expected.pssm
master['coverage_split']=np.where(master.acc.isin(discovery),'discovery','confirmation')
plddt_accs=np.array(sorted(master.dropna(subset=['plddt']).acc.unique()))
np.random.default_rng(7).shuffle(plddt_accs)
plddt_discovery=set(plddt_accs[:len(plddt_accs)//2]);plddt_confirmation=set(plddt_accs[len(plddt_accs)//2:])
master['split']=np.where(master.acc.isin(plddt_discovery),'discovery',np.where(master.acc.isin(plddt_confirmation),'confirmation','not_in_pLDDT'))
master[['assay','acc','position','mutant','coverage_split','split']].to_csv(OUT/'axis_specific_sample_splits.csv.gz',index=False)
master.to_parquet(OUT/'variant_scores_and_features.parquet',index=False)
print('All mapped pLDDT values reproduced:',int(master.plddt.notna().sum()),'of',len(master),'variants.')
display(master[master.plddt.isna()].groupby('acc').size().rename('excluded_from_pLDDT'))

# Original notebook cell 16
def sp(g,col):
    if len(g)>=50 and g.DMS_score.nunique()>5 and g[col].nunique()>2:
        return abs(spearmanr(g[col],g.DMS_score).statistic)
    return np.nan

labels=['P1_disordered(<50)','P2_low(50-70)','P3_confident(70-90)','P4_ordered(>=90)']
df=master.dropna(subset=['plddt']).copy()
df['level']=np.select([df.plddt<50,df.plddt<70,df.plddt<90],[labels[0],labels[1],labels[2]],default=labels[3])
rows=[]
for (assay,level),g in df.groupby(['assay','level']):
    rows.append(dict(assay=assay,level=level,n=len(g),plddt=g.plddt.mean(),
                     AM=sp(g,'am_pathogenicity'),BLOSUM=sp(g,'blosum'),PSSM=sp(g,'pssm'),
                     PSSM_recomputed=sp(g,'pssm_recomputed'),
                     dms_sd=g.DMS_score.std(),acc=g.acc.iloc[0],split=g.split.iloc[0],selection=g.coarse_selection_type.iloc[0]))
scores=pd.DataFrame(rows)
old=pd.read_csv(INPUT/'FINAL_SPC_plddt_by_assay_level.csv')
joined=scores.merge(old,on=['assay','level'],suffixes=('_new','_old'),validate='one_to_one')
assert len(joined)==len(scores)==len(old)
for c in ['n','plddt','AM','BLOSUM','PSSM']:
    np.testing.assert_allclose(joined[c+'_new'],joined[c+'_old'],rtol=1e-10,atol=1e-10,equal_nan=True)
df[['assay','acc','position','mutant','split','plddt','level','am_pathogenicity','DMS_score','blosum','pssm']].to_csv(OUT/'plddt_sample_measurements.csv.gz',index=False)
scores.to_csv(OUT/'plddt_assay_scores.csv',index=False)
print(len(scores),'assay/level cells match the original table.')
print('Fresh-PSSM versus historical-PSSM maximum cell discrepancy:',(scores.PSSM_recomputed-scores.PSSM).abs().max())

# Original notebook cell 18
curve=[]
for method in ['AM','BLOSUM','PSSM']:
    for level in labels:
        values=scores.loc[scores.level==level,method].dropna().to_numpy()
        rng=np.random.default_rng(11)
        boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        curve.append(dict(method=method,level=level,mean=values.mean(),low=lo,high=hi,n_assays=len(values)))
curve=pd.DataFrame(curve);curve.to_csv(OUT/'plddt_curve.csv',index=False)
fig,ax=plt.subplots(figsize=(7,4))
for method,g in curve.groupby('method',sort=False):
    ax.errorbar(np.arange(4),g['mean'],yerr=[g['mean']-g.low,g.high-g['mean']],marker='o',capsize=3,label=method)
ax.set(xticks=range(4),xticklabels=['<50','50–<70','70–<90','≥90'],xlabel='Residue pLDDT',ylabel='Mean within-assay |Spearman|')
ax.legend(frameon=False);plt.tight_layout();plt.show()
display(curve.round(6))

# Original notebook cell 20
def paired(t,label):
    piv={m:t.pivot(index='assay',columns='level',values=m) for m in ['AM','BLOSUM','PSSM']}
    common=set.intersection(*[set(p[[labels[0],labels[3]]].dropna().index) for p in piv.values()])
    ids=sorted(common);delta={m:(p.loc[ids,labels[3]]-p.loc[ids,labels[0]]).to_numpy() for m,p in piv.items()}
    result=[]
    for name,v in list(delta.items())+[('AM minus BLOSUM',delta['AM']-delta['BLOSUM']),('AM minus PSSM',delta['AM']-delta['PSSM'])]:
        rng=np.random.default_rng(11);b=v[rng.integers(0,len(v),(2000,len(v)))].mean(axis=1)
        lo,hi=np.quantile(b,[.025,.975]);result.append(dict(subset=label,comparison=name,mean=v.mean(),low=lo,high=hi,n=len(v),positive=int((v>0).sum()),wilcoxon_p=wilcoxon(v).pvalue))
    return result
effects=[]
for label,t in [('All',scores),('Discovery',scores[scores.split=='discovery']),('Confirmation',scores[scores.split=='confirmation']),('Excluding stability',scores[scores.selection!='Stability'])]:
    effects+=paired(t,label)
effects=pd.DataFrame(effects);effects.to_csv(OUT/'paired_effects.csv',index=False);display(effects.round(6))
reported=json.loads((INPUT/'RESULTS_SUMMARY.json').read_text())
actual=effects.query("subset=='All' and comparison=='AM'").iloc[0]
assert actual['n']==27 and abs(actual['mean']-reported['plddt_axis']['AM_drop_ordered_minus_disordered'][0])<.0006
for name,key in [('Discovery','discovery_AM_drop'),('Confirmation','confirmation_AM_drop')]:
    actual=effects.query('subset==@name and comparison=="AM"')['mean'].iloc[0]
    assert abs(actual-reported['held_out'][key])<.0006
print('Pooled and split-specific point estimates match the original report at its stated precision.')

# Original notebook cell 22
def horse(binby):
    rows=[]
    for assay,g in df.groupby('assay'):
        g=g.copy()
        try:g['cell']=pd.qcut(g[binby],6,labels=False,duplicates='drop')
        except ValueError:continue
        for cell,gg in g.groupby('cell'):
            rows.append(dict(assay=assay,am=sp(gg,'am_pathogenicity'),cov=gg.coverage.mean(),plddt=gg.plddt.mean(),ldms=np.log(gg.DMS_score.std()+1e-9)))
    r=pd.DataFrame(rows).dropna();r=r[np.isfinite(r.ldms)].copy()
    columns=['cov','plddt','ldms']
    for c in ['am']+columns:r[c+'_c']=r[c]-r.groupby('assay')[c].transform('mean')
    X=np.column_stack([np.ones(len(r))]+[r[c+'_c'] for c in columns]);y=r.am_c.to_numpy()
    beta=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@beta
    se=np.sqrt(np.diag((res@res)/(len(r)-X.shape[1])*np.linalg.inv(X.T@X)))
    return [dict(binning=binby,covariate=c,beta=beta[i+1],t=beta[i+1]/se[i+1],cells=len(r)) for i,c in enumerate(columns)]
control=pd.DataFrame(horse('coverage')+horse('plddt'));control.to_csv(OUT/'adjusted_control.csv',index=False);display(control)

# Original notebook cell 24
coverage=master.copy();edges=[-1e-9,.10,.50,.80,.95,.99,1.01]
covlabels=['<.10','.10–.50','.50–.80','.80–.95','.95–.99','≥.99']
coverage['level']=pd.cut(coverage.coverage,edges,labels=covlabels)
rows=[]
for (assay,level),g in coverage.groupby(['assay','level'],observed=True):
    rows.append(dict(assay=assay,level=level,n=len(g),AM=sp(g,'am_pathogenicity'),BLOSUM=sp(g,'blosum'),PSSM=sp(g,'pssm')))
coverage_scores=pd.DataFrame(rows);coverage_scores.to_csv(OUT/'coverage_assay_scores.csv',index=False)
coverage[['assay','acc','position','mutant','coverage','level','am_pathogenicity','DMS_score','blosum','pssm']].to_csv(OUT/'coverage_sample_measurements.csv.gz',index=False)
coverage_curve=[]
fig,ax=plt.subplots(figsize=(7,4))
for method in ['AM','BLOSUM','PSSM']:
    means=[];lows=[];highs=[]
    for level in covlabels:
        values=coverage_scores.loc[coverage_scores.level==level,method].dropna().to_numpy()
        rng=np.random.default_rng(11);bs=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
        lo,hi=np.quantile(bs,[.025,.975]);mu=values.mean()
        means.append(mu);lows.append(lo);highs.append(hi)
        coverage_curve.append(dict(method=method,level=level,mean=mu,low=lo,high=hi,n_assays=len(values)))
    ax.errorbar(range(6),means,yerr=[np.array(means)-lows,np.array(highs)-means],marker='o',capsize=3,label=method)
pd.DataFrame(coverage_curve).to_csv(OUT/'coverage_curve.csv',index=False)
ax.set(xticks=range(6),xticklabels=covlabels,xlabel='Alignment coverage',ylabel='Mean within-assay |Spearman|');ax.legend(frameon=False);plt.tight_layout();plt.show()
meta=pd.read_csv(INPUT/'assay_meta.csv')
rows=[]
for assay,g in master.groupby('assay'):
    m=meta[meta.assay==assay].iloc[0]
    rows.append(dict(assay=assay,agreement=sp(g,'am_pathogenicity'),depth=m.MSA_Neff_L,selection=m.coarse_selection_type))
depth=pd.DataFrame(rows);depth.to_csv(OUT/'whole_protein_depth_assays.csv',index=False)
depth_stats=[]
for label,g in [('All',depth),('Excluding stability',depth[depth.selection!='Stability'])]:
    rho,p=spearmanr(np.log1p(g.depth),g.agreement);depth_stats.append(dict(group=label,n=len(g),rho=rho,p=p))
display(pd.DataFrame(depth_stats));pd.DataFrame(depth_stats).to_csv(OUT/'whole_protein_depth_control.csv',index=False)

# Original notebook cell 26
os.chdir(ROOT)
paths=[p for p in OUT.glob('*') if p.is_file()]
manifest=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(paths)]
(OUT/'OUTPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print(len(manifest),'recomputed output tables saved and hashed.')
