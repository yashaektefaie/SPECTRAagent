# Original notebook cell 2
from pathlib import Path
import sys,json,hashlib,io,os,tempfile,shutil,runpy,contextlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from IPython.display import display,Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
root=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'STATE_INPUTS.json').is_file()),None)
assert root is not None,'Extract the complete STATE bundle first.'
manifest=json.loads((root/'STATE_INPUTS.json').read_text())
P=root/'assets'/'state-opus-r2'
for relative,meta in manifest.items():
    if relative.startswith('state-opus-r2'+'/'):
        assert hashlib.sha256((root/'assets'/relative).read_bytes()).hexdigest()==meta['sha256'],relative
print('Verified all run input and code hashes.')

# Original notebook cell 3
def draw(data,x,y,group):
    rng=np.random.default_rng(7);rows=[]
    for keys,g in data.groupby([group,x],observed=True):
        v=pd.to_numeric(g[y],errors='coerce').dropna().values
        if not len(v):continue
        bs=np.array([rng.choice(v,len(v),True).mean() for _ in range(1000)])
        rows.append({group:keys[0],x:keys[1],'n':len(v),'mean':v.mean(),'lo':np.quantile(bs,.025),'hi':np.quantile(bs,.975)})
    t=pd.DataFrame(rows);display(t)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');levels=sorted(t[x].unique(),key=str)
    for name,g in t.groupby(group,observed=True):
        g=g.assign(xpos=g[x].map({v:i for i,v in enumerate(levels)})).sort_values('xpos')
        ax.errorbar(g.xpos,g['mean'],yerr=np.stack([g['mean']-g.lo,g.hi-g['mean']]),marker='o',capsize=2,label=str(name))
    ax.set_xticks(range(len(levels)),levels,rotation=25);ax.set(xlabel=x,ylabel=y);ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120);plt.close(fig);display(Image(data=buf.getvalue()))

# Original notebook cell 4
raw=pd.read_csv(P/'data/predictions.csv',low_memory=False)
feat=pd.read_csv(P/'spectra_run/artifacts/replogle_go_features.csv')
print('Rows:',len(raw),'annotated gene identities:',len(feat))
tmp=tempfile.TemporaryDirectory();run=Path(tmp.name);shutil.copytree(P/'data',run/'data');shutil.copytree(P/'spectra_run',run/'spectra_run')

# Original notebook cell 5
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
go=json.load(open(P/'spectra_run/artifacts/go_terms.json'))
df=pd.read_csv(P/'data/predictions.csv',low_memory=False)
r=df[df.dataset_family=='Replogle-Nadig'].copy()
genes=sorted(r['perturbation_name'].unique())

BP={g:set(go.get(g,{}).get('BP',[])) for g in genes}
nbp={g:len(BP[g]) for g in genes}
have=[g for g in genes if nbp[g]>0]
print('genes with BP terms:',len(have))

# Functional neighborhood density via GO-BP Jaccard.
# For each gene, mean of top-5 Jaccard to OTHER genes = functional coverage.
# Compute efficiently via inverted index -> but Jaccard needs pairwise. Do it with sets (2023^2/2 ~ 2M, ok)
import itertools
arr=have
# Precompute for speed: represent as frozensets
S={g:BP[g] for g in arr}
# To bound cost, compute for each gene its top-k jaccard by iterating (2M ops)
top5=[]; nnj=[]
den_thr=[]  # count neighbors with jaccard>=0.2
for i,g in enumerate(arr):
    sg=S[g]; lg=len(sg)
    sims=[]
    for h in arr:
        if h==g: continue
        sh=S[h]
        inter=len(sg & sh)
        if inter==0:
            sims.append(0.0); continue
        j=inter/(lg+len(sh)-inter)
        sims.append(j)
    sims=np.array(sims); sims.sort()
    top5.append(sims[-5:].mean()); nnj.append(sims[-1]); den_thr.append((sims>=0.2).sum())
feat=pd.DataFrame({'perturbation_name':arr,'go_nbp':[nbp[g] for g in arr],
                   'go_meantop5':top5,'go_nn':nnj,'go_ndense':den_thr})

expected=pd.read_csv(P/'spectra_run/artifacts/replogle_go_features.csv')
a=feat.set_index('perturbation_name').sort_index();b=expected.set_index('perturbation_name').sort_index()
assert a.index.equals(b.index)
for col in ['go_meantop5','go_nn','go_nbp','go_ndense']:np.testing.assert_allclose(a[col],b[col],atol=1e-12)
print('GO-BP neighborhood feature exactly reproduced from retained GO membership lists.')

# Original notebook cell 6
old=Path.cwd();os.chdir(run)
try:
    import json, numpy as np, pandas as pd
    from scipy.stats import spearmanr
    df=pd.read_csv('data/predictions.csv',low_memory=False)
    go=json.load(open('spectra_run/artifacts/tahoe_target_go.json'))
    t=df[df.dataset_family=='Tahoe-100M'].copy()
    du=t.drop_duplicates('perturbation_name').copy()

    # target-gene functional density among the Tahoe target-gene set (training proxy)
    tgenes=sorted(go.keys())
    S={g:set(go.get(g,[])) for g in tgenes}
    def gdens(g,topk=5):
        sg=S.get(g,set())
        if not sg: return np.nan
        sims=[]
        for h in tgenes:
            if h==g: continue
            sh=S[h]
            if not sh: continue
            inter=len(sg&sh); sims.append(inter/(len(sg)+len(sh)-inter) if inter else 0.0)
        sims=np.sort(sims)[::-1]; return np.mean(sims[:topk])
    gd={g:gdens(g) for g in tgenes}

    def drug_density(targets):
        if pd.isna(targets): return np.nan
        vals=[gd.get(x.strip(),np.nan) for x in str(targets).split(',')]
        vals=[v for v in vals if v==v]
        return np.max(vals) if vals else np.nan   # best-covered target
    du['tgt_funcdens']=du['drugmeta_targets'].apply(drug_density)
    print('drugs with target functional density:', du['tgt_funcdens'].notna().sum(),'/',len(du))

    z=t.merge(du[['perturbation_name','tgt_funcdens']],on='perturbation_name',how='left')
    print('\n=== TAHOE CROSS-FAMILY CONFIRMATION: drug target-gene functional density vs overlap ===')
    for proto in ['zeroshot','fewshot']:
        for rep in ['HVG','SE']:
            zz=z[(z.protocol==proto)&(z.representation==rep)].dropna(subset=['tgt_funcdens'])
            rho,p=spearmanr(zz['tgt_funcdens'],zz['overlap_at_N'])
            print(f'  {proto:9s} {rep}: rho={rho:+.3f} p={p:.1e} n={len(zz)}')
    # level curve zeroshot
    print('\n=== Tahoe zeroshot SPC by target-density quartile ===')
    zt=z[z.protocol=='zeroshot'].dropna(subset=['tgt_funcdens']).copy()
    zt['lvl']=pd.qcut(zt['tgt_funcdens'],4,labels=['L1_novel','L2','L3','L4_covered'],duplicates='drop')
    for rep in ['HVG','SE']:
        print(f'-- {rep} --')
        print(zt[zt.representation==rep].groupby('lvl',observed=True)['overlap_at_N'].agg(['mean','count']).round(4))
finally:os.chdir(old)

# Original notebook cell 7
old=Path.cwd();os.chdir(run)
try:
    import json, numpy as np, pandas as pd
    from scipy.stats import spearmanr
    df=pd.read_csv('data/predictions.csv',low_memory=False)
    M=json.load(open('spectra_run/artifacts/parse_ligand_map.json'))
    go=json.load(open('spectra_run/artifacts/parse_gene_go.json'))
    p=df[df.dataset_family=='Parse-PBMC'].copy()
    p['gene']=p['perturbation_name'].map(M)

    # functional density among the Parse ligand-gene set (training proxy for this family)
    genes=sorted(set(M.values()))
    S={g:set(go.get(g,[])) for g in genes}
    def dens(g,topk=5):
        sg=S.get(g,set())
        if not sg: return np.nan
        sims=[]
        for h in genes:
            if h==g: continue
            sh=S[h]
            if not sh: continue
            inter=len(sg&sh); sims.append(inter/(len(sg)+len(sh)-inter) if inter else 0.0)
        sims=np.sort(sims)[::-1]; return np.mean(sims[:topk])
    gd={g:dens(g) for g in genes}
    p['funcdens']=p['gene'].map(gd)
    print('Parse rows with funcdens:', p['funcdens'].notna().sum(),'/',len(p))

    print('\n=== PARSE-PBMC FRESH-FAMILY CONFIRMATION: cytokine functional density vs overlap ===')
    for proto in ['zeroshot','fewshot']:
        for rep in ['HVG','SE']:
            z=p[(p.protocol==proto)&(p.representation==rep)].dropna(subset=['funcdens'])
            if len(z)<20: continue
            rho,pv=spearmanr(z['funcdens'],z['overlap_at_N'])
            print(f'  {proto:9s} {rep}: rho={rho:+.3f} p={pv:.1e} n={len(z)}')
    # pooled zeroshot both reps
    z=p[p.protocol=='zeroshot'].dropna(subset=['funcdens'])
    rho,pv=spearmanr(z['funcdens'],z['overlap_at_N'])
    print(f'  POOLED zeroshot (both reps): rho={rho:+.3f} p={pv:.1e} n={len(z)}')
    # level curve
    z=z.copy(); z['lvl']=pd.qcut(z['funcdens'],3,labels=['L1_novel','L2','L3_covered'],duplicates='drop')
    print('\n Parse zeroshot SPC (tertiles, pooled reps):')
    print(z.groupby('lvl',observed=True)['overlap_at_N'].agg(['mean','count']).round(4))
finally:os.chdir(old)

# Original notebook cell 8
tmp.cleanup()
