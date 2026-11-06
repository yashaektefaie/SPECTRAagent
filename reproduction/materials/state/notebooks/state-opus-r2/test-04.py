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
    import pandas as pd, numpy as np
    from scipy.stats import spearmanr
    df = pd.read_csv('data/predictions.csv', low_memory=False)
    r = df[df.dataset_family=='Replogle-Nadig'].copy()

    # cross-context coverage: for each gene, how many of the 4 contexts it appears in (overall)
    gene_ctx = r.groupby('perturbation_name')['context'].nunique()
    r['gene_ncontexts']=r['perturbation_name'].map(gene_ctx)

    # For a zeroshot row in context c, 'other-context coverage' = ncontexts - 1 (present elsewhere)
    r['other_ctx_cov']=r['gene_ncontexts']-1

    print('=== Replogle zeroshot: overlap by other_ctx_cov (0..3) ===')
    z=r[r.protocol=='zeroshot']
    for rep in ['HVG','SE']:
        zz=z[z.representation==rep]
        print(f'-- {rep} --')
        print(zz.groupby('other_ctx_cov')['overlap_at_N'].agg(['mean','median','count']).round(3))
        rho,p=spearmanr(zz['other_ctx_cov'],zz['overlap_at_N'])
        print(f'   spearman={rho:+.3f} p={p:.1e}')
finally:os.chdir(old)

# Original notebook cell 7
tmp.cleanup()
