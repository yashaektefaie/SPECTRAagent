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
    import numpy as np, pandas as pd
    from scipy.stats import spearmanr, mannwhitneyu
    df=pd.read_csv('data/predictions.csv',low_memory=False)
    r=df[df.dataset_family=='Replogle-Nadig'].copy()
    # Drop the later-added level column before recreating this earlier analysis step.
    feat=pd.read_csv('spectra_run/artifacts/replogle_go_features.csv').drop(columns=['novlevel'],errors='ignore')
    d=r.merge(feat,on='perturbation_name',how='inner')

    # Freeze levels by go_meantop5 quartiles (gene-level, context-independent)
    gu=feat.copy()
    gu['novlevel']=pd.qcut(gu['go_meantop5'],4,labels=['L1_novel','L2','L3','L4_covered'])
    d=d.merge(gu[['perturbation_name','novlevel']],on='perturbation_name',how='left')

    print('=== SPC: overlap_at_N by novelty level (Replogle zeroshot) ===')
    print('(L1_novel = sparsest functional neighborhood/farthest from training ... L4_covered = densest)')
    for rep in ['HVG','SE']:
        z=d[(d.protocol=='zeroshot')&(d.representation==rep)]
        tab=z.groupby('novlevel',observed=True)['overlap_at_N'].agg(['mean','median','count'])
        print(f'\n-- {rep} zeroshot --'); print(tab.round(4))
        lo=z[z.novlevel=='L1_novel']['overlap_at_N']; hi=z[z.novlevel=='L4_covered']['overlap_at_N']
        print(f'   L4/L1 mean ratio = {hi.mean()/lo.mean():.2f}x ; MWU p={mannwhitneyu(hi,lo).pvalue:.1e}')

    print('\n\n=== HELD-OUT CONTEXT CONFIRMATION (frozen feature) ===')
    print('Discovery contexts {k562,rpe1} -> Confirm on held-out {jurkat,hepg2}')
    disc=['k562','rpe1']; conf=['jurkat','hepg2']
    for rep in ['HVG','SE']:
        for name,ctxs in [('DISCOVERY',disc),('CONFIRM(held-out)',conf)]:
            z=d[(d.protocol=='zeroshot')&(d.representation==rep)&(d.context.isin(ctxs))]
            rho,p=spearmanr(z['go_meantop5'],z['overlap_at_N'])
            # level means
            m=z.groupby('novlevel',observed=True)['overlap_at_N'].mean()
            ratio = m.get('L4_covered',np.nan)/m.get('L1_novel',np.nan)
            print(f'  {rep} {name:18s}: rho={rho:+.3f} p={p:.1e} n={len(z)}  L4/L1={ratio:.2f}x')
finally:os.chdir(old)

# Original notebook cell 7
old=Path.cwd();os.chdir(run)
try:
    import json, numpy as np, pandas as pd
    from scipy.stats import spearmanr
    df=pd.read_csv('data/predictions.csv',low_memory=False)

    # ---------- Independent similarity validation on Replogle ----------
    go=json.load(open('spectra_run/artifacts/go_terms.json'))
    pw=json.load(open('spectra_run/artifacts/replogle_pathways.json'))
    feat=pd.read_csv('spectra_run/artifacts/replogle_go_features.csv')
    genes=list(feat['perturbation_name'])

    def density(setmap, arr, topk=5):
        S={g:set(setmap.get(g,[])) for g in arr}
        out={}
        for g in arr:
            sg=S[g]
            if not sg: out[g]=np.nan; continue
            sims=[]
            for h in arr:
                if h==g: continue
                sh=S[h]
                if not sh: continue
                inter=len(sg&sh)
                sims.append(inter/(len(sg)+len(sh)-inter) if inter else 0.0)
            sims=np.sort(sims)[::-1]
            out[g]=np.mean(sims[:topk]) if len(sims)>=1 else np.nan
        return out

    # GO-CC density (independent from BP)
    cc={g:go.get(g,{}).get('CC',[]) for g in genes}
    den_cc=density(cc, genes)
    # Reactome+KEGG combined co-membership density
    rk={g: list(pw.get(g,{}).get('reactome',[]))+list(pw.get(g,{}).get('kegg',[])) for g in genes}
    den_rk=density(rk, genes)

    feat['den_cc']=feat['perturbation_name'].map(den_cc)
    feat['den_rk']=feat['perturbation_name'].map(den_rk)
    feat['novlevel']=pd.qcut(feat['go_meantop5'],4,labels=['L1_novel','L2','L3','L4_covered'])

    print('=== VALIDATE: independent similarity by frozen GO-BP novelty level ===')
    print('(should INCREASE L1->L4 if levels are ordered by genuine functional similarity)')
    print(feat.groupby('novlevel',observed=True)[['go_meantop5','den_cc','den_rk']].mean().round(3))
    print('\n corr(go_meantop5, den_cc) =',round(feat[['go_meantop5','den_cc']].corr(method='spearman').iloc[0,1],3),
          ' (n=%d)'%feat['den_cc'].notna().sum())
    print(' corr(go_meantop5, den_rk) =',round(feat[['go_meantop5','den_rk']].corr(method='spearman').iloc[0,1],3),
          ' (n=%d)'%feat['den_rk'].notna().sum())

    # do the INDEPENDENT densities predict overlap too?
    r=df[df.dataset_family=='Replogle-Nadig']
    d=r[r.protocol=='zeroshot'].merge(feat,on='perturbation_name',how='inner')
    print('\n=== independent-source density vs overlap (Replogle zeroshot) ===')
    for rep in ['HVG','SE']:
        z=d[d.representation==rep]
        for f in ['den_cc','den_rk']:
            zz=z.dropna(subset=[f])
            rho,p=spearmanr(zz[f],zz['overlap_at_N'])
            print(f'  {rep} {f}: rho={rho:+.3f} p={p:.1e} n={len(zz)}')
    feat.to_csv('spectra_run/artifacts/replogle_go_features.csv',index=False)
finally:os.chdir(old)

# Original notebook cell 8
data=raw[raw.dataset_family=='Replogle-Nadig'].merge(expected,on='perturbation_name',validate='many_to_one')
data=data[data.protocol=='zeroshot'].copy();data['group']=data.representation
draw(data,'novlevel','overlap_at_N','group')

# Original notebook cell 9
tmp.cleanup()
