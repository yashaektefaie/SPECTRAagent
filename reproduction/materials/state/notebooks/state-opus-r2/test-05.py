# Original notebook cell 3
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

# Original notebook cell 4
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

# Original notebook cell 5
raw=pd.read_csv(P/'data/predictions.csv',low_memory=False)
feat=pd.read_csv(P/'spectra_run/artifacts/replogle_go_features.csv')
print('Rows:',len(raw),'annotated gene identities:',len(feat))
tmp=tempfile.TemporaryDirectory();run=Path(tmp.name);shutil.copytree(P/'data',run/'data');shutil.copytree(P/'spectra_run',run/'spectra_run')

# Original notebook cell 6
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

# Original notebook cell 7
old=Path.cwd();os.chdir(run)
try:
    import json, numpy as np, pandas as pd
    from scipy.stats import spearmanr
    df=pd.read_csv('data/predictions.csv',low_memory=False)
    r=df[df.dataset_family=='Replogle-Nadig'].copy()
    feat=pd.read_csv('spectra_run/artifacts/replogle_go_features.csv')
    gene_ctx=r.groupby('perturbation_name')['context'].nunique();
    r['other_ctx_cov']=r['perturbation_name'].map(gene_ctx)-1
    d=r.merge(feat,on='perturbation_name',how='inner')

    print('=== 1) fewshot vs zeroshot: density effect (go_ndense) — train-novelty should SHRINK in fewshot ===')
    for proto in ['zeroshot','fewshot']:
        for rep in ['HVG','SE']:
            zz=d[(d.protocol==proto)&(d.representation==rep)]
            rho,p=spearmanr(zz['go_ndense'],zz['overlap_at_N'])
            print(f'  {proto:9s} {rep}: rho(go_ndense,overlap)={rho:+.3f} p={p:.1e} n={len(zz)}')
    print()
    print('=== 2) control for REAL effect size (de_nsig_counts_real = # sig DE genes in real data) ===')
    # stratify by real-effect quartile within zeroshot, check density effect persists
    z=d[d.protocol=='zeroshot'].copy()
    z=z.dropna(subset=['de_nsig_counts_real'])
    z['eff_q']=pd.qcut(z['de_nsig_counts_real'],4,labels=['Q1low','Q2','Q3','Q4high'],duplicates='drop')
    for rep in ['HVG','SE']:
        print(f'-- {rep} --')
        zz=z[z.representation==rep]
        # overall corr of effect size itself
        rho_eff,_=spearmanr(zz['de_nsig_counts_real'],zz['overlap_at_N'])
        rho_den,_=spearmanr(zz['go_ndense'],zz['overlap_at_N'])
        print(f'   overall: rho(realeffect,overlap)={rho_eff:+.3f}  rho(density,overlap)={rho_den:+.3f}')
        for q,g in zz.groupby('eff_q',observed=True):
            rho,p=spearmanr(g['go_ndense'],g['overlap_at_N'])
            print(f'   within {q}: rho(density,overlap)={rho:+.3f} p={p:.1e} n={len(g)}')
    print()
    print('=== 3) is density just cross-context coverage? corr(go_ndense, other_ctx_cov) ===')
    gu=d.drop_duplicates('perturbation_name')
    rho,p=spearmanr(gu['go_ndense'],gu['other_ctx_cov'])
    print(f'  gene-level rho(go_ndense, other_ctx_cov)={rho:+.3f}')
    # density effect within genes present in ALL 4 contexts (other_ctx_cov==3)
    z3=z[z.other_ctx_cov==3]
    for rep in ['HVG','SE']:
        zz=z3[z3.representation==rep]
        rho,p=spearmanr(zz['go_ndense'],zz['overlap_at_N'])
        print(f'  within full-coverage genes (ctx_cov=3) {rep}: rho(density,overlap)={rho:+.3f} p={p:.1e} n={len(zz)}')
finally:os.chdir(old)

# Original notebook cell 8
old=Path.cwd();os.chdir(run)
try:
    import numpy as np, pandas as pd
    from scipy.stats import spearmanr, rankdata
    df=pd.read_csv('data/predictions.csv',low_memory=False)
    r=df[df.dataset_family=='Replogle-Nadig'].copy()
    feat=pd.read_csv('spectra_run/artifacts/replogle_go_features.csv')
    gene_ctx=r.groupby('perturbation_name')['context'].nunique()
    r['other_ctx_cov']=r['perturbation_name'].map(gene_ctx)-1
    d=r.merge(feat,on='perturbation_name',how='inner')
    z=d[d.protocol=='zeroshot'].copy()

    def partial_spearman(a,b,controls,data):
        # residualize ranks of a and b on ranks of controls via linear regression
        import numpy as np
        Y=rankdata(data[a]); X=rankdata(data[b])
        C=np.column_stack([rankdata(data[c]) for c in controls]+[np.ones(len(data))])
        def resid(v):
            beta,_,_,_=np.linalg.lstsq(C,v,rcond=None); return v-C@beta
        ry,rx=resid(Y),resid(X)
        from scipy.stats import pearsonr
        rr,pp=pearsonr(rx,ry); return rr,pp

    print('=== 1) FIXED-N metrics (removes N-varies-with-effect-size confound) ===')
    print('   density=go_meantop5 ; expect + at fixed N')
    for rep in ['HVG','SE']:
        zz=z[z.representation==rep]
        for mcol in ['overlap_at_50','overlap_at_100','overlap_at_200','overlap_at_500','precision_at_100']:
            rho,p=spearmanr(zz['go_meantop5'],zz[mcol])
            print(f'   {rep} {mcol:16s}: rho={rho:+.3f} p={p:.1e}')
        print()

    print('=== 2) PARTIAL correlation: density vs overlap_at_N controlling for real effect size (nsig_real) ===')
    for rep in ['HVG','SE']:
        zz=z[z.representation==rep].dropna(subset=['de_nsig_counts_real','go_meantop5'])
        rr,pp=partial_spearman('overlap_at_N','go_meantop5',['de_nsig_counts_real'],zz)
        print(f'   {rep}: partial rho(density,overlap | nsig_real)={rr:+.3f} p={pp:.1e} n={len(zz)}')

    print('\n=== 3) PARTIAL controlling for nsig_real + cross-context coverage + context ===')
    z['ctx_code']=z['context'].astype('category').cat.codes
    for rep in ['HVG','SE']:
        zz=z[z.representation==rep].dropna(subset=['de_nsig_counts_real','go_meantop5'])
        rr,pp=partial_spearman('overlap_at_N','go_meantop5',['de_nsig_counts_real','other_ctx_cov','ctx_code'],zz)
        print(f'   {rep}: partial rho(density,overlap | nsig_real,ctxcov,context)={rr:+.3f} p={pp:.1e}')

    print('\n=== 4) does density predict real effect size itself? (mechanism check) ===')
    gu=d.drop_duplicates('perturbation_name')
    # gene-level mean nsig
    gmean=z.groupby('perturbation_name').agg(nsig=('de_nsig_counts_real','mean'),dens=('go_meantop5','first'))
    rho,p=spearmanr(gmean['dens'],gmean['nsig'])
    print(f'   gene-level rho(density, mean nsig_real)={rho:+.3f} p={p:.1e}  (dense genes DO have larger real effects)')
finally:os.chdir(old)

# Original notebook cell 9
old=Path.cwd();os.chdir(run)
try:
    import json, numpy as np, pandas as pd
    from scipy.stats import spearmanr
    df=pd.read_csv('data/predictions.csv',low_memory=False)
    r=df[df.dataset_family=='Replogle-Nadig'].copy()
    feat=pd.read_csv('spectra_run/artifacts/replogle_go_features.csv')
    go=json.load(open('spectra_run/artifacts/go_terms.json'))
    d=r.merge(feat,on='perturbation_name',how='inner')
    z=d[d.protocol=='zeroshot']

    print('=== A) Within-context associations (not new independent cohorts) ===')
    for rep in ['HVG','SE']:
        for ctx in sorted(z['context'].unique()):
            zz=z[(z.representation==rep)&(z.context==ctx)]
            rho,p=spearmanr(zz['go_meantop5'],zz['overlap_at_N'])
            print(f'   {rep} {ctx:7s}: rho={rho:+.3f} p={p:.1e} n={len(zz)}')
        print()

    print('=== B) MODULE ROBUSTNESS: exclude big functional modules, re-test ===')
    # tag genes whose top GO includes ribosome/translation/proteasome/splic/mito
    big_terms_keywords=None
    # use CC/BP to flag ribosomal & spliceosomal etc via term presence is hard; instead exclude top-density decile
    for rep in ['HVG','SE']:
        zz=z[z.representation==rep].copy()
        # exclude the densest 10% (likely core complexes) and re-test on the remaining 90%
        thr=feat['go_meantop5'].quantile(0.90)
        sub=zz[zz['go_meantop5']<thr]
        rho,p=spearmanr(sub['go_meantop5'],sub['overlap_at_N'])
        print(f'   {rep} excl top-decile density: rho={rho:+.3f} p={p:.1e} n={len(sub)}')
        # also exclude bottom decile (orphans) to test middle range
        thr2=feat['go_meantop5'].quantile(0.10)
        sub2=zz[(zz['go_meantop5']>thr2)&(zz['go_meantop5']<thr)]
        rho2,p2=spearmanr(sub2['go_meantop5'],sub2['overlap_at_N'])
        print(f'   {rep} middle 80% only    : rho={rho2:+.3f} p={p2:.1e} n={len(sub2)}')

    print('Original gene-half check not replayed: Python hash randomization and the original split membership/seed were not preserved.')
    print('\n=== D) sample of L1_novel (orphan) vs L4_covered (dense) genes ===')
    feat2=feat.sort_values('go_meantop5')
    print('  most NOVEL (sparse) genes:', feat2['perturbation_name'].head(15).tolist())
    print('  most COVERED (dense) genes:', feat2['perturbation_name'].tail(15).tolist())
finally:os.chdir(old)

# Original notebook cell 10
old=Path.cwd();os.chdir(run)
try:
    import numpy as np, pandas as pd
    from scipy.stats import spearmanr
    df=pd.read_csv('data/predictions.csv',low_memory=False)
    r=df[df.dataset_family=='Replogle-Nadig'].copy()
    feat=pd.read_csv('spectra_run/artifacts/replogle_go_features.csv')
    d=r.merge(feat,on='perturbation_name',how='inner')
    z=d[d.protocol=='zeroshot'].copy()

    # Densified SPC: deciles of functional-similarity (train coverage). D1=most novel .. D10=most covered
    z['decile']=pd.qcut(z['go_meantop5'],10,labels=range(1,11))
    spc=z.groupby(['representation','decile'],observed=True)['overlap_at_N'].agg(['mean','median','count']).reset_index()
    spc.to_csv('spectra_run/artifacts/SPC_replogle_deciles.csv',index=False)
    print('=== Densified SPC (deciles; D1=most novel, D10=most covered) — overlap_at_N mean ===')
    piv=spc.pivot(index='decile',columns='representation',values='mean').round(4)
    print(piv)

    # Pure train-coverage curve: within STRONG-effect genes only (top half nsig_real), isolate coverage from effect size
    z['eff_half']=(z['de_nsig_counts_real']>z['de_nsig_counts_real'].median()).map({True:'strong',False:'weak'})
    print('\n=== "Pure" train-coverage: SPC within STRONG-effect perturbations only (removes intrinsic-difficulty) ===')
    for rep in ['HVG','SE']:
        zz=z[(z.representation==rep)&(z.eff_half=="strong")].copy()
        zz['q']=pd.qcut(zz['go_meantop5'],4,labels=['L1_novel','L2','L3','L4_covered'])
        m=zz.groupby('q',observed=True)['overlap_at_N'].mean()
        rho,p=spearmanr(zz['go_meantop5'],zz['overlap_at_N'])
        print(f'  {rep} (strong-effect only): rho={rho:+.3f} p={p:.1e} n={len(zz)}  L1={m["L1_novel"]:.3f} L4={m["L4_covered"]:.3f} ratio={m["L4_covered"]/m["L1_novel"]:.2f}x')

    # Final headline SPC (quartiles) table saved
    z['novlevel']=pd.qcut(z['go_meantop5'],4,labels=['L1_novel','L2','L3','L4_covered'])
    head=z.groupby(['representation','novlevel'],observed=True)['overlap_at_N'].agg(['mean','median','std','count']).reset_index()
    head.to_csv('spectra_run/artifacts/SPC_replogle_quartiles.csv',index=False)
    print('\nSaved SPC tables.')
    print('Curve statistics above were recomputed. The historical script also wrote a literal rounded summary; that summary is not emitted as a recomputation here.')
finally:os.chdir(old)

# Original notebook cell 11
tmp.cleanup()
