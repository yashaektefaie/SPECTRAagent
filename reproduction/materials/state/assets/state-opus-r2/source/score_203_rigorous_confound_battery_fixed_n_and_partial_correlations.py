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
