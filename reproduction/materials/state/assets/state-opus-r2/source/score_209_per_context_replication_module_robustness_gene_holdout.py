import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
df=pd.read_csv('data/predictions.csv',low_memory=False)
r=df[df.dataset_family=='Replogle-Nadig'].copy()
feat=pd.read_csv('spectra_run/artifacts/replogle_go_features.csv')
go=json.load(open('spectra_run/artifacts/go_terms.json'))
d=r.merge(feat,on='perturbation_name',how='inner')
z=d[d.protocol=='zeroshot']

print('=== A) PER-CONTEXT replication (each context independent) ===')
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

print('\n=== C) UNBIASED effect: gene-level 50/50 holdout (median split by hash of gene name) ===')
genes=sorted(feat['perturbation_name'])
half={g:(hash(g)%2) for g in genes}  # deterministic-ish; note hash may vary but fine for split
z=z.copy(); z['half']=z['perturbation_name'].map(half)
for rep in ['HVG','SE']:
    for h in [0,1]:
        zz=z[(z.representation==rep)&(z.half==h)]
        rho,p=spearmanr(zz['go_meantop5'],zz['overlap_at_N'])
        print(f'   {rep} gene-half {h}: rho={rho:+.3f} p={p:.1e} n={len(zz)}')

print('\n=== D) sample of L1_novel (orphan) vs L4_covered (dense) genes ===')
feat2=feat.sort_values('go_meantop5')
print('  most NOVEL (sparse) genes:', feat2['perturbation_name'].head(15).tolist())
print('  most COVERED (dense) genes:', feat2['perturbation_name'].tail(15).tolist())
