import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
df=pd.read_csv('data/predictions.csv',low_memory=False)
r=df[df.dataset_family=='Replogle-Nadig'].copy()
feat=pd.read_csv('spectra_run/artifacts/replogle_go_features.csv')
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
