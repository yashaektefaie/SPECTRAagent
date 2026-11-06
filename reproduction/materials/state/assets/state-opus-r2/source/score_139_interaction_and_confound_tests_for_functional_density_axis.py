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
