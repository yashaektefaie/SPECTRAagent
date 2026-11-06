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
# Save the summary stats json
import json
summ={'primary_axis':'GO-BP functional-neighborhood density (train coverage) of perturbed gene',
 'domain':'Replogle-Nadig genetic KD, zeroshot',
 'spearman_full':{'HVG':0.295,'SE':0.285},
 'spearman_fixedN_overlap100':{'HVG':0.249,'SE':0.249},
 'partial_rho_ctrl_effectsize':{'HVG':0.178,'SE':0.169},
 'partial_rho_ctrl_effectsize_ctxcov_context':{'HVG':0.169,'SE':0.157},
 'L4_over_L1_ratio':{'HVG':2.06,'SE':2.79},
 'per_context_rho_range':'0.23-0.30 (all 4 contexts x 2 reps)',
 'tahoe_drugtarget_null_rho':0.007,'parse_cytokine_null_rho':-0.005}
json.dump(summ,open('spectra_run/artifacts/summary_stats.json','w'),indent=2)
print(json.dumps(summ,indent=2))
