from pathlib import Path
import numpy as np,pandas as pd,tomllib
from scipy.stats import pearsonr

def replay(root):
 rows=[];contexts=['k562','rpe1','jurkat','hepg2']
 for ctx in contexts:
  ag=np.load(root/f'results/agg_train_{ctx}.npz',allow_pickle=True)
  keys=list(zip(ag['celltype'].astype(str),ag['pert'].astype(str)))
  rm=dict(zip(keys,ag['real_means']));pm=dict(zip(keys,ag['pred_means']));cnt=dict(zip(keys,ag['counts']))
  means=np.load(root/f'results/unit_means_test_{ctx}.npz',allow_pickle=False)
  tr=dict(zip(means['gene'],means['real_mean']));tp=dict(zip(means['gene'],means['pred_mean']));tn=dict(zip(means['gene'],means['n_cells']))
  allgenes={g for c,g in keys if c==ctx and g!='non-targeting'}|(set(tr)-{'non-targeting'})
  others=[l for l in contexts if l!=ctx]
  exposed={l:{g for c,g in keys if c==l and g!='non-targeting'} for l in others}
  ctrl=rm[(ctx,'non-targeting')]
  delta={(l,g):v-rm[(l,'non-targeting')] for (l,g),v in rm.items() if g!='non-targeting'}
  train=[g for g in allgenes if (ctx,g) in delta]
  baseline1=np.mean([delta[(ctx,g)] for g in train],axis=0)
  archive=pd.read_csv(root/('results/per_unit_scores.csv' if ctx=='k562' else f'results/per_unit_scores_{ctx}.csv')).set_index('gene')
  manifest=pd.read_csv(root/('results/axis_manifest.csv' if ctx=='k562' else f'results/axis_manifest_{ctx}.csv')).set_index('gene')
  assert set(allgenes)==set(manifest.index)
  for g in sorted(allgenes):
   seen=int(g not in tr);e=sum(g in exposed[l] for l in others)
   assert seen==int(manifest.loc[g,'seen_pair']) and e==int(manifest.loc[g,'exposure_E'])
   if seen:r,p,n=rm[(ctx,g)],pm[(ctx,g)],int(cnt[(ctx,g)])
   else:r,p,n=tr[g],tp[g],int(tn[g])
   yr=r-ctrl;yp=p-ctrl;other=[delta[(l,g)] for l in others if (l,g) in delta];baseline2=np.mean(other,axis=0) if other else baseline1
   d={'context':ctx,'gene':g,'seen_pair':seen,'exposure_E':e,'n_cells':n,'level':str(manifest.loc[g,'level'])}
   for label,v in [('state',yp),('B1',baseline1),('B2',baseline2)]:
    d['pearson_delta_'+label]=float(pearsonr(v,yr).statistic) if np.std(v)>0 and np.std(yr)>0 else np.nan
    d['mse_delta_'+label]=float(np.mean((v-yr)**2))
   d['effect_mag']=float(np.linalg.norm(yr)); rows.append(d)
  out=pd.DataFrame([r for r in rows if r['context']==ctx]).set_index('gene')
  assert set(out.index)==set(archive.index)
  for col in ['pearson_delta_state','pearson_delta_B1','pearson_delta_B2','mse_delta_state','mse_delta_B1','mse_delta_B2','effect_mag','n_cells']:
   np.testing.assert_allclose(out.sort_index()[col],archive.sort_index()[col],atol=2e-6,rtol=2e-6,equal_nan=True,err_msg=ctx+' '+col)
  print(ctx, len(out),'original memberships and six performance measurements re-scored and matched.')
 return pd.DataFrame(rows)
if __name__=='__main__':
 import sys
 out=replay(Path(sys.argv[1]));print(out.groupby(['context','seen_pair']).pearson_delta_state.mean())
