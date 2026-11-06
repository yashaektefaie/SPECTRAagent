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
