import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
df=pd.read_csv('data/predictions.csv',low_memory=False)
go=json.load(open('spectra_run/artifacts/tahoe_target_go.json'))
t=df[df.dataset_family=='Tahoe-100M'].copy()
du=t.drop_duplicates('perturbation_name').copy()

# target-gene functional density among the Tahoe target-gene set (training proxy)
tgenes=sorted(go.keys())
S={g:set(go.get(g,[])) for g in tgenes}
def gdens(g,topk=5):
    sg=S.get(g,set())
    if not sg: return np.nan
    sims=[]
    for h in tgenes:
        if h==g: continue
        sh=S[h]
        if not sh: continue
        inter=len(sg&sh); sims.append(inter/(len(sg)+len(sh)-inter) if inter else 0.0)
    sims=np.sort(sims)[::-1]; return np.mean(sims[:topk])
gd={g:gdens(g) for g in tgenes}

def drug_density(targets):
    if pd.isna(targets): return np.nan
    vals=[gd.get(x.strip(),np.nan) for x in str(targets).split(',')]
    vals=[v for v in vals if v==v]
    return np.max(vals) if vals else np.nan   # best-covered target
du['tgt_funcdens']=du['drugmeta_targets'].apply(drug_density)
print('drugs with target functional density:', du['tgt_funcdens'].notna().sum(),'/',len(du))

z=t.merge(du[['perturbation_name','tgt_funcdens']],on='perturbation_name',how='left')
print('\n=== TAHOE CROSS-FAMILY CONFIRMATION: drug target-gene functional density vs overlap ===')
for proto in ['zeroshot','fewshot']:
    for rep in ['HVG','SE']:
        zz=z[(z.protocol==proto)&(z.representation==rep)].dropna(subset=['tgt_funcdens'])
        rho,p=spearmanr(zz['tgt_funcdens'],zz['overlap_at_N'])
        print(f'  {proto:9s} {rep}: rho={rho:+.3f} p={p:.1e} n={len(zz)}')
# level curve zeroshot
print('\n=== Tahoe zeroshot SPC by target-density quartile ===')
zt=z[z.protocol=='zeroshot'].dropna(subset=['tgt_funcdens']).copy()
zt['lvl']=pd.qcut(zt['tgt_funcdens'],4,labels=['L1_novel','L2','L3','L4_covered'],duplicates='drop')
for rep in ['HVG','SE']:
    print(f'-- {rep} --')
    print(zt[zt.representation==rep].groupby('lvl',observed=True)['overlap_at_N'].agg(['mean','count']).round(4))
