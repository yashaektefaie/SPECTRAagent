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
