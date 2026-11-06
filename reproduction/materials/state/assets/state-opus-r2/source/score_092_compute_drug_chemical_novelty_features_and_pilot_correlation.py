import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

df = pd.read_csv('data/predictions.csv', low_memory=False)
t = df[df.dataset_family=='Tahoe-100M'].copy()
du = t.drop_duplicates('perturbation_name')[['perturbation_name','drugmeta_canonical_smiles','drugmeta_moa-fine','drugmeta_targets','drugmeta_human-approved']].reset_index(drop=True)

# Morgan fingerprints
fps={};
for _,r in du.iterrows():
    s=r['drugmeta_canonical_smiles']
    if pd.isna(s): continue
    m=Chem.MolFromSmiles(str(s))
    if m is None: continue
    fps[r['perturbation_name']]=AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048)
print('fingerprints computed for', len(fps),'/',len(du),'drugs')

names=list(fps.keys())
# pairwise nearest-neighbor Tanimoto (leave-one-out within panel = training coverage proxy)
nn_sim={}; mean_top5={}
for i,n in enumerate(names):
    sims=DataStructs.BulkTanimotoSimilarity(fps[n],[fps[m] for m in names])
    sims=np.array(sims); sims[i]=-1  # exclude self
    order=np.sort(sims)[::-1]
    nn_sim[n]=order[0]           # max Tanimoto to any other drug
    mean_top5[n]=order[:5].mean()
nn=pd.Series(nn_sim);
print('\nnearest-neighbor Tanimoto distribution across drugs:')
print(nn.describe().round(3))
print('quartiles:', np.percentile(list(nn_sim.values()),[0,25,50,75,100]).round(3))

# save features
feat=du.copy()
feat['nn_tanimoto']=feat['perturbation_name'].map(nn_sim)
feat['meantop5_tanimoto']=feat['perturbation_name'].map(mean_top5)
feat['chem_novelty']=1-feat['nn_tanimoto']
feat.to_csv('spectra_run/artifacts/tahoe_drug_features.csv',index=False)
print('\nsaved tahoe_drug_features.csv')

# PILOT: correlation of chem_novelty with overlap_at_N on Tahoe zeroshot (both reps)
z=t[t.protocol=='zeroshot'].merge(feat[['perturbation_name','chem_novelty','nn_tanimoto']],on='perturbation_name',how='left')
z=z.dropna(subset=['chem_novelty'])
from scipy.stats import spearmanr
for rep in ['HVG','SE']:
    zz=z[z.representation==rep]
    rho,p=spearmanr(zz['nn_tanimoto'], zz['overlap_at_N'])
    print(f'PILOT zeroshot {rep}: spearman(nn_tanimoto, overlap)={rho:.3f} p={p:.1e} n={len(zz)}  (expect POSITIVE: more similar->better)')
