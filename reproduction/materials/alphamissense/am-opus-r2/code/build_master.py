import numpy as np, pandas as pd
from Bio.Align import substitution_matrices
AA="ACDEFGHIKLMNPQRSTVWY"; aa2i={c:i for i,c in enumerate(AA)}
blos=substitution_matrices.load("BLOSUM62")

df=pd.read_csv('../data/predictions.csv')
df['wt']=df.mutant.str[0]; df['mt']=df.mutant.str[-1]
amap=pd.read_csv('assay_msa_map.csv')[['assay','MSA_filename','MSA_start']]
cov=pd.read_csv('msa_coverage_by_col.csv')  # MSA_filename,col0,nseq,coverage,n_present

df=df.merge(amap,on='assay',how='left')
df['col0']=df['position']-df['MSA_start']   # 0-based column

# attach coverage + nseq via (MSA_filename,col0)
df=df.merge(cov[['MSA_filename','col0','coverage','n_present','nseq']],on=['MSA_filename','col0'],how='left')

# PSSM score per variant from aacount matrices
pssm=np.full(len(df),np.nan)
alpha=1.0
aacount_cache={}
for fn,g in df.groupby('MSA_filename'):
    cnt=np.load(f'msa_feat/{fn}.aacount.npy')  # (L,20)
    tot=cnt.sum(axis=1,keepdims=True)+20*alpha
    freq=(cnt+alpha)/tot
    for idx,c0,wt,mt in zip(g.index,g.col0,g.wt,g.mt):
        if 0<=c0<cnt.shape[0] and wt in aa2i and mt in aa2i:
            pssm[idx]=np.log(freq[c0,aa2i[mt]]/freq[c0,aa2i[wt]])
df['pssm']=pssm

# BLOSUM62
def bl(w,m):
    try: return float(blos[w,m])
    except: return np.nan
df['blosum']=[bl(w,m) for w,m in zip(df.wt,df.mt)]

# sanity
print('rows',len(df))
print('coverage NaN:',df.coverage.isna().sum(),'(out-of-MSA-range positions)')
print('pssm NaN:',df.pssm.isna().sum())
print('blosum NaN:',df.blosum.isna().sum())
print('coverage describe:'); print(df.coverage.describe())
df.to_parquet('master.parquet')
print('saved master.parquet')
