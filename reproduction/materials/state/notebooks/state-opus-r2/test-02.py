# Original notebook cell 2
from pathlib import Path
import sys,json,hashlib,io,os,tempfile,shutil,runpy,contextlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from IPython.display import display,Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
root=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'STATE_INPUTS.json').is_file()),None)
assert root is not None,'Extract the complete STATE bundle first.'
manifest=json.loads((root/'STATE_INPUTS.json').read_text())
P=root/'assets'/'state-opus-r2'
for relative,meta in manifest.items():
    if relative.startswith('state-opus-r2'+'/'):
        assert hashlib.sha256((root/'assets'/relative).read_bytes()).hexdigest()==meta['sha256'],relative
print('Verified all run input and code hashes.')

# Original notebook cell 3
def draw(data,x,y,group):
    rng=np.random.default_rng(7);rows=[]
    for keys,g in data.groupby([group,x],observed=True):
        v=pd.to_numeric(g[y],errors='coerce').dropna().values
        if not len(v):continue
        bs=np.array([rng.choice(v,len(v),True).mean() for _ in range(1000)])
        rows.append({group:keys[0],x:keys[1],'n':len(v),'mean':v.mean(),'lo':np.quantile(bs,.025),'hi':np.quantile(bs,.975)})
    t=pd.DataFrame(rows);display(t)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');levels=sorted(t[x].unique(),key=str)
    for name,g in t.groupby(group,observed=True):
        g=g.assign(xpos=g[x].map({v:i for i,v in enumerate(levels)})).sort_values('xpos')
        ax.errorbar(g.xpos,g['mean'],yerr=np.stack([g['mean']-g.lo,g.hi-g['mean']]),marker='o',capsize=2,label=str(name))
    ax.set_xticks(range(len(levels)),levels,rotation=25);ax.set(xlabel=x,ylabel=y);ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120);plt.close(fig);display(Image(data=buf.getvalue()))

# Original notebook cell 4
raw=pd.read_csv(P/'data/predictions.csv',low_memory=False)
feat=pd.read_csv(P/'spectra_run/artifacts/replogle_go_features.csv')
print('Rows:',len(raw),'annotated gene identities:',len(feat))
tmp=tempfile.TemporaryDirectory();run=Path(tmp.name);shutil.copytree(P/'data',run/'data');shutil.copytree(P/'spectra_run',run/'spectra_run')

# Original notebook cell 5
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
go=json.load(open(P/'spectra_run/artifacts/go_terms.json'))
df=pd.read_csv(P/'data/predictions.csv',low_memory=False)
r=df[df.dataset_family=='Replogle-Nadig'].copy()
genes=sorted(r['perturbation_name'].unique())

BP={g:set(go.get(g,{}).get('BP',[])) for g in genes}
nbp={g:len(BP[g]) for g in genes}
have=[g for g in genes if nbp[g]>0]
print('genes with BP terms:',len(have))

# Functional neighborhood density via GO-BP Jaccard.
# For each gene, mean of top-5 Jaccard to OTHER genes = functional coverage.
# Compute efficiently via inverted index -> but Jaccard needs pairwise. Do it with sets (2023^2/2 ~ 2M, ok)
import itertools
arr=have
# Precompute for speed: represent as frozensets
S={g:BP[g] for g in arr}
# To bound cost, compute for each gene its top-k jaccard by iterating (2M ops)
top5=[]; nnj=[]
den_thr=[]  # count neighbors with jaccard>=0.2
for i,g in enumerate(arr):
    sg=S[g]; lg=len(sg)
    sims=[]
    for h in arr:
        if h==g: continue
        sh=S[h]
        inter=len(sg & sh)
        if inter==0:
            sims.append(0.0); continue
        j=inter/(lg+len(sh)-inter)
        sims.append(j)
    sims=np.array(sims); sims.sort()
    top5.append(sims[-5:].mean()); nnj.append(sims[-1]); den_thr.append((sims>=0.2).sum())
feat=pd.DataFrame({'perturbation_name':arr,'go_nbp':[nbp[g] for g in arr],
                   'go_meantop5':top5,'go_nn':nnj,'go_ndense':den_thr})

expected=pd.read_csv(P/'spectra_run/artifacts/replogle_go_features.csv')
a=feat.set_index('perturbation_name').sort_index();b=expected.set_index('perturbation_name').sort_index()
assert a.index.equals(b.index)
for col in ['go_meantop5','go_nn','go_nbp','go_ndense']:np.testing.assert_allclose(a[col],b[col],atol=1e-12)
print('GO-BP neighborhood feature exactly reproduced from retained GO membership lists.')

# Original notebook cell 6
old=Path.cwd();os.chdir(run)
try:
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
finally:os.chdir(old)

# Original notebook cell 7
tmp.cleanup()
