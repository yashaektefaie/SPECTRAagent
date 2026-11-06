import numpy as np, pandas as pd, glob, os
from Bio import Align
aa3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
ref=pd.read_csv('DMS_substitutions.csv').set_index('DMS_id')
df=pd.read_parquet('master.parquet'); df['wt']=df.mutant.str[0]

afseq={}; afpl={}
for f in glob.glob('af/*.pdb'):
    acc=os.path.basename(f)[:-4]; s={}; pl={}
    for line in open(f):
        if line.startswith('ATOM') and line[12:16].strip()=='CA':
            r=int(line[22:26]); s[r]=aa3.get(line[17:20].strip(),'X'); pl[r]=float(line[60:66])
    resis=sorted(s); afseq[acc]=(resis[0],''.join(s[r] for r in resis)); afpl[acc]=pl

aligner=Align.PairwiseAligner(); aligner.mode='local'; aligner.open_gap_score=-10; aligner.extend_gap_score=-1; aligner.match_score=2; aligner.mismatch_score=-1

# per-assay position -> AF resi map
plddt_col=np.full(len(df),np.nan)
diag=[]
for assay,g in df.groupby('assay'):
    acc=g.acc.iloc[0]
    if acc not in afseq: diag.append((assay,acc,'no_AF',np.nan)); continue
    tseq=ref.loc[assay,'target_seq']; start,aseq=afseq[acc]
    idx=aseq.find(tseq)
    posmap={}
    if idx>=0:
        base=start+idx
        for p in range(1,len(tseq)+1): posmap[p]=base+(p-1)
        method='sub'
    else:
        aln=aligner.align(tseq,aseq)[0]
        # aligned blocks: aln.aligned = ([ (t0,t1),...],[(a0,a1),...])
        tb,ab=aln.aligned
        for (t0,t1),(a0,a1) in zip(tb,ab):
            for k in range(t1-t0):
                posmap[t0+k+1]=start+a0+k   # target pos (1-based) -> af resi
        method='aln'
    # fill
    idxs=g.index; pls=[]; okc=0; tot=0
    for i,p,w in zip(idxs,g.position,g.wt):
        r=posmap.get(p)
        if r is not None:
            plddt_col[i]=afpl[acc].get(r,np.nan)
            afaa=aseq[r-start] if 0<=r-start<len(aseq) else None
            tot+=1; okc+= (afaa==w)
    diag.append((assay,acc,method,okc/tot if tot else np.nan))
df['plddt']=plddt_col
D=pd.DataFrame(diag,columns=['assay','acc','method','wtmatch'])
print('assays with wtmatch<0.9:', (D.wtmatch<0.9).sum())
print(D[D.wtmatch<0.9].to_string(index=False))
print('\noverall plddt non-null:',df.plddt.notna().sum(),'/',len(df))
# per-variant WT match overall
meta=pd.read_csv('assay_meta.csv')[['assay','coarse_selection_type']]
df=df.merge(meta,on='assay',how='left')
df.to_parquet('master_plddt.parquet')
D.to_csv('plddt_mapping_diag.csv',index=False)
print('saved corrected master_plddt.parquet')
