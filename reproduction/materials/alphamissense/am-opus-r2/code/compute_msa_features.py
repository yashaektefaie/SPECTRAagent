import numpy as np, pandas as pd, os, time, sys

AA="ACDEFGHIKLMNPQRSTVWY"
aa2i={c:i for i,c in enumerate(AA)}
GAP=set(b"-.")  # gap bytes

mp=pd.read_csv('assay_msa_map.csv')
# one MSA file may serve multiple assays; process unique files, store by file
files=mp.drop_duplicates('MSA_filename')[['MSA_filename','MSA_start','MSA_end','MSA_len','seq_len']]

os.makedirs('msa_feat',exist_ok=True)
covrows=[]
def load_matrix(path):
    # fast path: single line per seq
    with open(path,'rb') as f:
        data=f.read()
    lines=data.split(b'\n')
    seqs=[]
    cur=[]
    for ln in lines:
        if not ln: continue
        if ln[:1]==b'>':
            if cur: seqs.append(b''.join(cur)); cur=[]
        else:
            cur.append(ln.strip())
    if cur: seqs.append(b''.join(cur))
    L=len(seqs[0])
    # verify uniform
    buf=b''.join(seqs)
    if len(buf)%L!=0:
        # non-uniform: pad/skip
        seqs=[s for s in seqs if len(s)==L]
        buf=b''.join(seqs)
    arr=np.frombuffer(buf,dtype=np.uint8).reshape(-1,L)
    return arr

for _,r in files.iterrows():
    fn=r.MSA_filename; path=f'msa/{fn}'
    t0=time.time()
    arr=load_matrix(path)
    nseq,L=arr.shape
    # coverage: present = alphabetic (A-Z or a-z); gap = '-' or '.'
    isalpha=((arr>=65)&(arr<=90))|((arr>=97)&(arr<=122))
    present=isalpha  # letters (incl X)
    coverage=present.mean(axis=0)            # frac present per column
    # AA counts per column (uppercase standard AAs; fold lowercase to upper)
    upper=np.where((arr>=97)&(arr<=122), arr-32, arr)  # to upper
    cnt=np.zeros((L,20),dtype=np.float64)
    for i,c in enumerate(AA):
        cb=ord(c)
        cnt[:,i]=(upper==cb).sum(axis=0)
    np.save(f'msa_feat/{fn}.aacount.npy', cnt)
    for col in range(L):
        covrows.append((fn,col,int(nseq),float(coverage[col]),int(present[:,col].sum())))
    print(f'{fn}: nseq={nseq} L={L} in {time.time()-t0:.1f}s', flush=True)

cov=pd.DataFrame(covrows,columns=['MSA_filename','col0','nseq','coverage','n_present'])
cov.to_csv('msa_coverage_by_col.csv',index=False)
print('DONE total columns:',len(cov))
