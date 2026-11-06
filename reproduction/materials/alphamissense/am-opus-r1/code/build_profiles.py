import pandas as pd, numpy as np, time, sys; sys.path.insert(0,'work')
import pyhmmer
from pyhmmer.easel import Alphabet, TextSequence, SequenceFile
from pyhmmer.plan7 import Pipeline
from collections import Counter
abc=Alphabet.amino()
AAs=list("ACDEFGHIKLMNPQRSTVWY")
def load_db(path):
    with SequenceFile(path, digital=True, alphabet=abc) as sf: return sf.read_block()
def profile(name, seq, db, max_hits=8000, incE=1e-3):
    L=len(seq)
    q=TextSequence(name=name.encode(), sequence=seq).digitize(abc)
    hits=Pipeline(abc, incE=incE).search_seq(q, db)
    counts=[Counter() for _ in range(L)]
    for hi,hit in enumerate(hits):
        if hi>=max_hits: break
        for dom in hit.domains:
            ali=dom.alignment; qpos=ali.hmm_from-1
            for hc,tc in zip(ali.hmm_sequence, ali.target_sequence):
                if hc!='.':
                    if 0<=qpos<L and tc not in ('-','.'):
                        u=tc.upper()
                        if u in AAs: counts[qpos][u]+=1
                    qpos+=1
    return L,len(hits),counts
if __name__=="__main__":
    m=pd.read_csv('work/assay_full.csv')
    uniq=m.drop_duplicates('target_seq')[['UniProt_ID','target_seq']].reset_index(drop=True)
    db=load_db('seqs/uniprot_sprot.fasta.gz'); print("db",len(db))
    rows=[]; t0=time.time()
    for i,row in uniq.iterrows():
        L,tot,counts=profile(row.UniProt_ID,row.target_seq,db)
        for p in range(L):
            c=counts[p]; n=sum(c.values())
            d=dict(UniProt_ID=row.UniProt_ID, position=p+1, wt_seq=row.target_seq[p], n_aln=n, total_hits=tot)
            for a in AAs: d[a]=c.get(a,0)
            rows.append(d)
        if i%15==0: print(f"{i+1}/{len(uniq)} {row.UniProt_ID} ({time.time()-t0:.0f}s)")
    prof=pd.DataFrame(rows)
    prof.to_parquet('work/col_profiles.parquet')
    print("saved col_profiles.parquet", prof.shape, f"{time.time()-t0:.0f}s")
