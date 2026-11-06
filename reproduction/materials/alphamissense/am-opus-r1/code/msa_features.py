import pyhmmer, time, numpy as np
from pyhmmer.easel import Alphabet, TextSequence, SequenceFile
from pyhmmer.plan7 import Pipeline

abc = Alphabet.amino()

def load_db(path):
    t=time.time()
    with SequenceFile(path, digital=True, alphabet=abc) as sf:
        seqs = sf.read_block()
    return seqs

def per_position_features(name, seq, db, max_hits=8000, incE=1e-3):
    """Return per-query-position coverage counts and residue lists via phmmer."""
    L=len(seq)
    q = TextSequence(name=name.encode(), sequence=seq).digitize(abc)
    pli = Pipeline(abc, incE=incE)
    hits = pli.search_seq(q, db)
    cov = np.zeros(L, dtype=int)          # homologs aligned (non-gap) at each query pos
    # store residues per position for conservation (cap memory by counting)
    from collections import defaultdict, Counter
    rescount = [Counter() for _ in range(L)]
    nhits=min(len(hits), max_hits)
    for hi,hit in enumerate(hits):
        if hi>=max_hits: break
        for dom in hit.domains:
            ali = dom.alignment
            qpos = ali.hmm_from - 1   # 0-based query index pointer (match states)
            hmmseq = ali.hmm_sequence
            tgtseq = ali.target_sequence
            for hc, tc in zip(hmmseq, tgtseq):
                if hc != '.':            # match state -> consumes a query position
                    if 0 <= qpos < L and tc not in ('-','.'):
                        cov[qpos]+=1
                        rescount[qpos][tc.upper()]+=1
                    qpos+=1
    # conservation: shannon entropy per position (bits) over observed residues incl self
    ent=np.full(L, np.nan)
    for i in range(L):
        c=rescount[i]
        tot=sum(c.values())
        if tot>0:
            ps=np.array([v/tot for v in c.values()])
            ent[i]=-(ps*np.log2(ps)).sum()
    return dict(L=L, nhits=nhits, total_hits=len(hits), cov=cov, entropy=ent)

if __name__=="__main__":
    import pandas as pd, json, sys
    m=pd.read_csv('work/assay_full.csv')
    uniq=m.drop_duplicates('target_seq')[['UniProt_ID','assay','target_seq']]
    print("loading SwissProt into memory...")
    db=load_db('seqs/uniprot_sprot.fasta.gz');
    print("db loaded:",len(db),"seqs")
    # test on 2 proteins: APP (P05067) and a small one
    test=uniq.head(2)
    for _,row in test.iterrows():
        t=time.time()
        f=per_position_features(row.UniProt_ID, row.target_seq, db)
        dt=time.time()-t
        cov=f['cov']
        print(f"{row.UniProt_ID} {row.assay}: L={f['L']} total_hits={f['total_hits']} "
              f"cov[min/med/max]={cov.min()}/{int(np.median(cov))}/{cov.max()} "
              f"frac_pos_cov0={(cov==0).mean():.2f} time={dt:.1f}s")
