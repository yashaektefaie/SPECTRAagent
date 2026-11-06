import pandas as pd, numpy as np, time, json
import sys; sys.path.insert(0,'work')
from msa_features import load_db, per_position_features

m=pd.read_csv('work/assay_full.csv')
uniq=m.drop_duplicates('target_seq')[['UniProt_ID','target_seq']].reset_index(drop=True)
print("unique proteins:",len(uniq))
db=load_db('seqs/uniprot_sprot.fasta.gz'); print("db:",len(db))

recs=[]
prot_summary=[]
t0=time.time()
for i,row in uniq.iterrows():
    up=row.UniProt_ID; seq=row.target_seq
    try:
        f=per_position_features(up, seq, db)
    except Exception as e:
        print("FAIL",up,e); continue
    cov=f['cov']; ent=f['entropy']; tot=f['total_hits']
    covf = cov/max(tot,1)
    for p in range(f['L']):
        recs.append((row.target_seq[:0], up, p+1, row.target_seq[p], int(cov[p]), float(covf[p]),
                     float(ent[p]) if not np.isnan(ent[p]) else np.nan, tot))
    prot_summary.append((up, f['L'], tot, int(cov.min()), int(np.median(cov)), int(cov.max())))
    if i%10==0: print(f"  {i+1}/{len(uniq)} {up} hits={tot} ({time.time()-t0:.0f}s)")
print(f"done in {time.time()-t0:.0f}s")

pos=pd.DataFrame(recs, columns=['_x','UniProt_ID','position','wt_seq','cov','cov_frac','entropy','total_hits']).drop(columns=['_x'])
pos.to_parquet('work/pos_features.parquet')
ps=pd.DataFrame(prot_summary, columns=['UniProt_ID','L','total_hits','cov_min','cov_med','cov_max'])
ps.to_csv('work/protein_msa_summary.csv', index=False)
print("saved pos_features.parquet rows=",len(pos))
print(ps.describe()[['total_hits','cov_med','cov_max']].to_string())
print("\nproteins with <10 total hits (shallow):")
print(ps[ps.total_hits<10].to_string(index=False))
