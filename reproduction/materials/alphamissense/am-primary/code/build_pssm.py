#!/usr/bin/env python3
"""Site-independent PSSM baseline (theta=0.2 weighted per-column AA frequencies) for the SAME
a2m MSAs / clustering / column-mapping used by build_axes.py. Emits weighted AA counts per
assayed aligned position so a position-aware, AM-INDEPENDENT predictor can be scored on the
identical frozen bins. Memory-bounded chunked accumulation (handles nseq up to ~1.4M). New file."""
import zipfile, os, csv, subprocess, shutil, time, re
import numpy as np
OUT="/path/to/audit-home/spectra_alphamissense/stepB"; PG="/path/to/audit-home/spectra_alphamissense/proteingym"
MM=os.path.expanduser("~/spectra_alphamissense/bin/mmseqs/bin/mmseqs")
mzip=zipfile.ZipFile(f"{OUT}/DMS_msa_files_v1.3.zip"); mnames={os.path.basename(n):n for n in mzip.namelist()}
WORK=f"{OUT}/clu_work_pssm"; os.makedirs(WORK, exist_ok=True)
RM_INS=str.maketrans("","","abcdefghijklmnopqrstuvwxyz.")
RM_ALL=str.maketrans("","","abcdefghijklmnopqrstuvwxyz.-")
AA="ACDEFGHIKLMNPQRSTVWY"
LUT=np.full(256,-1,np.int64)
for i,c in enumerate(AA): LUT[ord(c)]=i

assays=[]; msa_meta={}
with open(f"{PG}/kept_assays_verified.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        mfn=r.get("MSA_filename","").strip()
        def gi(x):
            try: return int(float(r.get(x,"")))
            except: return None
        msa_meta.setdefault(mfn,{"start":gi("MSA_start"),"end":gi("MSA_end")})
        assays.append({"id":r["DMS_id"],"acc":r["resolved_accession"],"mfn":mfn,"dms_file":r["DMS_filename"]})
single_re=re.compile(r'^([A-Z])(\d+)([A-Z])$')
for a in assays:
    pos=set()
    with open(f"{PG}/dms_final_73/{a['dms_file']}") as dh:
        for row in csv.DictReader(dh):
            m=row["mutant"]
            if ":" in m: continue
            mm=single_re.match(m)
            if mm: pos.add(int(mm.group(2)))
    a["pos"]=pos

def process_msa(mfn):
    start=msa_meta[mfn]["start"]
    fa=f"{WORK}/seqs.fa"; focus=None; nseq=0
    with open(fa,"w") as out, mzip.open(mnames[mfn]) as fh:
        cur=[]
        def flush():
            nonlocal nseq,focus
            if not cur: return
            raw="".join(cur)
            if nseq==0: focus=raw
            s=raw.translate(RM_ALL); out.write(f">{nseq}\n{s if s else 'X'}\n"); nseq+=1
        for rawl in fh:
            line=rawl.decode('latin1').rstrip("\n")
            if line.startswith(">"): flush(); cur=[]
            else: cur.append(line)
        flush()
    res=f"{WORK}/clu"; tmp=f"{WORK}/tmp"
    subprocess.run([MM,"easy-cluster",fa,res,tmp,"--min-seq-id","0.8","-c","0.8","--cov-mode","1",
                    "--threads","16","-v","1"],capture_output=True,text=True)
    size={}; rep_of={}
    with open(res+"_cluster.tsv") as fh:
        for line in fh:
            rep,mem=line.rstrip("\n").split("\t"); rep_of[int(mem)]=rep; size[rep]=size.get(rep,0)+1
    w=np.ones(nseq)
    for rid in range(nseq):
        rep=rep_of.get(rid)
        if rep is not None: w[rid]=1.0/size[rep]
    # focus map
    pos2mc={}; mc=0
    for i,c in enumerate(focus):
        tpos=start+i
        if not (c.islower() or c=="."): pos2mc[tpos]=mc; mc+=1
    ncol=mc
    counts=np.zeros((ncol,20))           # weighted AA counts per match column
    # chunked stream
    BATCH=8000; batch=[]; wb=[]; rid=0; cur=[]
    def addrow(s,wt):
        arr=np.frombuffer(s.encode('latin1')[:ncol],dtype=np.uint8)
        if arr.size<ncol: arr=np.concatenate([arr,np.full(ncol-arr.size,ord('-'),np.uint8)])
        batch.append(arr); wb.append(wt)
    def flush_batch():
        if not batch: return
        M=np.vstack(batch); wv=np.asarray(wb)
        idx=LUT[M]
        for a in range(20):
            counts[:,a]+=((idx==a)*wv[:,None]).sum(0)
        batch.clear(); wb.clear()
    with mzip.open(mnames[mfn]) as fh:
        for rawl in fh:
            line=rawl.decode('latin1').rstrip("\n")
            if line.startswith(">"):
                if cur:
                    addrow("".join(cur).translate(RM_INS),w[rid]); rid+=1
                    if len(batch)>=BATCH: flush_batch()
                cur=[]
            else: cur.append(line)
        if cur:
            addrow("".join(cur).translate(RM_INS),w[rid])
        flush_batch()
    for p in [fa,res+"_cluster.tsv",res+"_rep_seq.fasta",res+"_all_seqs.fasta"]:
        try: os.remove(p)
        except: pass
    shutil.rmtree(tmp,ignore_errors=True)
    return pos2mc,counts

distinct=sorted(set(a["mfn"] for a in assays))
out=open(f"{OUT}/pssm_counts.tsv","w")
out.write("assay\ttarget_pos\tneff_col\t"+"\t".join(AA)+"\n")
t0=time.time()
for k,mfn in enumerate(distinct):
    st=time.time(); pos2mc,counts=process_msa(mfn)
    for a in assays:
        if a["mfn"]!=mfn: continue
        for p in sorted(a["pos"]):
            if p in pos2mc:
                c=counts[pos2mc[p]]; tot=c.sum()
                out.write(f"{a['id']}\t{p}\t{tot:.4f}\t"+"\t".join(f"{x:.4f}" for x in c)+"\n")
    print(f"[{k+1}/{len(distinct)}] {mfn:42s} ncol={counts.shape[0]:>5}  {time.time()-st:.1f}s",flush=True)
out.close()
print(f"PSSM counts written in {time.time()-t0:.0f}s -> {OUT}/pssm_counts.tsv")
print("DONE")
