#!/usr/bin/env python3
"""Step B Path-C: per-column DEPTH (theta=0.2 weighted Neff) + OCCUPANCY (raw non-gap fraction)
on ProteinGym v1.3 a2m MSAs, mapped to every assayed single-mutant position. Freeze + report.
Self-computed theta=0.2 weights via 80%-id clustering (shipped PG weights don't align with a2m;
validated against MSA_N_eff). NO AM scoring. New files only."""
import zipfile, os, csv, subprocess, io, shutil, time
import numpy as np
OUT="/path/to/audit-home/spectra_alphamissense/stepB"; PG="/path/to/audit-home/spectra_alphamissense/proteingym"
MM=os.path.expanduser("~/spectra_alphamissense/bin/mmseqs/bin/mmseqs")
mzip=zipfile.ZipFile(f"{OUT}/DMS_msa_files_v1.3.zip"); mnames={os.path.basename(n):n for n in mzip.namelist()}
WORK=f"{OUT}/clu_work"; os.makedirs(WORK, exist_ok=True)
RM_INS=str.maketrans("","","abcdefghijklmnopqrstuvwxyz.")      # keep match cols (upper + '-')
RM_ALL=str.maketrans("","","abcdefghijklmnopqrstuvwxyz.-")     # raw residues for clustering

# ---- panel: assays, MSA meta, single positions ----
assays=[]
msa_meta={}
with open(f"{PG}/kept_assays_verified.tsv") as fh:
    for r in csv.DictReader(fh,delimiter="\t"):
        mfn=r.get("MSA_filename","").strip()
        def gi(x):
            try: return int(float(r.get(x,"")))
            except: return None
        msa_meta.setdefault(mfn,{"start":gi("MSA_start"),"end":gi("MSA_end"),"len":gi("MSA_len")})
        assays.append({"id":r["DMS_id"],"acc":r["resolved_accession"],"gene":r["DMS_id"].split("_")[0],
                       "mfn":mfn,"dms_file":r["DMS_filename"],
                       "fam":("Tsuboyama_domain" if "Tsuboyama" in r["DMS_id"] else "other")})
import re
single_re=re.compile(r'^([A-Z])(\d+)([A-Z])$')
# positions + variant counts per assay (target_seq coords = DMS coords)
for a in assays:
    pos={}
    with open(f"{PG}/dms_final_73/{a['dms_file']}") as dh:
        for row in csv.DictReader(dh):
            m=row["mutant"]
            if ":" in m: continue
            mm=single_re.match(m)
            if mm: pos[int(mm.group(2))]=pos.get(int(mm.group(2)),0)+1
    a["pos"]=pos

# ---- per-MSA: compute per-column neff + occupancy ----
def process_msa(mfn):
    meta=msa_meta[mfn]; start=meta["start"]
    # 1. extract raw seqs (clustering) + capture focus(row0 raw) ; also count nseq
    fa=f"{WORK}/seqs.fa"; focus=None; nseq=0
    with open(fa,"w") as out, mzip.open(mnames[mfn]) as fh:
        cur=[]
        def flush():
            nonlocal nseq,focus
            if not cur: return
            raw="".join(cur)
            if nseq==0: focus=raw
            s=raw.translate(RM_ALL)
            out.write(f">{nseq}\n{s if s else 'X'}\n"); nseq+=1
        for rawl in fh:
            line=rawl.decode('latin1').rstrip("\n")
            if line.startswith(">"): flush(); cur=[]
            else: cur.append(line)
        flush()
    # 2. cluster 80% -> weights aligned to row id
    res=f"{WORK}/clu"; tmp=f"{WORK}/tmp"
    subprocess.run([MM,"easy-cluster",fa,res,tmp,"--min-seq-id","0.8","-c","0.8","--cov-mode","1",
                    "--threads","16","-v","1"],capture_output=True,text=True)
    rep_of={}; size={}
    with open(res+"_cluster.tsv") as fh:
        for line in fh:
            rep,mem=line.rstrip("\n").split("\t")
            rep_of[int(mem)]=rep; size[rep]=size.get(rep,0)+1
    w=np.zeros(nseq)
    for rid in range(nseq):
        rep=rep_of.get(rid)
        w[rid]=1.0/size[rep] if rep is not None else 1.0
    # 3. focus map: target_pos -> matchcol ; lowcov set
    pos2mc={}; lowcov=set(); mc=0
    for i,c in enumerate(focus):
        tpos=start+i
        if c.islower() or c==".": lowcov.add(tpos)
        else: pos2mc[tpos]=mc; mc+=1
    ncol=mc
    # 4. stream a2m -> per matchcol weighted neff + raw nongap count
    neff=np.zeros(ncol); occ=np.zeros(ncol)
    rid=0; cur=[]
    with mzip.open(mnames[mfn]) as fh:
        def acc(cur,rid):
            if not cur: return
            s="".join(cur).translate(RM_INS)            # match cols (upper + '-')
            arr=np.frombuffer(s.encode('latin1')[:ncol],dtype=np.uint8)
            if arr.size<ncol: arr=np.concatenate([arr,np.full(ncol-arr.size,ord('-'),np.uint8)])
            ng=arr!=ord('-')
            neff[ng]+=w[rid]; occ[ng]+=1.0
        for rawl in fh:
            line=rawl.decode('latin1').rstrip("\n")
            if line.startswith(">"):
                acc(cur,rid); rid+=1 if cur else 0
                if cur: pass
                cur=[]
            else: cur.append(line)
        acc(cur,rid)
    occ=occ/nseq
    # cleanup
    for p in [fa,res+"_cluster.tsv",res+"_rep_seq.fasta",res+"_all_seqs.fasta"]:
        try: os.remove(p)
        except: pass
    shutil.rmtree(tmp,ignore_errors=True)
    return {"pos2mc":pos2mc,"lowcov":lowcov,"neff":neff,"occ":occ,"nseq":nseq,"ncol":ncol,
            "global_neff":float(w.sum())}

# process distinct MSAs (sorted by file size proxy: nseq unknown yet -> just iterate)
distinct=sorted(set(a["mfn"] for a in assays))
msa_res={}
t0=time.time()
for k,mfn in enumerate(distinct):
    st=time.time(); r=process_msa(mfn); msa_res[mfn]=r
    print(f"[{k+1}/{len(distinct)}] {mfn:42s} nseq={r['nseq']:>7} ncol={r['ncol']:>5} "
          f"gNeff={r['global_neff']:.1f}  {time.time()-st:.1f}s",flush=True)
print(f"all MSAs processed in {time.time()-t0:.0f}s",flush=True)

# ---- map to assayed positions, FREEZE ----
DEPTH_EDGES=[2,10,50,200]
def dbin(v):
    b=0
    for e in DEPTH_EDGES:
        if v>=e: b+=1
    return b
rows=[]
lowcov_hits=0
for a in assays:
    R=msa_res[a["mfn"]]
    for p,nv in a["pos"].items():
        if p in R["pos2mc"]:
            mc=R["pos2mc"][p]; nf=float(R["neff"][mc]); oc=float(R["occ"][mc]); lc=0
        elif p in R["lowcov"]:
            nf=float('nan'); oc=float('nan'); lc=1; lowcov_hits+=1
        else:
            continue
        rows.append({"assay":a["id"],"acc":a["acc"],"gene":a["gene"],"fam":a["fam"],"pos":p,
                     "neff":nf,"occ":oc,"lowcov":lc,"nvar":nv})

frozen=f"{OUT}/axis3_pathC_percolumn_FROZEN.tsv"
with open(frozen,"w") as out:
    out.write("assay\taccession\tgene\tassay_family\ttarget_pos\tneff_theta02\toccupancy\tlow_cov_insert\tdepth_bin\tn_variants\n")
    for r in rows:
        db = "" if r["lowcov"] else dbin(r["neff"])
        out.write(f"{r['assay']}\t{r['acc']}\t{r['gene']}\t{r['fam']}\t{r['pos']}\t"
                  f"{'' if r['lowcov'] else round(r['neff'],3)}\t{'' if r['lowcov'] else round(r['occ'],4)}\t"
                  f"{r['lowcov']}\t{db}\t{r['nvar']}\n")
print(f"\nFROZEN: {frozen}  ({len(rows)} assay-positions; low_cov_insert={lowcov_hits})",flush=True)
print("Run report_axes.py next for distributions + confound + interaction.")
