"""Execute recovered original AlphaMissense feature functions with path-only changes.
Original 16-thread MMseqs parameters retained; cached original calibration clusters
or newly generated identical-MSA clusters are reused between axes and PSSM counts.
No model inference, samples, axis thresholds or scientific outcomes are changed.
"""
from pathlib import Path
import argparse,subprocess,json,time,hashlib,shutil,traceback,resource,zipfile,os
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--msa',nargs='*');ap.add_argument('--all',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
SOURCE=Path('/mnt/spectra-am/path/to/audit-home');ARCH=SOURCE/'QUARANTINE/stepB/DMS_msa_files_v1.3.zip'
PG=SOURCE/'spectra_alphamissense/proteingym';OUT=Path('/tmp/alpha-upstream-replay');OUT.mkdir(exist_ok=True)
(OUT/'clusters').mkdir(exist_ok=True);(OUT/'results').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
MM='/tmp/mmseqs-original';current='';events=json.loads((OUT/'events.json').read_text()) if (OUT/'events.json').exists() else []
NATIVE_RUN=subprocess.run
# Hook the external command, preserving all original arguments. Record resources
# and keep verified clusters for the original PSSM function's second use.
def cluster_run(command,**kwargs):
    if not command or command[0]!=MM:return NATIVE_RUN(command,**kwargs)
    res=Path(command[3]);saved=OUT/'clusters'/(current+'.clu_cluster.tsv')
    original=SOURCE/'QUARANTINE/stepB/calib'/(current+'.clu_cluster.tsv')
    origin='replay_cache'
    # Freshly regenerate even the five historical calibration clusters; compare them after execution.
    if saved.exists():
        shutil.copyfile(saved,str(res)+'_cluster.tsv')
        events.append(dict(msa=current,kind='cluster_reuse',origin=origin,sha256=hashlib.sha256(saved.read_bytes()).hexdigest()))
        return subprocess.CompletedProcess(command,0,'','')
    started=time.monotonic()
    result=NATIVE_RUN(command,**kwargs)
    elapsed=time.monotonic()-started
    (OUT/'logs'/(current+'.mmseqs.log')).write_text((result.stdout or '')+'\n'+(result.stderr or ''))
    events.append(dict(msa=current,kind='fresh_clustering',command=command,returncode=result.returncode,elapsed_seconds=elapsed,child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
    (OUT/'events.json').write_text(json.dumps(events,indent=2))
    if result.returncode:raise RuntimeError(f'MMseqs failed ({result.returncode}) for {current}; inspect saved log')
    shutil.copyfile(str(res)+'_cluster.tsv',saved)
    if original.exists():
        events.append(dict(msa=current,kind='original_cluster_comparison',identical=hashlib.sha256(original.read_bytes()).hexdigest()==hashlib.sha256(saved.read_bytes()).hexdigest()))
    return result
subprocess.run=cluster_run
# Original source is executed only through its reusable process_msa definition.
def load_original(path,stop_marker,work_name):
    source=path.read_text().split(stop_marker)[0]
    source=source.replace('OUT="/path/to/audit-home/spectra_alphamissense/stepB"; PG="/path/to/audit-home/spectra_alphamissense/proteingym"',f'OUT={str(OUT)!r}; PG={str(PG)!r}')
    source=source.replace('MM=os.path.expanduser("~/spectra_alphamissense/bin/mmseqs/bin/mmseqs")',f'MM={MM!r}')
    source=source.replace('mzip=zipfile.ZipFile(f"{OUT}/DMS_msa_files_v1.3.zip")',f'mzip=zipfile.ZipFile({str(ARCH)!r})')
    env={};exec(compile(source,str(path),'exec'),env)
    return env
axes=load_original(SOURCE/'build_axes.py','# process distinct MSAs','clu_work')
pssm=load_original(SOURCE/'QUARANTINE/build_pssm.py','distinct=sorted','clu_work_pssm')
selected=sorted({a['mfn'] for a in axes['assays']}) if args.all else args.msa
assert selected,'Specify --msa <exact filename> or --all'
# Small-to-large ordering affects no within-MSA calculation and enables checkpoints.
selected=sorted(selected,key=lambda name:axes['mzip'].getinfo(axes['mnames'][name]).file_size)
for idx,name in enumerate(selected,1):
    current=name;dest=OUT/'results'/(name+'.npz')
    if args.resume and dest.exists():print('RESUME',name,flush=True);continue
    started=time.monotonic();print('START',idx,len(selected),name,flush=True)
    try:
        ar=axes['process_msa'](name)
        pos2mc,counts=pssm['process_msa'](name)
        assert pos2mc==ar['pos2mc']
        np.savez_compressed(dest,neff=ar['neff'],occupancy=ar['occ'],counts=counts,
            positions=np.array(list(ar['pos2mc'])),match_columns=np.array(list(ar['pos2mc'].values())),
            lowcov=np.array(sorted(ar['lowcov'])),nseq=ar['nseq'],ncol=ar['ncol'],global_neff=ar['global_neff'])
        events.append(dict(msa=name,kind='features_complete',elapsed_seconds=time.monotonic()-started,nseq=ar['nseq'],ncol=ar['ncol'],global_neff=ar['global_neff'],output_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        print('DONE',name,'nseq',ar['nseq'],'seconds',round(time.monotonic()-started,2),flush=True)
    except Exception:
        events.append(dict(msa=name,kind='failed',error=traceback.format_exc()));(OUT/'events.json').write_text(json.dumps(events,indent=2));raise
    (OUT/'events.json').write_text(json.dumps(events,indent=2))
print('FINISHED',len(selected),'MSAs',flush=True)
