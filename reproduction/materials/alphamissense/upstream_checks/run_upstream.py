"""Execute recovered original AlphaMissense feature functions with path-only changes.
Original 16-thread MMseqs parameters retained; cached original calibration clusters
or newly generated identical-MSA clusters are reused between axes and PSSM counts.
No model inference, samples, axis thresholds or scientific outcomes are changed.
"""
from pathlib import Path
import argparse,subprocess,json,time,hashlib,shutil,traceback,resource,zipfile,os,tarfile,platform
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--msa',nargs='*');ap.add_argument('--all',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--bundle-root',type=Path,default=Path(__file__).resolve().parent.parent);ap.add_argument('--output',type=Path);args=ap.parse_args()
BUNDLE=args.bundle_root.resolve();SOURCE=BUNDLE/'upstream_checks/code';PG=BUNDLE/'am-primary/inputs/proteingym'
OUT=(args.output or BUNDLE/'upstream_checks/rebuilt').resolve();OUT.mkdir(exist_ok=True,parents=True)
for sub in ['clusters','results','logs']:(OUT/sub).mkdir(exist_ok=True)
MM=str(BUNDLE/'upstream_checks/bin/mmseqs')
if platform.system()!='Linux' or platform.machine() not in ['x86_64','AMD64']:
    raise RuntimeError('The preserved MMseqs executable requires Linux x86-64. Run this upstream stage there; the checked downstream notebook can run on macOS.')
assert hashlib.sha256(Path(MM).read_bytes()).hexdigest()=='c58e5b422092544aac95646828eb563cee32ca609c260c50a195bb560d24f2e6'
ARCH=BUNDLE/'am-opus-r2/inputs/raw_references.tar.gz'
expected=json.loads((BUNDLE/'am-opus-r2/INPUT_MANIFEST.json').read_text())
archive_hash=next(x['sha256'] for x in expected if x['path']=='inputs/raw_references.tar.gz')
h=hashlib.sha256()
with ARCH.open('rb') as f:
    for chunk in iter(lambda:f.read(16777216),b''):h.update(chunk)
assert h.hexdigest()==archive_hash
MSADIR=OUT/'raw/msa';MSADIR.mkdir(exist_ok=True,parents=True)
with tarfile.open(ARCH,'r|gz') as tar:
    for member in tar:
        if not member.isfile() or not member.name.startswith('msa/'):continue
        target=MSADIR/Path(member.name).name
        if not target.exists():
            with tar.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
for item in json.loads((BUNDLE/'upstream_checks/opus_r2_msa_member_hashes.json').read_text()):
    f=MSADIR/item['filename'];assert f.stat().st_size==item['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],item['filename']
# I/O adaptation only: selected members were proven byte-identical to the
# original primary zip. The original numerical functions consume those bytes.
class MSAArchive:
    def namelist(self):return [p.name for p in MSADIR.glob('*.a2m')]
    def open(self,name):return (MSADIR/name).open('rb')
    def getinfo(self,name):
        from types import SimpleNamespace
        return SimpleNamespace(file_size=(MSADIR/name).stat().st_size)
current='';events=json.loads((OUT/'events.json').read_text()) if (OUT/'events.json').exists() else []
NATIVE_RUN=subprocess.run
# Hook the external command, preserving all original arguments. Record resources
# and keep verified clusters for the original PSSM function's second use.
def cluster_run(command,**kwargs):
    if not command or command[0]!=MM:return NATIVE_RUN(command,**kwargs)
    res=Path(command[3]);saved=OUT/'clusters'/(current+'.clu_cluster.tsv')
    original=BUNDLE/'upstream_checks/original_calibration'/(current+'.clu_cluster.tsv')
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
    source=source.replace('mzip=zipfile.ZipFile(f"{OUT}/DMS_msa_files_v1.3.zip")','mzip=MSAArchive()')
    env={'MSAArchive':MSAArchive};exec(compile(source,str(path),'exec'),env)
    return env
axes=load_original(SOURCE/'build_axes.py','# process distinct MSAs','clu_work')
pssm=load_original(SOURCE/'build_pssm.py','distinct=sorted','clu_work_pssm')
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
