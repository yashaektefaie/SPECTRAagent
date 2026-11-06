"""Map recomputed arrays to original assayed positions using original formatting.
No scores or historical expected tables participate in construction.
"""
from pathlib import Path
import csv,re,json,hashlib,argparse
import numpy as np
AA='ACDEFGHIKLMNPQRSTVWY'
def materialize(bundle_root=None,result_dir=None,output_dir=None):
    root=Path(bundle_root or Path(__file__).resolve().parent.parent)
    result=Path(result_dir or root/'upstream_checks/results')
    output=Path(output_dir or root/'upstream_checks/reconstructed');output.mkdir(exist_ok=True,parents=True)
    pg=root/'am-primary/inputs/proteingym';assays=[]
    with (pg/'kept_assays_verified.tsv').open() as fh:
        for r in csv.DictReader(fh,delimiter='\t'):
            pos={}
            with (pg/'dms_final_73'/r['DMS_filename']).open() as f:
                for row in csv.DictReader(f):
                    match=re.fullmatch(r'([A-Z])(\d+)([A-Z])',row['mutant'])
                    if match:p=int(match[2]);pos[p]=pos.get(p,0)+1
            assays.append(dict(id=r['DMS_id'],acc=r['resolved_accession'],gene=r['DMS_id'].split('_')[0],fam='Tsuboyama_domain' if 'Tsuboyama' in r['DMS_id'] else 'other',msa=r['MSA_filename'].strip(),pos=pos))
    features={}
    for name in sorted({a['msa'] for a in assays}):
        z=np.load(result/(name+'.npz'));features[name]=dict(neff=z['neff'],occupancy=z['occupancy'],counts=z['counts'],positions=dict(zip(z['positions'],z['match_columns'])),lowcov=set(z['lowcov']))
    n_axis=n_counts=0
    with (output/'axis3_pathC_percolumn_FROZEN.tsv').open('w') as out:
        out.write('assay\taccession\tgene\tassay_family\ttarget_pos\tneff_theta02\toccupancy\tlow_cov_insert\tdepth_bin\tn_variants\n')
        for a in assays:
            z=features[a['msa']]
            for p,nv in a['pos'].items():
                if p in z['positions']:
                    col=z['positions'][p];nf=float(z['neff'][col]);oc=float(z['occupancy'][col]);lc=0
                elif p in z['lowcov']:nf=oc=float('nan');lc=1
                else:continue
                db='' if lc else sum(nf>=edge for edge in [2,10,50,200])
                out.write(f"{a['id']}\t{a['acc']}\t{a['gene']}\t{a['fam']}\t{p}\t{'' if lc else round(nf,3)}\t{'' if lc else round(oc,4)}\t{lc}\t{db}\t{nv}\n");n_axis+=1
    with (output/'pssm_counts.tsv').open('w') as out:
        out.write('assay\ttarget_pos\tneff_col\t'+'\t'.join(AA)+'\n')
        for name in sorted(features):
            z=features[name]
            for a in assays:
                if a['msa']!=name:continue
                for p in sorted(a['pos']):
                    if p not in z['positions']:continue
                    counts=z['counts'][z['positions'][p]]
                    out.write(f"{a['id']}\t{p}\t{counts.sum():.4f}\t"+'\t'.join(f'{v:.4f}' for v in counts)+'\n');n_counts+=1
    record=dict(msas=len(features),axis_rows=n_axis,pssm_rows=n_counts,outputs=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(output.glob('*.tsv'))])
    (output/'MANIFEST.json').write_text(json.dumps(record,indent=2));return record
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result-dir',type=Path);p.add_argument('--output-dir',type=Path);a=p.parse_args();print(json.dumps(materialize(result_dir=a.result_dir,output_dir=a.output_dir),indent=2))
