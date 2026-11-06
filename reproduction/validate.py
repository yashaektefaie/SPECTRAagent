"""Check the published record contract; does not execute scientific analyses."""
import ast,gzip,hashlib,json,re
from functools import lru_cache
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def read(path):return json.loads((ROOT/path).read_text())
def file(path):
    p=(ROOT/path).resolve()
    assert p.is_relative_to(ROOT) and p.is_file(),path
    return p
@lru_cache(None)
def digest(path):return hashlib.sha256(file(path).read_bytes()).hexdigest()

def main():
    index=read('index.json');sources=read(index['sources']);sourceids={x['id'] for x in sources}
    assert len(sourceids)==len(sources)
    for s in sources:
        assert s['url'].startswith('https://') and s['anonymous_access_observed'] is True
        assert s['http_status']==200 and s['checked_at']
        assert not re.search(r'(?:token|api_key|signature)=',s['url'],re.I)
    count=0;cells=0;ids=set();models={};samplefiles=set()
    for investigation in index['investigations']:
        for entry in investigation['analyses']:
            d=read(entry['path']);assert d['id']==entry['id'] and d['id'] not in ids;ids.add(d['id'])
            assert d['investigation']==investigation['id']
            assert not {'status','verdict','grade','pass','fail','accepted','rejected'}&d.keys()
            for name in ['dataset_split','axis_levels','model_evaluation','curve','insight']:
                assert isinstance(d[name],dict) and d[name],(entry['path'],name)
            assert set(d['dataset_split']['public_sources'])<=sourceids
            file(d['dataset_split']['input_inventory'])
            for s in d['dataset_split']['sample_tables']:
                assert digest(s['path'])==s['sha256']
                if s['path'] not in samplefiles:
                    table=json.loads(gzip.decompress(file(s['path']).read_bytes()))
                    assert len(table['rows'])==s['rows'] and table['columns']==s['columns']
                    assert all(set(row)==set(s['columns']) for row in table['rows'])
                    samplefiles.add(s['path'])
            impl=d['implementation'];assert digest(impl['notebook'])==impl['published_notebook_sha256']
            b=read(impl['notebook']);assert b['nbformat']==4
            assert all(c['cell_type']=='code' and c['execution_count'] is None and c['outputs']==[] for c in b['cells'])
            assert len(b['cells'])==impl['saved_code_cells'] and impl['saved_error_outputs']==0
            ast.parse(file(impl['script']).read_text())
            for s in read(impl['supporting_code']):assert digest(s['path'])==s['sha256']
            for s in d['model_evaluation']['fine_tuning']['comparators']:file(s['evidence'])
            cells+=len(b['cells']);count+=1;models[investigation['model']]=models.get(investigation['model'],0)+1
    assert len(index['investigations'])==28 and count==143 and cells==613
    assert models=={'ESMFold2':35,'STATE':44,'CONCH':30,'AlphaMissense':34}
    assert {d['id'] for d in read('reconstruction.json')}==ids
    print(json.dumps({'investigations':28,'analyses':count,'code_cells':cells,'public_sources':len(sources),'linked_sample_tables':len(samplefiles),'per_model':models},indent=2))

if __name__=='__main__':main()
