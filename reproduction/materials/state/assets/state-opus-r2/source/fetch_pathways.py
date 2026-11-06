import urllib.request, urllib.parse, json, time
def mygene_batch(gl, fields):
    data=urllib.parse.urlencode({'q':','.join(gl),'scopes':'symbol','fields':fields,'species':'human'}).encode()
    req=urllib.request.Request('https://mygene.info/v3/query',data=data,headers={'User-Agent':'spectra','Content-Type':'application/x-www-form-urlencoded'})
    return json.load(urllib.request.urlopen(req,timeout=120))

def fetch(genes, fields):
    out={}
    for i in range(0,len(genes),900):
        b=genes[i:i+900]
        for a in range(3):
            try: res=mygene_batch(b,fields); break
            except Exception as e: print('retry',i,e); time.sleep(5)
        for rec in res:
            out.setdefault(rec.get('query'),rec)
        print('done',i+len(b),flush=True); time.sleep(1)
    return out

# 1) Reactome + KEGG pathway membership for Replogle genes
rg=[l.strip() for l in open('spectra_run/artifacts/replogle_genes.txt') if l.strip()]
res=fetch(rg,'symbol,pathway.reactome.id,pathway.kegg.id')
pw={}
for g,rec in res.items():
    p=rec.get('pathway',{}) if isinstance(rec.get('pathway',{}),dict) else {}
    def ids(k):
        v=p.get(k,[])
        if isinstance(v,dict): v=[v]
        return [x['id'] for x in v if isinstance(x,dict) and 'id' in x]
    pw[g]={'reactome':ids('reactome'),'kegg':ids('kegg')}
json.dump(pw,open('spectra_run/artifacts/replogle_pathways.json','w'))
nr=sum(1 for v in pw.values() if v['reactome'])
print('Replogle pathways: genes with reactome',nr,'/',len(pw),flush=True)

# 2) GO-BP for Tahoe target genes
tg=[l.strip() for l in open('spectra_run/artifacts/tahoe_target_genes.txt') if l.strip()]
res2=fetch(tg,'symbol,go.BP.id')
go={}
for g,rec in res2.items():
    gg=rec.get('go',{}); bp=[]
    if isinstance(gg,dict):
        v=gg.get('BP',[]);
        if isinstance(v,dict): v=[v]
        bp=[x['id'] for x in v if isinstance(x,dict) and 'id' in x]
    go[g]=sorted(set(bp))
json.dump(go,open('spectra_run/artifacts/tahoe_target_go.json','w'))
print('Tahoe target GO: genes with BP',sum(1 for v in go.values() if v),'/',len(go),flush=True)
