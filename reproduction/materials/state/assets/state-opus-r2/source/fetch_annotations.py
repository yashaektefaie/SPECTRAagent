import urllib.request, urllib.parse, json, time, sys
genes=[l.strip() for l in open('spectra_run/artifacts/replogle_genes.txt') if l.strip()]
print('genes',len(genes),flush=True)

# ---- 1. GO BP + MF via mygene batch POST ----
def mygene_batch(gl):
    data=urllib.parse.urlencode({'q':','.join(gl),'scopes':'symbol','fields':'symbol,go.BP.id,go.MF.id,go.CC.id,type_of_gene','species':'human'}).encode()
    req=urllib.request.Request('https://mygene.info/v3/query',data=data,headers={'User-Agent':'spectra','Content-Type':'application/x-www-form-urlencoded'})
    return json.load(urllib.request.urlopen(req,timeout=120))

go={}
for i in range(0,len(genes),900):
    b=genes[i:i+900]
    for attempt in range(3):
        try:
            res=mygene_batch(b); break
        except Exception as e:
            print('retry',i,e); time.sleep(5)
    for rec in res:
        sym=rec.get('query')
        def terms(k):
            g=rec.get('go',{})
            if not isinstance(g,dict): return []
            v=g.get(k,[])
            if isinstance(v,dict): v=[v]
            return [x['id'] for x in v if isinstance(x,dict) and 'id' in x]
        if sym not in go: go[sym]={'BP':set(),'MF':set(),'CC':set()}
        go[sym]['BP']|=set(terms('BP')); go[sym]['MF']|=set(terms('MF')); go[sym]['CC']|=set(terms('CC'))
    print('mygene done',i+len(b),flush=True); time.sleep(1)

json.dump({k:{kk:sorted(vv) for kk,vv in v.items()} for k,v in go.items()}, open('spectra_run/artifacts/go_terms.json','w'))
nbp=sum(1 for v in go.values() if v['BP'])
print('GO fetched: genes with BP terms',nbp,'/',len(go),flush=True)
