import urllib.request, urllib.parse, json, time
genes=[l.strip() for l in open('spectra_run/artifacts/replogle_genes.txt') if l.strip()]
# STRING network API: submit all genes, get internal edges. Use required_score=0 to get all, filter later.
def string_net(gl, score=150):
    data=urllib.parse.urlencode({'identifiers':'\r'.join(gl),'species':9606,'required_score':score,'caller_identity':'spectra_audit'}).encode()
    req=urllib.request.Request('https://string-db.org/api/tsv-no-header/network',data=data,headers={'User-Agent':'spectra'})
    return urllib.request.urlopen(req,timeout=300).read().decode()

# STRING may cap identifiers; try full, fallback to note
try:
    txt=string_net(genes,150)
    open('spectra_run/artifacts/string_network.tsv','w').write(txt)
    nlines=txt.count('\n')
    print('STRING network edges (score>=150):',nlines,flush=True)
except Exception as e:
    print('STRING full failed:',str(e)[:200],flush=True)
