import urllib.request, urllib.parse, json, time
M={'4-1BBL':'TNFSF9','APRIL':'TNFSF13','BAFF':'TNFSF13B','C3a':'C3','C5a':'C5',
'CD27L':'CD70','CD30L':'TNFSF8','CD40L':'CD40LG','CT-1':'CTF1','Decorin':'DCN',
'EGF':'EGF','EPO':'EPO','FGF-beta':'FGF2','FLT3L':'FLT3LG','FasL':'FASLG','G-CSF':'CSF3',
'GDNF':'GDNF','GITRL':'TNFSF18','GM-CSF':'CSF2','HGF':'HGF','IFN-alpha1':'IFNA1',
'IFN-beta':'IFNB1','IFN-epsilon':'IFNE','IFN-gamma':'IFNG','IFN-lambda1':'IFNL1',
'IFN-lambda2':'IFNL2','IFN-lambda3':'IFNL3','IFN-omega':'IFNW1','IGF-1':'IGF1',
'IL-1-alpha':'IL1A','IL-1-beta':'IL1B','IL-10':'IL10','IL-11':'IL11','IL-12':'IL12A',
'IL-13':'IL13','IL-15':'IL15','IL-16':'IL16','IL-17A':'IL17A','IL-17B':'IL17B',
'IL-17C':'IL17C','IL-17D':'IL17D','IL-17E':'IL25','IL-17F':'IL17F','IL-18':'IL18',
'IL-19':'IL19','IL-1Ra':'IL1RN','IL-2':'IL2','IL-20':'IL20','IL-21':'IL21','IL-22':'IL22',
'IL-23':'IL23A','IL-24':'IL24','IL-26':'IL26','IL-27':'IL27','IL-3':'IL3','IL-31':'IL31',
'IL-32-beta':'IL32','IL-33':'IL33','IL-34':'IL34','IL-35':'EBI3','IL-36-alpha':'IL36A',
'IL-36Ra':'IL36RN','IL-4':'IL4','IL-5':'IL5','IL-6':'IL6','IL-7':'IL7','IL-8':'CXCL8',
'IL-9':'IL9','LIF':'LIF','LIGHT':'TNFSF14','LT-alpha1-beta2':'LTA','LT-alpha2-beta1':'LTB',
'Leptin':'LEP','M-CSF':'CSF1','Noggin':'NOG','OSM':'OSM','OX40L':'TNFSF4','PRL':'PRL',
'PSPN':'PSPN','RANKL':'TNFSF11','SCF':'KITLG','TGF-beta1':'TGFB1','TL1A':'TNFSF15',
'TNF-alpha':'TNF','TPO':'THPO','TRAIL':'TNFSF10','TSLP':'TSLP','TWEAK':'TNFSF12','VEGF':'VEGFA'}
json.dump(M,open('spectra_run/artifacts/parse_ligand_map.json','w'))
genes=sorted(set(M.values()))
data=urllib.parse.urlencode({'q':','.join(genes),'scopes':'symbol','fields':'symbol,go.BP.id','species':'human'}).encode()
req=urllib.request.Request('https://mygene.info/v3/query',data=data,headers={'User-Agent':'spectra','Content-Type':'application/x-www-form-urlencoded'})
res=json.load(urllib.request.urlopen(req,timeout=120))
go={}
for rec in res:
    gg=rec.get('go',{}); bp=[]
    if isinstance(gg,dict):
        v=gg.get('BP',[]);
        if isinstance(v,dict): v=[v]
        bp=[x['id'] for x in v if isinstance(x,dict) and 'id' in x]
    go[rec.get('query')]=sorted(set(bp))
json.dump(go,open('spectra_run/artifacts/parse_gene_go.json','w'))
print('mapped ligands:',len(M),' genes with GO-BP:',sum(1 for v in go.values() if v))
