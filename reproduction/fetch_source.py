"""Download one explicitly identified public source; never reads credentials."""
import argparse,hashlib,json,os,tempfile,urllib.request
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--source')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--sha256',help='Expected SHA-256 of the original required file, when known')
    args=parser.parse_args()
    entries=json.loads((Path(__file__).parent/'sources.json').read_text())
    if args.list:
        for item in entries:print(item['id'],item['kind'],item['url'],sep='\t')
        return
    if not args.source or not args.output:parser.error('--source and --output are required')
    item=next((s for s in entries if s['id']==args.source),None)
    if item is None:parser.error('Unknown source ID')
    if item['kind']!='download':parser.error('This is a repository/metadata endpoint, not a data-file download; inspect its URL directly')
    if args.output.exists():parser.error('Output already exists')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.spectra-download-',dir=args.output.parent)
    try:
        request=urllib.request.Request(item['url'],headers={'User-Agent':'SPECTRA-source-fetch/1.0'})
        with urllib.request.urlopen(request,timeout=60) as response,os.fdopen(fd,'wb') as out:
            if response.status!=200 or any(s in response.geturl().lower() for s in ['/login','/signin']):
                raise RuntimeError('An anonymous data response was not returned')
            if 'text/html' in response.headers.get('content-type','').lower():
                raise RuntimeError('Received an HTML page instead of the requested data')
            digest=hashlib.sha256()
            while chunk:=response.read(1024*1024):
                out.write(chunk);digest.update(chunk)
        if args.sha256 and digest.hexdigest()!=args.sha256:
            raise RuntimeError('Downloaded bytes do not match the required original SHA-256')
        os.link(tmp,args.output) # no overwrite if another process created the output
        print(digest.hexdigest(),args.output)
        print(item['qualification'])
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

if __name__=='__main__':main()
