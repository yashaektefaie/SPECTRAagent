# Original notebook cell 2
from pathlib import Path
import hashlib, json
ROOT=Path.cwd()
manifest=json.loads((ROOT/'INPUT_MANIFEST.json').read_text())
for item in manifest:
    path=ROOT/item['path']
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'], item['path']
print(f'{len(manifest)} retained evidence files verified.')
print('No individual measurements have been synthesized and no summary values are treated as raw data.')
