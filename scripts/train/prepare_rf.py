"""Decode train/validation RF images once, preserving colours and orientation."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=json.loads((a.root/'data/trident/local-presence-manifest.json').read_text())
with (a.root/'data/audit/trident/curated-evaluation-runs.csv').open() as f:
    selected={r['clip_id'] for r in csv.DictReader(f) if r['run']=='0' and r['role'] in ('train','validation')}
a.output.mkdir(parents=True,exist_ok=True)
def prepare(r):
    paths=sorted(r['rf_paths'],key=lambda x:int(re.search(r'_frame_(\d+)',x)[1]))
    assert len(paths)==40
    arrays=[];source_hash=hashlib.sha256()
    for rel in paths:
        path=a.root/rel;source_hash.update(path.read_bytes())
        with Image.open(path) as im:
            arrays.append(np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)).transpose(2,0,1))
    name=hashlib.sha256(r['clip_id'].encode()).hexdigest()+'.npy'
    np.save(a.output/name,np.stack(arrays))
    return r['clip_id'],name,source_hash.hexdigest()
with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(prepare,[r for r in rows if r['clip_id'] in selected]))
index=dict(shape=[40,3,224,224],dtype='uint8',preprocessing='RGB decode, bilinear224, /255 at load; no fitted normalization',clips={i:n for i,n,h in results},source_sha256={i:h for i,n,h in results})
(a.output/'index.json').write_text(json.dumps(index,indent=2))
print('Cached train/validation clips:',len(results))
