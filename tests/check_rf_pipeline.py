"""Real-data contract checks before the overfit experiment (run on compute node)."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/train'))
from drone_presence.rf_data import RFClips,partition_records
from train_rf import RFClassifier
root=Path(sys.argv[1]);cache=Path(sys.argv[2]);parts=partition_records(root)
index=json.loads((cache/'index.json').read_text())
assert not {r['clip_id'] for r in parts['test']} & set(index['clips'])
ds=RFClips(cache,parts['train']);x,y,c=ds[0]
assert x.shape==(40,3,224,224) and x.min()>=0 and x.max()<=1 and y.item() in (0,1)
torch.set_num_threads(2)
model=RFClassifier().eval();sample=torch.rand(2,4,3,32,32)
a=model(sample);b=model(sample[:,[3,1,0,2]])
torch.testing.assert_close(a,b)
a.sum().backward();assert model.encoder.net[0].weight.grad.abs().sum()>0
print('PASS: partition coverage/group isolation, test exclusion, real loader, clip pooling invariance, encoder gradients',flush=True)
