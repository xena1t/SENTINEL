"""One real balanced batch through cached pretrained features and fusion backward."""
import hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from drone_presence.rf_data import partition_records
from drone_presence.fusion_baseline import FeatureClips,FusionClassifier
torch.manual_seed(4422)
root,features=map(Path,sys.argv[1:3]);parts=partition_records(root)
rows=[next(r for r in parts['train'] if r['label']==label) for label in (0,1)]
ds=FeatureClips(features,rows);samples=[ds[i] for i in range(2)]
streams=[torch.stack([s[0][i] for s in samples]).cuda() for i in range(3)];labels=torch.stack([s[1] for s in samples]).cuda()
model=FusionClassifier().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
available=torch.ones((2,3),dtype=torch.bool,device='cuda');before=model.head.weight.detach().clone()
loss=torch.nn.functional.binary_cross_entropy_with_logits(model(streams,available),labels);loss.backward()
assert torch.isfinite(loss)
for p in model.parameters():
 if p.grad is not None:assert torch.isfinite(p.grad).all()
for projection in model.fusion.projections:assert projection.weight.grad.abs().sum()>0
optimizer.step();assert not torch.equal(before,model.head.weight)
report=dict(status='PASS',feature_index_sha256=hashlib.sha256((features/'index.json').read_bytes()).hexdigest(),loss=float(loss.detach()),clip_ids=[r['clip_id'] for r in rows],feature_shapes=[list(x.shape) for x in streams],note='One optimization step on real frozen features, not validation accuracy')
(features/'smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
