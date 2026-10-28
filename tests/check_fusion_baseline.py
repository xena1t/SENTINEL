"""Verify availability behavior and classifier/MLLM latent compatibility."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from drone_presence.fusion_baseline import FusionClassifier,SENSOR_MASKS
from drone_presence.model import SensorFusion

torch.set_num_threads(4);torch.manual_seed(4422)
model=FusionClassifier();streams=[torch.randn(2,128,4096),torch.randn(2,40,128),torch.randn(2,2,1024)]
for mask in SENSOR_MASKS:
    available=mask.expand(2,-1)
    poisoned=[x if mask[i] else torch.full_like(x,float('nan')) for i,x in enumerate(streams)]
    torch.testing.assert_close(model(streams,available),model(poisoned,available),rtol=0,atol=0)
try:model(streams,torch.zeros((2,3),dtype=torch.bool))
except ValueError:pass
else:raise AssertionError('All missing not rejected')
model(streams,torch.ones((2,3),dtype=torch.bool)).sum().backward()
for projection in model.fusion.projections:assert projection.weight.grad.abs().sum()>0
adapter=SensorFusion(4096,4096)
result=adapter.load_state_dict(model.fusion.state_dict(),strict=False)
assert not result.unexpected_keys and all(k.startswith('connector.') for k in result.missing_keys)
with torch.no_grad():
    mask=torch.ones((2,3),dtype=torch.bool)
    torch.testing.assert_close(adapter.encode(streams,mask),model.fusion.encode(streams,mask))
    assert adapter(streams,mask).shape==(2,16,4096)
print('PASS: all 7 masks, hidden NaN isolation, all-missing guard, gradients, adapter weight transfer')
