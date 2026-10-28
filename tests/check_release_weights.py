import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from drone_presence.fusion_baseline import FusionClassifier, SENSOR_MASKS
from drone_presence.model import RFEncoder, SensorFusion
from drone_presence.pipeline import load_checkpoint

torch.set_num_threads(4)
torch.manual_seed(4422)
protocol = json.loads((ROOT / 'research/test-v2/protocol.json').read_text())
weights = {}
for name, expected in protocol['checkpoints'].items():
    path = ROOT / 'checkpoints/version-2' / f'{name}.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name
    weights[name] = load_checkpoint(path)
    assert weights[name]['epoch'] == expected['epoch'], name

rf = RFEncoder().eval().requires_grad_(False)
rf.load_state_dict(weights['rf-cnn-best']['encoder'], strict=True)
fusion = FusionClassifier().eval().requires_grad_(False)
fusion.load_state_dict(weights['fusion-best']['state'], strict=True)
adapter = SensorFusion(4096, 4096).eval().requires_grad_(False)
adapter.load_state_dict(weights['qwen-dropout-connector-best']['adapter'], strict=True)
assert all(torch.equal(value, adapter.state_dict()[name.removeprefix('fusion.')])
           for name, value in fusion.state_dict().items() if name.startswith('fusion.'))

with torch.inference_mode():
    features = [torch.randn(1, 128, 4096), rf(torch.rand(4, 3, 224, 224)).unsqueeze(0),
                torch.randn(1, 1, 1024)]
    assert features[1].shape == (1, 4, 128)
    for mask in SENSOR_MASKS:
        available = mask.unsqueeze(0)
        poisoned = [stream if mask[index] else torch.full_like(stream, float('nan'))
                    for index, stream in enumerate(features)]
        probability = fusion(features, available).sigmoid()
        tokens = adapter(features, available)
        assert torch.isfinite(probability).all() and (probability >= 0).all() and (probability <= 1).all()
        assert tokens.shape == (1, 16, 4096) and torch.isfinite(tokens).all()
        torch.testing.assert_close(probability, fusion(poisoned, available).sigmoid(), rtol=0, atol=0)
        torch.testing.assert_close(tokens, adapter(poisoned, available), rtol=0, atol=0)
print('PASS: three release checksums, selected epochs, strict loads, frozen fusion identity, RF forward and seven masks')
