"""rgb-visibility-v1 adapter: RGB routes, RF/audio provenance, labels, splits, bounds, padding, masks, invalid pairs."""
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
sys.path.insert(0, str(root / 'src'))
from drone_presence.composite_windows import (load_samples, partition, sampling_weights, expected_positive_rate,
                                              clap_input, WindowFeatures, AUDIO_SAMPLES)
from drone_presence.fusion_baseline import FusionClassifier, SENSOR_MASKS
from drone_presence.model import SensorFusion

samples = load_samples(root)
train, val, diagnostic = partition(samples)
windows = {w['window_id']: w for w in map(json.loads, (root / 'data/windows/manifest.jsonl').read_text().splitlines())}
sources = {s['clip_id']: s for s in json.loads((root / 'data/windows/sources.json').read_text())}

# Counts and split isolation: real selection rows and diagnostic composites never mix.
assert Counter(s['kind'] for s in samples) == {'real': 8214, 'composite': 2918}
assert (len(train), len(val), len(diagnostic)) == (6253 + 2108, 1961, 810)
assert all(s['kind'] == 'real' for s in val) and all(s['use'] == 'diagnostic_only' for s in diagnostic)
assert all(s['role'] == 'train' for s in train)
assert Counter(s['slice'] for s in val) == {'background': 888, 'visible_positive': 446,
                                           'not_visible_positive': 405, 'ambiguous_positive': 222}

# Both RGB routes and RF/audio provenance.
for s in samples:
    rgb, sensor = windows[s['rgb_window_id']], windows[s['rf_audio_window_id']]
    assert s['rgb_paths'] == [sources[rgb['clip_id']]['rgb_files'][i - 1]['path'] for i in rgb['rgb_indices']]
    assert s['rf_indices'] == sensor['rf_indices'] and s['audio_path'] == sources[sensor['clip_id']]['audio']['path']
    assert s['audio_end_sample'] - s['audio_start_sample'] == AUDIO_SAMPLES
    if s['kind'] == 'real':
        assert s['rgb_window_id'] == s['rf_audio_window_id'] and s['presence_label'] == rgb['label']
    else:
        assert s['rgb_clip_id'] != s['rf_audio_clip_id'] and rgb['role'] == sensor['role'] == s['role']
        assert s['presence_label'] == {'A': 0, 'B': 1, 'C': 1}[s['composite_type']]
        assert s['rgb_visibility'] != 'ambiguous'
# Ambiguous real windows keep presence supervision.
assert any(s['slice'] == 'ambiguous_positive' and s['presence_label'] == 1 for s in train)

# Sampler: half composites, A:B:C = 2:1:1, so composites stay 50/50 positive.
weights = sampling_weights(train)
mass = Counter()
for s, w in zip(train, weights.tolist()):
    mass['real' if s['kind'] == 'real' else s['composite_type']] += w
assert abs(mass['real'] - .5) < 1e-9 and abs(mass['A'] - .25) < 1e-9 and abs(mass['B'] - .125) < 1e-9 and abs(mass['C'] - .125) < 1e-9
composite_positive = (mass['B'] + mass['C']) / (mass['A'] + mass['B'] + mass['C'])
assert abs(composite_positive - .5) < 1e-9 and 0 < expected_positive_rate(train, weights) < 1

# CLAP input: exact crop then zero padding; no neighbouring samples.
signal = np.arange(441000, dtype=np.float32)
padded = clap_input(signal, 44100)
assert padded.shape == (308700,) and np.array_equal(padded[:AUDIO_SAMPLES], signal[44100:88200])
assert not padded[AUDIO_SAMPLES:].any()
for bad in (-1, 441000 - AUDIO_SAMPLES + 1):
    try:
        clap_input(signal, bad)
        raise AssertionError('Out-of-range crop accepted')
    except ValueError:
        pass

# Invalid pairs are rejected: tamper with a copy and re-pin its hash so only the rule check can fail.
def rejected(mutate):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for relative in ('data/windows/manifest.jsonl', 'data/windows/sources.json', 'data/audit/trident/curated-evaluation-runs.csv',
                         'data/windows/composite-v1/window-labels.jsonl', 'data/windows/composite-v1/summary.json'):
            (tmp / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(root / relative, tmp / relative)
        directory = tmp / 'data/windows/composite-v1'
        rows = [json.loads(line) for line in (root / 'data/windows/composite-v1/composite-manifest.jsonl').read_text().splitlines()]
        mutate(rows)
        (directory / 'composite-manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        summary = json.loads((directory / 'summary.json').read_text())
        summary['outputs_sha256']['composite-manifest.jsonl'] = hashlib.sha256((directory / 'composite-manifest.jsonl').read_bytes()).hexdigest()
        (directory / 'summary.json').write_text(json.dumps(summary))
        try:
            load_samples(tmp)
        except ValueError:
            return True
        return False

labels = [json.loads(line) for line in (root / 'data/windows/composite-v1/window-labels.jsonl').read_text().splitlines()]
validation_background = next(r['window_id'] for r in labels if r['role'] == 'validation' and r['presence_label'] == 0)
ambiguous = next(r['window_id'] for r in labels if r['role'] == 'train' and r['rgb_visibility'] == 'ambiguous')
assert rejected(lambda rows: rows[0].update(rf_audio_window_id=validation_background)), 'cross-role pair'
assert rejected(lambda rows: rows[0].update(rgb_window_id=ambiguous)), 'ambiguous RGB'
assert rejected(lambda rows: rows[0].update(rf_audio_window_id='trident-window-v2.1:missing:t0000-0120@120')), 'missing source'
assert rejected(lambda rows: rows[0].update(presence_label=1)), 'wrong label'
assert rejected(lambda rows: rows[0].update(rf_audio_clip_id=rows[0]['rgb_clip_id'])), 'same recording'
assert not rejected(lambda rows: None), 'unmodified copy must load'

# Feature assembly from synthetic per-clip arrays: RGB from the RGB clip, RF and audio from the sensor clip.
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    chosen = [next(s for s in train if s['composite_type'] == t) for t in 'ABC'] + [next(s for s in val if s['slice'] == 'background')]
    clips = {c for s in chosen for c in (s['rgb_clip_id'], s['rf_audio_clip_id'])}
    index = {'complete': True, 'clips': {}}
    for n, clip in enumerate(sorted(clips)):
        folder = tmp / f'c{n}'; folder.mkdir()
        rgb_indices = sorted({i for s in chosen if s['rgb_clip_id'] == clip for i in s['rgb_indices']})
        starts = sorted({s['audio_start_sample'] for s in chosen if s['rf_audio_clip_id'] == clip})
        np.save(folder / 'rgb.npy', np.stack([np.full((16, 4096), 1000 * n + i, np.float32) for i in rgb_indices]) if rgb_indices else np.zeros((0, 16, 4096), np.float32))
        np.save(folder / 'rf.npy', np.stack([np.full(128, 1000 * n + i, np.float32) for i in range(1, 41)]))
        np.save(folder / 'audio.npy', np.stack([np.full((1, 1024), start, np.float32) for start in starts]) if starts else np.zeros((0, 1, 1024), np.float32))
        (folder / 'meta.json').write_text(json.dumps({'clip_id': clip, 'rgb_indices': rgb_indices, 'audio_starts': starts}))
        index['clips'][clip] = folder.name
    (tmp / 'index.json').write_text(json.dumps(index))
    names = {clip: int(folder[1:]) for clip, folder in index['clips'].items()}
    dataset = WindowFeatures(tmp, chosen)
    for i, s in enumerate(chosen):
        (rgb, rf, audio), label, sample_id = dataset[i]
        assert sample_id == s['sample_id'] and int(label) == s['presence_label']
        assert rgb.shape == (128, 4096) and rf.shape == (4, 128) and audio.shape == (1, 1024)
        assert rgb[::16, 0].tolist() == [1000 * names[s['rgb_clip_id']] + j for j in s['rgb_indices']]
        assert rf[:, 0].tolist() == [1000 * names[s['rf_audio_clip_id']] + j for j in s['rf_indices']]
        assert audio[0, 0].item() == s['audio_start_sample']
    del dataset, rgb, rf, audio  # close memory maps before the folder is removed

# Original fusion design with one-second token counts: masks, hidden NaN suppression, all-missing rejection.
torch.manual_seed(0)
model = FusionClassifier().eval()
streams = [torch.randn(2, 128, 4096), torch.randn(2, 4, 128), torch.randn(2, 1, 1024)]
for mask in SENSOR_MASKS:
    available = mask.expand(2, -1).clone()
    poisoned = [x if m else torch.full_like(x, float('nan')) for x, m in zip(streams, mask)]
    with torch.no_grad():
        clean, hidden = model(streams, available), model(poisoned, available)
    assert torch.isfinite(hidden).all() and torch.allclose(clean, hidden)
try:
    model(streams, torch.zeros((2, 3), dtype=torch.bool))
    raise AssertionError('All-missing row accepted')
except ValueError:
    pass
assert SensorFusion(4096, 4096).connector[-1].out_features == 4096
print(f'PASS: {len(samples)} samples ({len(train)} train, {len(val)} real validation, {len(diagnostic)} diagnostic composites); '
      'routes, provenance, splits, bounds, padding, masks and invalid-pair rejection verified')
