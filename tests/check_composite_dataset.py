"""rgb-visibility-v1: label coverage, test exclusion, split hygiene, composite label rules and balance."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
out = root / 'data/windows/composite-v1'
read = lambda p: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
windows = {w['window_id']: w for w in read(root / 'data/windows/manifest.jsonl') if w['duration_tick'] == 120}
labels = read(out / 'window-labels.jsonl')
composites = read(out / 'composite-manifest.jsonl')
summary = json.loads((out / 'summary.json').read_text())

# Outputs are the exact files described by summary.json.
for name, digest in summary['outputs_sha256'].items():
    assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name

# Every 1 s development window is labelled exactly once and keeps its role and presence label.
assert len(labels) == len(windows) == len({r['window_id'] for r in labels}) == 8214
for r in labels:
    w = windows[r['window_id']]
    assert (r['role'], r['presence_label'], r['clip_id']) == (w['role'], w['label'], w['clip_id'])
    assert r['rgb_visibility'] in ('visible', 'not_visible', 'ambiguous')
    assert r['use_for_rgb_visibility_loss'] == (r['rgb_visibility'] != 'ambiguous')
    if r['presence_label'] == 0:
        assert r['rgb_visibility'] == 'not_visible'

# No run0 observed-test clip appears anywhere.
with (root / 'data/audit/trident/curated-evaluation-runs.csv').open() as stream:
    test_clips = {r['clip_id'] for r in csv.DictReader(stream) if int(r['run']) == 0 and r['role'] == 'test'}
assert len(test_clips) == 53
used = {r['clip_id'] for r in labels} | {c['rgb_clip_id'] for c in composites} | {c['rf_audio_clip_id'] for c in composites}
assert not used & test_clips

# Composite label rules, split hygiene and no ambiguous RGB.
by_id = {r['window_id']: r for r in labels}
rules = {'A': ('not_visible', 1, 0, 0), 'B': ('not_visible', 0, 1, 1), 'C': ('visible', 1, 0, 1)}
for c in composites:
    rgb, sensors = by_id[c['rgb_window_id']], by_id[c['rf_audio_window_id']]
    visibility, rgb_presence, sensor_presence, label = rules[c['composite_id'].split(':')[2]]
    assert (rgb['rgb_visibility'], rgb['presence_label'], sensors['presence_label'], c['presence_label']) == (visibility, rgb_presence, sensor_presence, label), c['composite_id']
    assert rgb['role'] == sensors['role'] == c['role'] and rgb['clip_id'] != sensors['clip_id']
    assert c['use'] == ('train' if c['role'] == 'train' else 'diagnostic_only')

# Sensor mixing must not predict the label: composites are 50/50 within each role.
for role in ('train', 'validation'):
    counts = Counter(c['presence_label'] for c in composites if c['role'] == role)
    assert counts[0] == counts[1] > 0, (role, counts)
    # Every not-visible drone window is used exactly once as composite-negative RGB.
    negatives = Counter(c['rgb_window_id'] for c in composites if c['role'] == role and c['presence_label'] == 0)
    expected = {r['window_id'] for r in labels if r['role'] == role and r['presence_label'] == 1 and r['rgb_visibility'] == 'not_visible'}
    assert set(negatives) == expected and set(negatives.values()) == {1}
    # Background RF/audio reuse is even (max - min <= 1) across the background pool.
    reuse = Counter(c['rf_audio_window_id'] for c in composites if c['role'] == role and c['rf_audio_presence'] == 0)
    assert max(reuse.values()) - min(reuse.values()) <= 1, (role, max(reuse.values()), min(reuse.values()))

print(f"PASS: {len(labels)} window labels, {len(composites)} composites; no observed-test clips; rules, splits and 50/50 balance hold")
