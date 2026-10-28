"""rgb-visibility-v1 real one-second windows and two-way composites as fusion samples."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_VERSION = 'rgb-visibility-v1'
WINDOW_TICKS = 120
AUDIO_SAMPLES = 44100
RGB_FRAMES, RF_FRAMES = 8, 4
# Composite type -> (RGB visibility, RGB window presence, RF/audio window presence, target).
COMPOSITE_RULES = {'A': ('not_visible', 1, 0, 0), 'B': ('not_visible', 0, 1, 1), 'C': ('visible', 1, 0, 1)}
SLICES = ('background', 'visible_positive', 'not_visible_positive', 'ambiguous_positive')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def real_slice(label):
    if label['presence_label'] == 0:
        return 'background'
    return {'visible': 'visible_positive', 'not_visible': 'not_visible_positive',
            'ambiguous': 'ambiguous_positive'}[label['rgb_visibility']]


def check_window(window, source):
    """One-second bounds: 8 RGB frames, 4 whole RF bins and an exact 44,100-sample crop."""
    start, end = window['start_tick'], window['end_tick']
    if window['duration_tick'] != WINDOW_TICKS or end - start != WINDOW_TICKS:
        raise ValueError(f'Not a one-second window: {window["window_id"]}')
    if len(window['rgb_indices']) != RGB_FRAMES or len(window['rf_indices']) != RF_FRAMES:
        raise ValueError(f'Unexpected frame counts: {window["window_id"]}')
    for i in window['rgb_indices']:
        if not start <= source['rgb_files'][i - 1]['nominal_timestamp_tick'] < end:
            raise ValueError(f'RGB frame outside window: {window["window_id"]}')
    for i in window['rf_indices']:
        low, high = source['rf_files'][i - 1]['nominal_support_ticks']
        if not start <= low < high <= end:
            raise ValueError(f'RF bin outside window: {window["window_id"]}')
    if window['audio_end_sample'] - window['audio_start_sample'] != AUDIO_SAMPLES:
        raise ValueError(f'Audio crop is not one second: {window["window_id"]}')


def clap_input(signal, start, length=7 * 44100):
    """Exact one-second crop first, then right zero padding; neighbouring audio never leaks in."""
    if not 0 <= start <= len(signal) - AUDIO_SAMPLES:
        raise ValueError('Audio crop outside source')
    padded = np.zeros(length, dtype=np.float32)
    padded[:AUDIO_SAMPLES] = signal[start:start + AUDIO_SAMPLES]
    return padded


def load_samples(root):
    """Real windows plus composites; RGB comes from one window, RF and audio from another."""
    root = Path(root)
    directory = root / 'data/windows/composite-v1'
    summary = json.loads((directory / 'summary.json').read_text())
    if summary['label_version'] != LABEL_VERSION:
        raise ValueError('Unexpected label version')
    for name, digest in summary['outputs_sha256'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'Composite dataset file changed: {name}')
    if sha256(root / 'data/windows/manifest.jsonl') != summary['inputs_sha256']['window_manifest']:
        raise ValueError('Window manifest differs from the one used to build the composites')
    with (root / 'data/audit/trident/curated-evaluation-runs.csv').open() as stream:
        test_clips = {r['clip_id'] for r in csv.DictReader(stream) if int(r['run']) == 0 and r['role'] == 'test'}
    sources = {s['clip_id']: s for s in json.loads((root / 'data/windows/sources.json').read_text())}
    windows = {w['window_id']: w for w in read_jsonl(root / 'data/windows/manifest.jsonl')
               if w['duration_tick'] == WINDOW_TICKS}
    labels = {r['window_id']: r for r in read_jsonl(directory / 'window-labels.jsonl')}

    def resolve(window_id):
        if window_id not in windows or window_id not in labels:
            raise ValueError(f'Missing source window: {window_id}')
        window, label = windows[window_id], labels[window_id]
        if window['clip_id'] in test_clips or window['role'] not in ('train', 'validation'):
            raise ValueError(f'Observed-test or unknown role refused: {window_id}')
        if (label['role'], label['presence_label'], label['clip_id']) != (window['role'], window['label'], window['clip_id']):
            raise ValueError(f'Label and window manifest disagree: {window_id}')
        source = sources[window['clip_id']]
        check_window(window, source)
        return window, label, source

    def sample(sample_id, kind, role, target, rgb_id, sensor_id, composite_type=None):
        rgb, rgb_label, rgb_source = resolve(rgb_id)
        sensor, sensor_label, sensor_source = resolve(sensor_id)
        if rgb['role'] != role or sensor['role'] != role:
            raise ValueError(f'Cross-role pair refused: {sample_id}')
        return {'sample_id': sample_id, 'kind': kind, 'composite_type': composite_type, 'role': role,
                'presence_label': target,
                'use': 'diagnostic_only' if kind == 'composite' and role == 'validation' else role,
                'slice': real_slice(rgb_label) if kind == 'real' else 'composite_' + composite_type,
                'rgb_window_id': rgb_id, 'rf_audio_window_id': sensor_id,
                'rgb_clip_id': rgb['clip_id'], 'rf_audio_clip_id': sensor['clip_id'],
                'groups': sorted({rgb['group_proxy'], sensor['group_proxy']}),
                'rgb_visibility': rgb_label['rgb_visibility'],
                'rgb_indices': rgb['rgb_indices'],
                'rgb_paths': [rgb_source['rgb_files'][i - 1]['path'] for i in rgb['rgb_indices']],
                'rf_indices': sensor['rf_indices'],
                'audio_path': sensor_source['audio']['path'], 'audio_sha256': sensor_source['audio']['sha256'],
                'audio_start_sample': sensor['audio_start_sample'], 'audio_end_sample': sensor['audio_end_sample']}

    samples = []
    for window_id, label in sorted(labels.items()):
        samples.append(sample(window_id, 'real', label['role'], label['presence_label'], window_id, window_id))
    for row in read_jsonl(directory / 'composite-manifest.jsonl'):
        kind = row['composite_id'].split(':')[2]
        if kind not in COMPOSITE_RULES or not row['type'].startswith(kind + '_'):
            raise ValueError(f'Unknown composite type: {row["composite_id"]}')
        visibility, rgb_presence, sensor_presence, target = COMPOSITE_RULES[kind]
        rgb_label, sensor_label = labels.get(row['rgb_window_id']), labels.get(row['rf_audio_window_id'])
        if rgb_label is None or sensor_label is None:
            raise ValueError(f'Missing composite source: {row["composite_id"]}')
        if rgb_label['rgb_visibility'] == 'ambiguous':
            raise ValueError(f'Ambiguous composite RGB refused: {row["composite_id"]}')
        if (rgb_label['rgb_visibility'], rgb_label['presence_label'], sensor_label['presence_label'],
                row['presence_label']) != (visibility, rgb_presence, sensor_presence, target):
            raise ValueError(f'Composite label rule violated: {row["composite_id"]}')
        if row['rgb_clip_id'] == row['rf_audio_clip_id']:
            raise ValueError(f'Composite must pair two recordings: {row["composite_id"]}')
        if row['use'] != ('train' if row['role'] == 'train' else 'diagnostic_only'):
            raise ValueError(f'Unexpected composite use: {row["composite_id"]}')
        samples.append(sample(row['composite_id'], 'composite', row['role'], target,
                              row['rgb_window_id'], row['rf_audio_window_id'], kind))
    if len({s['sample_id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate sample identity')
    return samples


def partition(samples):
    """Training rows, real validation (selection) and validation composites (diagnostic only)."""
    return ([s for s in samples if s['role'] == 'train'],
            [s for s in samples if s['role'] == 'validation' and s['kind'] == 'real'],
            [s for s in samples if s['role'] == 'validation' and s['kind'] == 'composite'])


def sampling_weights(rows, composite_share=.5, type_ratio=(2, 1, 1)):
    """Per-row weights: composite_share of draws are composites, split A:B:C by type_ratio."""
    counts = Counter('real' if r['kind'] == 'real' else r['composite_type'] for r in rows)
    ratio = dict(zip('ABC', type_ratio))
    if not counts['real'] or any(not counts[t] for t in 'ABC'):
        raise ValueError('Sampler needs real windows and all three composite types')
    weights = []
    for r in rows:
        if r['kind'] == 'real':
            weights.append((1 - composite_share) / counts['real'])
        else:
            weights.append(composite_share * ratio[r['composite_type']] / sum(ratio.values()) / counts[r['composite_type']])
    return torch.tensor(weights, dtype=torch.double)


def expected_positive_rate(rows, weights):
    labels = torch.tensor([r['presence_label'] for r in rows], dtype=torch.double)
    return float((weights * labels).sum() / weights.sum())


class WindowFeatures(Dataset):
    """Frozen per-clip features assembled per sample: RGB [128,4096], RF [4,128], audio [1,1024]."""

    def __init__(self, directory, samples):
        self.directory = Path(directory)
        self.samples = samples
        self.index = json.loads((self.directory / 'index.json').read_text())
        if not self.index['complete']:
            raise ValueError('Feature extraction is incomplete')
        self.arrays = {}
        for s in samples:
            for clip in (s['rgb_clip_id'], s['rf_audio_clip_id']):
                if clip not in self.index['clips']:
                    raise ValueError(f'Missing clip features: {clip}')

    def clip(self, clip_id):
        # Memory-mapped so workers read only the selected frames.
        if clip_id not in self.arrays:
            folder = self.directory / self.index['clips'][clip_id]
            meta = json.loads((folder / 'meta.json').read_text())
            self.arrays[clip_id] = dict(meta=meta, **{k: np.load(folder / f'{k}.npy', mmap_mode='r') for k in ('rgb', 'rf', 'audio')})
        return self.arrays[clip_id]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        rgb_clip, sensor_clip = self.clip(s['rgb_clip_id']), self.clip(s['rf_audio_clip_id'])
        frames = [rgb_clip['meta']['rgb_indices'].index(j) for j in s['rgb_indices']]
        rgb = torch.from_numpy(np.ascontiguousarray(rgb_clip['rgb'][frames])).reshape(RGB_FRAMES * 16, 4096)
        rf = torch.from_numpy(np.array(sensor_clip['rf'][[j - 1 for j in s['rf_indices']]], dtype=np.float32))
        crop = sensor_clip['meta']['audio_starts'].index(s['audio_start_sample'])
        audio = torch.from_numpy(np.array(sensor_clip['audio'][crop], dtype=np.float32))
        streams = [rgb.float(), rf.float(), audio.float()]
        for x, shape in zip(streams, [(128, 4096), (4, 128), (1, 1024)]):
            if tuple(x.shape) != shape or not torch.isfinite(x).all():
                raise ValueError(f'Invalid cached features for {s["sample_id"]}')
        return streams, torch.tensor(s['presence_label'], dtype=torch.float32), s['sample_id']
