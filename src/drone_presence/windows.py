"""Nominal offline window identities and source selection, independent of v1 caches."""
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

VERSION = 'trident-window-v2.1'
TICKS_PER_SECOND = 120
CLIP_TICKS = 1200
STEP_TICKS = 30
DURATIONS = (30, 120, 240, 1200)
AUDIT = Path('data/audit/trident')
SOURCE_FILES = ('curated-paired-manifest.json', 'curated-evaluation-runs.csv',
                'audio-integrity.json', 'curated-manifest-checksums.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evenly_spaced(indices, maximum=8):
    """Include both endpoints; use exact nearest rounding with half ties upward."""
    count = min(maximum, len(indices))
    if count < 2:
        return indices[:count]
    denominator = count - 1
    return [indices[(2 * i * (len(indices) - 1) + denominator) // (2 * denominator)]
            for i in range(count)]


def source_records(root, *, roles=('train', 'validation')):
    if not roles or len(set(roles)) != len(roles) or any(role not in ('train', 'validation', 'test') for role in roles):
        raise ValueError('Explicit source roles must be unique known partitions')
    root = Path(root)
    curated = json.loads((root / AUDIT / 'curated-paired-manifest.json').read_text())
    pins = json.loads((root / AUDIT / 'curated-manifest-checksums.json').read_text())
    for name in ('curated-paired-manifest.json', 'curated-evaluation-runs.csv', 'audio-integrity.json'):
        if sha256(root / AUDIT / name) != pins[name]:
            raise ValueError(f'Curated source changed: {name}')
    with (root / AUDIT / 'curated-evaluation-runs.csv').open() as stream:
        assignments = [r for r in csv.DictReader(stream) if int(r['run']) == 0]
    clips = {r['clip_id']: r for r in curated}
    if len(clips) != 275 or len(assignments) != 275 or len({r['clip_id'] for r in assignments}) != 275:
        raise ValueError('Expected 275 unique curated clips and run-0 assignments')
    if {r['clip_id'] for r in assignments} != set(clips):
        raise ValueError('Assignments must cover the curated manifest exactly')
    if Counter(r['role'] for r in assignments) != {'train': 169, 'validation': 53, 'test': 53}:
        raise ValueError('Run-0 role counts changed')
    audio = {r['drive_id']: r for r in json.loads((root / AUDIT / 'audio-integrity.json').read_text())}
    groups = {}
    sources = []
    for assignment in sorted(assignments, key=lambda r: r['clip_id']):
        clip = clips[assignment['clip_id']]
        group, role = assignment['candidate_group'], assignment['role']
        if groups.setdefault(group, role) != role:
            raise ValueError('Folder proxy crosses partitions')
        if clip['candidate_group'] != group or clip['label'] != assignment['label']:
            raise ValueError('Curated labels or groups disagree with assignments')
        if role not in roles:
            continue
        record = {'clip_id': clip['clip_id'], 'group_proxy': group, 'role': role,
                  'label': {'Background': 0, 'Drone': 1}[clip['label']],
                  'label_name': clip['label'], 'published_split': clip['published_split'],
                  'label_provenance': 'inherited_clip_label_weak_window_supervision'}
        for modality, expected in (('rgb', 300), ('rf', 40)):
            indexed = sorted(clip[modality + '_files'], key=lambda f: int(re.search(r'_frame_(\d+)\.', f['name'])[1]))
            indices = [int(re.search(r'_frame_(\d+)\.', f['name'])[1]) for f in indexed]
            if indices != list(range(1, expected + 1)) or len({f['id'] for f in indexed}) != expected:
                raise ValueError(f'Invalid {modality} source coverage for {clip["clip_id"]}')
            refs = []
            for index, file in zip(indices, indexed):
                if any(file[key] != clip[other] for key, other in
                       (('split', 'published_split'), ('group', 'candidate_group'), ('label', 'label'))):
                    raise ValueError('Image source crosses clip metadata')
                existing = Path('data/trident/images') / modality / clip['clip_id'] / file['name']
                historical_rgb = {1, 44, 86, 129, 172, 215, 257, 300}
                path = existing if modality == 'rf' or index in historical_rgb else Path('data/windows/rgb') / clip['clip_id'] / file['name']
                ref = {'index': index, 'id': file['id'], 'name': file['name'],
                       'bytes': int(file['size']), 'md5': file['md5Checksum'], 'path': str(path)}
                if modality == 'rgb':
                    ref['nominal_timestamp_tick'] = (index - 1) * 4
                else:
                    ref['nominal_support_ticks'] = [(index - 1) * 30, index * 30]
                refs.append(ref)
            record[modality + '_files'] = refs
        wav = audio[clip['audio_id']]
        if wav['pcm_sha256'] != clip['audio_pcm_sha256'] or wav['samples'] != wav['rate'] * 10:
            raise ValueError('Audio source identity or ten-second duration changed')
        record['audio'] = {'id': wav['drive_id'], 'path': str(AUDIT / 'audio-payloads' / wav['split'] / wav['label'] / wav['group'] / wav['name']),
                           'sample_rate': wav['rate'], 'samples': wav['samples'], 'channels': wav['channels'],
                           'sample_width': wav['sample_width'], 'bytes': int(wav['bytes']),
                           'sha256': wav['file_sha256'], 'pcm_sha256': wav['pcm_sha256']}
        sources.append(record)
    return sources


def window_record(source, duration_tick, end_tick, *, allow_test=False):
    if source['role'] == 'test' and allow_test and duration_tick != 120:
        raise ValueError('Test windows must be exactly one second')
    if source['role'] not in ('train', 'validation') and not (allow_test and source['role'] == 'test'):
        raise ValueError('Window construction is development-only')
    if duration_tick not in DURATIONS or end_tick % STEP_TICKS or not duration_tick <= end_tick <= CLIP_TICKS:
        raise ValueError('Window must lie on the frozen development grid')
    start_tick = end_tick - duration_tick
    eligible = [f['index'] for f in source['rgb_files'] if start_tick <= f['nominal_timestamp_tick'] < end_tick]
    rgb = evenly_spaced(eligible)
    rf = [f['index'] for f in source['rf_files']
          if start_tick <= f['nominal_support_ticks'][0] and f['nominal_support_ticks'][1] <= end_tick]
    rate = source['audio']['sample_rate']
    if start_tick * rate % TICKS_PER_SECOND or end_tick * rate % TICKS_PER_SECOND:
        raise ValueError('Window boundaries must map to integer original-rate samples')
    return {'window_id': f'{VERSION}:{source["clip_id"]}:t{start_tick:04d}-{end_tick:04d}@120',
            'manifest_version': VERSION, 'clip_id': source['clip_id'], 'group_proxy': source['group_proxy'],
            'role': source['role'], 'label': source['label'], 'label_provenance': source['label_provenance'],
            'start_tick': start_tick, 'end_tick': end_tick, 'duration_tick': duration_tick,
            'ticks_per_second': TICKS_PER_SECOND, 'rgb_indices': rgb,
            'rgb_nominal_timestamp_ticks': [(i - 1) * 4 for i in rgb], 'rgb_eligible_count': len(eligible),
            'rf_indices': rf, 'rf_support_status': 'nominal_unverified',
            'audio_start_sample': start_tick * rate // TICKS_PER_SECOND,
            'audio_end_sample': end_tick * rate // TICKS_PER_SECOND,
            'audio_sample_rate': rate, 'preprocessing_recipe': VERSION + '-sampling',
            'encoder_recipe': 'unselected_phase01_no_features'}


def all_windows(sources):
    for source in sources:
        if source['role'] not in ('train', 'validation'):
            raise ValueError('Window generation is development-only')
        for duration in DURATIONS:
            for end in range(duration, CLIP_TICKS + 1, STEP_TICKS):
                yield window_record(source, duration, end)


def probe_selection(sources, seed=4422):
    rng = random.Random(seed)
    selected = []
    for role, per_class in (('train', 4), ('validation', 2)):
        for label in (0, 1):
            groups = {}
            for source in sources:
                if source['role'] == role and source['label'] == label:
                    groups.setdefault(source['group_proxy'], []).append(source)
            names = sorted(groups)
            rng.shuffle(names)
            if len(names) < per_class:
                raise ValueError('Insufficient distinct groups for balanced probe')
            for group in names[:per_class]:
                selected.append(rng.choice(sorted(groups[group], key=lambda r: r['clip_id'])))
    windows = []
    for source in selected:
        for duration in (30, 120):
            ends = list(range(duration, CLIP_TICKS + 1, STEP_TICKS))
            for position, end in zip(('beginning', 'middle', 'end'), (ends[0], ends[(len(ends) - 1) // 2], ends[-1])):
                windows.append({**window_record(source, duration, end), 'audit_position': position})
    return selected, windows


def probe_rgb_files(sources, windows):
    index = {source['clip_id']: source for source in sources}
    rows = {}
    for window in windows:
        for frame_index in window['rgb_indices']:
            frame = index[window['clip_id']]['rgb_files'][frame_index - 1]
            rows[frame['path']] = {**frame, 'clip_id': window['clip_id'], 'modality': 'rgb'}
    return sorted(rows.values(), key=lambda r: r['path'])
