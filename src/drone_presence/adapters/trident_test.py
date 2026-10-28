"""Raw-media adapter for the isolated, sealed TRIDENT observed-test windows."""
from pathlib import Path

from ..test_windows_v2 import load
from .trident import _audio, _images


def rows(root):
    return load(root)[1]


def windows(root, selected=None):
    root = Path(root)
    sources, expected, _ = load(root)
    source_map = {source['clip_id']: source for source in sources}
    expected_map = {row['sample_id']: row for row in expected}
    chosen = selected if selected is not None else expected
    for row in chosen:
        if row != expected_map.get(row.get('sample_id')):
            raise ValueError('Only exact pinned real test rows are allowed')
        source = source_map[row['clip_id']]
        rf_paths = [source['rf_files'][i - 1]['path'] for i in row['rf_indices']]
        yield {'id': row['sample_id'], 'label': row['presence_label'], 'slice': row['slice'],
               'recording': row['clip_id'], 'rgb': _images(root, row['rgb_paths']),
               'rf': _images(root, rf_paths), 'audio': _audio(root, row['audio_path'], row['audio_start_sample'])}
