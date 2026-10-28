"""TRIDENT real validation windows (rgb-visibility-v1, 1,961 windows); observed-test clips are refused."""
import json
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from ..composite_windows import AUDIO_SAMPLES, load_samples, partition


def _images(root, paths):
    frames = []
    for path in paths:
        with Image.open(root / path) as image:
            frames.append(image.convert('RGB').copy())
    return frames


def _audio(root, path, start):
    with wave.open(str(root / path)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) != (44100, 1, 2, 441000):
            raise ValueError(f'Unexpected WAV format: {path}')
        wav.setpos(start)
        signal = np.frombuffer(wav.readframes(AUDIO_SAMPLES), dtype='<i2').astype(np.float32) / 32768.
    if signal.shape != (AUDIO_SAMPLES,):
        raise ValueError(f'Short audio crop: {path}')
    return signal


def rows(root):
    """Window metadata only, in the order of the saved validation predictions."""
    _, validation, _ = partition(load_samples(root))
    return validation


def windows(root, selected=None):
    root = Path(root)
    sources = {s['clip_id']: s for s in json.loads((root / 'data/windows/sources.json').read_text())}
    for row in selected if selected is not None else rows(root):
        if row['role'] != 'validation' or row['kind'] != 'real':
            raise ValueError(f'Only real validation windows are allowed: {row["sample_id"]}')
        rf_paths = [sources[row['rf_audio_clip_id']]['rf_files'][i - 1]['path'] for i in row['rf_indices']]
        yield {'id': row['sample_id'], 'label': row['presence_label'], 'slice': row['slice'],
               'recording': row['rgb_clip_id'],
               'rgb': _images(root, row['rgb_paths']), 'rf': _images(root, rf_paths),
               'audio': _audio(root, row['audio_path'], row['audio_start_sample'])}
