"""RFVS external test (RGB + RF, mask 110): 580 released test pairs joined by exact stem.

Conversion rules are fixed in research/external-test-protocol.md:
- label 1 when labels/test/<stem>.txt has a box, label 0 when it is empty (released background sequences)
- RGB: the one still image repeated in the 8 frame slots
- RF: 248-byte header, int16 interleaved I/Q, 153.6 MS/s, 30 ms; split into 4 equal consecutive parts;
  each part -> two-sided STFT (Hamming 1024, overlap 512, FFT 1024, as the RFVS iq_to_stft.m),
  20*log10|S|, per-image min-max, viridis, frequency upward, time to the right, 610 x 610 like TRIDENT
"""
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image
from scipy.signal import stft

DIRECTORY = 'data/external/rfvs'
HEADER_BYTES, SAMPLE_RATE, RF_BYTES = 248, 153.6e6, 18_432_248
RF_PARTS, IMAGE_SIDE = 4, 610


def spectrogram_images(path):
    raw = Path(path).read_bytes()
    if len(raw) != RF_BYTES:
        raise ValueError(f'Unexpected RF file size: {path}')
    iq = np.frombuffer(raw[HEADER_BYTES:], dtype='<i2').astype(np.float32)
    signal = iq[0::2] + 1j * iq[1::2]
    images = []
    for part in np.array_split(signal, RF_PARTS):
        _, _, spectrum = stft(part, fs=SAMPLE_RATE, window='hamming', nperseg=1024, noverlap=512, nfft=1024,
                              return_onesided=False, boundary=None, padded=False)
        power = 20 * np.log10(np.abs(np.fft.fftshift(spectrum, axes=0)) + 1e-12)
        scaled = (power - power.min()) / max(float(power.max() - power.min()), 1e-12)
        rgb = (colormaps['viridis'](scaled[::-1])[..., :3] * 255).astype(np.uint8)
        images.append(Image.fromarray(rgb).resize((IMAGE_SIDE, IMAGE_SIDE), Image.Resampling.BILINEAR))
    return images


def sequence(stem):
    """Recording group: the stem without its trailing frame number."""
    return stem.rstrip('0123456789').rstrip('_')


def rows(root):
    directory = Path(root) / DIRECTORY
    stems = sorted(p.stem for p in (directory / 'images/test').glob('*.jpg'))
    if len(stems) != 580:
        raise ValueError(f'Expected 580 RFVS images, found {len(stems)}')
    result = []
    for stem in stems:
        label_file = directory / 'labels/test' / f'{stem}.txt'
        rf_file = directory / 'RF_raw/test' / stem
        if not label_file.exists() or not rf_file.exists():
            raise ValueError(f'Missing exact-stem partner for {stem}')
        label = int(bool(label_file.read_text().strip()))
        result.append({'sample_id': f'rfvs:{stem}', 'stem': stem, 'label': label,
                       'slice': 'drone' if label else 'background', 'recording': sequence(stem)})
    return result


def windows(root, selected=None):
    directory = Path(root) / DIRECTORY
    for row in selected if selected is not None else rows(root):
        with Image.open(directory / 'images/test' / f'{row["stem"]}.jpg') as image:
            frame = image.convert('RGB').copy()
        yield {'id': row['sample_id'], 'label': row['label'], 'slice': row['slice'], 'recording': row['recording'],
               'rgb': [frame] * 8, 'rf': spectrogram_images(directory / 'RF_raw/test' / row['stem']), 'audio': None}
