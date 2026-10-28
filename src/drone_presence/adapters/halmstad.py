"""Halmstad (Svanstrom 2021, Zenodo 5500576) single-sensor tests; video and audio are separate recordings.

Rules are fixed in research/external-test-protocol.md:
- video (mask 100): Video_V MP4 at 30 fps; non-overlapping 1 s windows from the start; frames at
  offsets 0,4,8,12,17,21,25,29 of each second (the TRIDENT rule); label 1 for DRONE, 0 for AIRPLANE, BIRD, HELICOPTER
- audio (mask 001): Audio WAV 44.1 kHz stereo 10 s; channel mean; non-overlapping 1 s windows;
  label 1 for DRONE, 0 for HELICOPTER, BACKGROUND
- never pair a video window with an audio window
"""
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DIRECTORY = 'data/external/halmstad/DroneDetectionThesis-Drone-detection-dataset-e7a6eaf/Data'
FRAME_OFFSETS = (0, 4, 8, 12, 17, 21, 25, 29)
SAMPLE_RATE, FPS = 44100, 30


def _label(stem):
    return int(stem.split('_')[-2] == 'DRONE')


def video_rows(root):
    result = []
    for path in sorted((Path(root) / DIRECTORY / 'Video_V').glob('*.mp4')):
        capture = cv2.VideoCapture(str(path))
        frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if round(fps) != FPS:
            raise ValueError(f'Expected 30 fps: {path.name}')
        for second in range(frames // FPS):
            result.append({'sample_id': f'halmstad-video:{path.stem}:s{second:02d}', 'path': str(path),
                           'second': second, 'label': _label(path.stem), 'slice': path.stem.split('_')[1].lower(),
                           'recording': path.stem})
    return result


def audio_rows(root):
    result = []
    for path in sorted((Path(root) / DIRECTORY / 'Audio').glob('*.wav')):
        with wave.open(str(path)) as wav:
            if (wav.getframerate(), wav.getsampwidth()) != (SAMPLE_RATE, 2):
                raise ValueError(f'Unexpected WAV format: {path.name}')
            seconds = wav.getnframes() // SAMPLE_RATE
        for second in range(seconds):
            result.append({'sample_id': f'halmstad-audio:{path.stem}:s{second:02d}', 'path': str(path),
                           'second': second, 'label': _label(path.stem), 'slice': path.stem.split('_')[0].lower(),
                           'recording': path.stem})
    return result


def video_windows(root, selected=None):
    capture, current = None, None
    for row in selected if selected is not None else video_rows(root):
        if row['path'] != current:
            if capture is not None:
                capture.release()
            capture, current = cv2.VideoCapture(row['path']), row['path']
        frames = []
        for offset in FRAME_OFFSETS:
            capture.set(cv2.CAP_PROP_POS_FRAMES, row['second'] * FPS + offset)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Cannot decode frame in {row["sample_id"]}')
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        yield {'id': row['sample_id'], 'label': row['label'], 'slice': row['slice'], 'recording': row['recording'],
               'rgb': frames, 'rf': None, 'audio': None}
    if capture is not None:
        capture.release()


def audio_windows(root, selected=None):
    for row in selected if selected is not None else audio_rows(root):
        with wave.open(row['path']) as wav:
            channels = wav.getnchannels()
            wav.setpos(row['second'] * SAMPLE_RATE)
            pcm = np.frombuffer(wav.readframes(SAMPLE_RATE), dtype='<i2').astype(np.float32) / 32768.
        signal = pcm.reshape(-1, channels).mean(1)
        if signal.shape != (SAMPLE_RATE,):
            raise ValueError(f'Short audio window: {row["sample_id"]}')
        yield {'id': row['sample_id'], 'label': row['label'], 'slice': row['slice'], 'recording': row['recording'],
               'rgb': None, 'rf': None, 'audio': signal}
