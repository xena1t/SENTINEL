"""Run the Version 2 inference pipeline on a dataset adapter and write one predictions JSON.

Each output row: sample_id, label, slice, recording, mask, fusion_probability and, unless --no-qwen,
probability (Qwen dropout yes-probability) with no/yes log probabilities, as in
research/windows/rgb-visibility-v1-220956/raw/alignment-dropout/predictions-*.json.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from drone_presence.pipeline import Pipeline, sha256
from drone_presence.adapters import halmstad, rfvs, trident, trident_test
from drone_presence.test_windows_v2 import load as load_test, verify_protocol, verify_media, verify_predictions

ALL_MASKS = ['100', '010', '001', '110', '101', '011', '111']
# name: (rows, windows, default masks, adapter source)
ADAPTERS = {'trident-test': (trident_test.rows, trident_test.windows, ALL_MASKS, 'trident_test'),
            'trident': (trident.rows, trident.windows, ALL_MASKS, 'trident'),
            'rfvs': (rfvs.rows, rfvs.windows, ['110', '100', '010'], 'rfvs'),
            'halmstad-video': (halmstad.video_rows, halmstad.video_windows, ['100'], 'halmstad'),
            'halmstad-audio': (halmstad.audio_rows, halmstad.audio_windows, ['001'], 'halmstad')}


def encoder_provenance(provenance):
    """Only what sets the encoder features; the fusion and connector checkpoints do not."""
    return {k: provenance[k] for k in ('qwen_revision', 'clap_revision', 'qwen_load')} | {
        'rf-cnn-best': provenance['checkpoints']['rf-cnn-best']}


def cached_features(pipeline, window, cache):
    if cache is None:
        return pipeline.encode(window)
    path = cache / (hashlib.sha256(window['id'].encode()).hexdigest()[:32] + '.pt')
    if path.exists():
        stored = torch.load(path, map_location='cpu', weights_only=True)
        if stored['id'] != window['id'] or encoder_provenance(stored['provenance']) != encoder_provenance(pipeline.provenance):
            raise ValueError(f'Cached features differ in identity or provenance: {path}')
        return stored['features']
    features = pipeline.encode(window)
    temporary = path.with_suffix('.tmp')
    torch.save({'id': window['id'], 'provenance': pipeline.provenance, 'features': features}, temporary)
    temporary.replace(path)
    return features


def main(a):
    test_seal = None
    if a.dataset == 'trident-test':
        if a.limit != 0 or a.masks is not None:
            raise ValueError('Sealed test inference requires every window and the fixed seven masks')
        if a.output.exists():
            raise ValueError('Sealed test output already exists; refusing overwrite')
        test_seal = verify_protocol(ROOT, a.protocol, a.qwen_load)
        sources, _, _ = load_test(ROOT)
        test_seal['media'] = verify_media(ROOT, sources)
        if a.cache:
            marker = a.cache / 'test-seal.json'
            if a.cache.exists() and any(a.cache.iterdir()):
                if not marker.exists() or json.loads(marker.read_text()) != test_seal:
                    raise ValueError('Test feature cache lacks the current protocol and media seal')
            else:
                a.cache.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps(test_seal, indent=2) + '\n')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    rows, windows, default_masks, source = ADAPTERS[a.dataset]
    selected = rows(ROOT)
    if a.limit:
        selected = selected[:a.limit]
    masks = a.masks.split(',') if a.masks else default_masks
    if any(m not in ALL_MASKS for m in masks):
        raise ValueError(f'Masks must be from {ALL_MASKS}')
    started = time.monotonic()
    pipeline = Pipeline(ROOT, qwen_load=a.qwen_load, use_qwen=not a.no_qwen)
    if test_seal:
        expected = json.loads(a.protocol.read_text())
        for key in ('qwen_revision', 'clap_revision', 'qwen_load'):
            if pipeline.provenance[key] != expected[key]:
                raise ValueError(f'Loaded model differs from test protocol: {key}')
        for name, checkpoint in pipeline.provenance['checkpoints'].items():
            if checkpoint != expected['checkpoints'][name]:
                raise ValueError(f'Loaded checkpoint differs from test protocol: {name}')
    if a.cache:
        a.cache.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'dataset': a.dataset, 'windows': len(selected), 'masks': masks, **pipeline.provenance,
                      'load_s': round(time.monotonic() - started)}), flush=True)
    output = []
    for i, window in enumerate(windows(ROOT, selected)):
        features = cached_features(pipeline, window, a.cache)
        for mask in masks:
            result = pipeline.predict(window, features, mask)
            row = {'sample_id': window['id'], 'label': window['label'], 'slice': window.get('slice'),
                   'recording': window.get('recording'), 'mask': mask,
                   'fusion_probability': result['fusion_probability']}
            if not a.no_qwen:
                row.update(probability=result['qwen_probability'], no_log_probability=result['no_log_probability'],
                           yes_log_probability=result['yes_log_probability'])
            output.append(row)
        if (i + 1) % 25 == 0:
            print(json.dumps({'windows_done': i + 1, 'total': len(selected),
                              'elapsed_s': round(time.monotonic() - started),
                              'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)}), flush=True)
    report = {'dataset': a.dataset, 'masks': masks, 'windows': len(selected), 'limit': a.limit,
              'provenance': pipeline.provenance,
              'code_sha256': {p: sha256(ROOT / p) for p in ('scripts/infer/run_pipeline.py', 'src/drone_presence/pipeline.py',
                                                            f'src/drone_presence/adapters/{source}.py')},
              'elapsed_s': round(time.monotonic() - started),
              'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2), 'rows': output}
    if test_seal:
        report['test_seal'] = test_seal
        report['phase'] = 'learned_fusion' if a.no_qwen else 'learned_fusion_and_qwen_dropout'
        verify_predictions(report, selected, require_qwen=not a.no_qwen)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x' if test_seal else 'w') as stream:
        stream.write(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=sorted(ADAPTERS), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qwen-load', choices=['bf16', '4bit', 'offload'], default='4bit')
    p.add_argument('--no-qwen', action='store_true', help='learned fusion only; Qwen still encodes RGB')
    p.add_argument('--masks', help='comma-separated; default depends on the dataset')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--protocol', type=Path, default=ROOT / 'research/test-v2/protocol.json')
    p.add_argument('--cache', type=Path, help='directory for per-window encoder features')
    main(p.parse_args())
