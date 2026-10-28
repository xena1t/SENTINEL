"""Pipeline gates on the laptop: strict checkpoint load, all 7 masks run, invalid masks refused.

Usage: .venv/bin/python tests/check_pipeline.py [--qwen-load 4bit]
Uses a synthetic window (random frames, noise audio), so it needs no dataset media.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from drone_presence.pipeline import Pipeline

MASKS = ['100', '010', '001', '110', '101', '011', '111']


def refused(call):
    try:
        call()
    except ValueError:
        return True
    return False


def main(a):
    pipeline = Pipeline(ROOT, qwen_load=a.qwen_load)  # load_state_dict(strict=True) for every checkpoint
    rng = np.random.default_rng(0)
    image = lambda size: Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8))
    window = {'id': 'synthetic', 'rgb': [image((480, 640)) for _ in range(8)],
              'rf': [image((300, 400)) for _ in range(4)], 'audio': rng.normal(0, .05, 44100).astype(np.float32)}
    features = pipeline.encode(window)
    shapes = {s: list(x.shape) for s, x in features.items()}
    results = {m: pipeline.predict(window, features, m) for m in MASKS}
    checks = {
        'feature_shapes': shapes == {'rgb': [128, 4096], 'rf': [4, 128], 'audio': [1, 1024]},
        'all_masks_finite': all(np.isfinite(r['fusion_probability']) and np.isfinite(r['qwen_probability'])
                                for r in results.values()),
        'probabilities_in_range': all(0 <= r[k] <= 1 for r in results.values() for k in ('fusion_probability', 'qwen_probability')),
        'all_missing_refused': refused(lambda: pipeline.predict(window, features, '000')),
        'absent_sensor_refused': refused(lambda: pipeline.predict({**window, 'audio': None}, {**features, 'audio': None}, '111')),
        'vision_tower_not_quantized': not any('4bit' in type(m).__name__ for m in pipeline.qwen.model.visual.modules())
                                      and pipeline.qwen.model.visual.patch_embed.proj.weight.dtype == torch.bfloat16,
        'missing_input_sets_mask': pipeline.predict({**window, 'rf': None}, {**features, 'rf': None})['mask'] == '101',
    }
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
              'provenance': pipeline.provenance,
              'synthetic_probabilities': {m: [round(r['fusion_probability'], 4), round(r['qwen_probability'], 4)]
                                          for m, r in results.items()},
              'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)}
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qwen-load', choices=['bf16', '4bit', 'offload'], default='4bit')
    main(p.parse_args())
