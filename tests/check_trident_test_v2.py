"""Prove test partition isolation, exact shared sampling and complete scoring coverage."""
import copy
import hashlib
import json
import importlib.util
import wave
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from drone_presence.test_windows_v2 import (DIRECTORY, MASKS, DEVELOPMENT_ARTIFACTS,
                                          build, load, prepare, verify_predictions)
from drone_presence.windows import all_windows, source_records, window_record


def refuses(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('Invalid operation accepted')


def main():
    before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in DEVELOPMENT_ARTIFACTS}
    assert before == DEVELOPMENT_ARTIFACTS
    sources, rows, contract, payloads = build(ROOT)
    assert len(sources) == 53 and Counter(s['label'] for s in sources) == {0: 25, 1: 28}
    assert len(rows) == len({r['sample_id'] for r in rows}) == 1961
    assert Counter(r['label'] for r in rows) == {0: 925, 1: 1036}
    development = source_records(ROOT)
    assert development == json.loads((ROOT / 'data/windows/sources.json').read_text())
    assert len(development) == 222 and {s['role'] for s in development} == {'train', 'validation'}
    assert {s['clip_id'] for s in development}.isdisjoint({s['clip_id'] for s in sources})
    assert {s['group_proxy'] for s in development}.isdisjoint({s['group_proxy'] for s in sources})
    expected_development = [json.loads(line) for line in (ROOT / 'data/windows/manifest.jsonl').read_text().splitlines()]
    assert list(all_windows(development)) == expected_development
    assert sum(row['duration_tick'] == 120 for row in expected_development) == 8214
    test_source = sources[0]
    refuses(window_record, test_source, 120, 120)
    refuses(window_record, test_source, 30, 30, allow_test=True)
    refuses(lambda: list(all_windows(sources)))
    for roles in ((), ('unknown',), ('test', 'test')):
        refuses(source_records, ROOT, roles=roles)
    by_clip = {s['clip_id']: s for s in sources}
    for row in rows:
        shared = window_record(by_clip[row['clip_id']], 120, row['end_tick'], allow_test=True)
        assert {key: row[key] for key in shared} == shared
        assert row['kind'] == 'real' and row['role'] == 'test'
        assert row['rgb_clip_id'] == row['rf_audio_clip_id'] == row['clip_id']
        assert 'rgb_visibility' not in row
        assert row['audio_end_sample'] - row['audio_start_sample'] == 44100
        assert len(row['rgb_indices']) == 8 and len(row['rf_indices']) == 4
        assert row['end_tick'] - row['start_tick'] == 120
    assert all(len([r for r in rows if r['clip_id'] == s['clip_id']]) == 37 for s in sources)
    with tempfile.TemporaryDirectory() as name:
        temporary = Path(name)
        target = temporary / 'data/audit'
        target.mkdir(parents=True)
        (target / 'trident').symlink_to(ROOT / 'data/audit/trident', target_is_directory=True)
        assert prepare(temporary) == contract
        first = {name: (temporary / DIRECTORY / name).read_bytes() for name in payloads}
        assert prepare(temporary) == contract
        assert first == {name: (temporary / DIRECTORY / name).read_bytes() for name in payloads}
        assert load(temporary) == (sources, rows, contract)
        manifest = temporary / DIRECTORY / 'manifest.jsonl'
        manifest.write_text(manifest.read_text() + '\n')
        refuses(load, temporary)
        refuses(prepare, temporary)
    predictions = [dict(sample_id=r['sample_id'], mask=mask, label=r['label'], recording=r['clip_id'],
                        slice=r['slice'], fusion_probability=.5, probability=.5)
                   for r in rows for mask in MASKS]
    report = {'dataset': 'trident-test', 'masks': MASKS, 'windows': 1961, 'limit': 0,
              'test_seal': {'media': {'status': 'PASS'}}, 'rows': predictions}
    assert verify_predictions(report, rows)
    for mutation in ('missing', 'duplicate', 'label', 'recording', 'nan', 'missing_qwen', 'mask'):
        invalid = copy.deepcopy(report)
        if mutation == 'missing':
            invalid['rows'].pop()
        elif mutation == 'duplicate':
            invalid['rows'][-1] = invalid['rows'][0]
        elif mutation == 'label':
            invalid['rows'][0]['label'] = 1 - invalid['rows'][0]['label']
        elif mutation == 'recording':
            invalid['rows'][0]['recording'] = 'other-recording'
        elif mutation == 'nan':
            invalid['rows'][0]['probability'] = float('nan')
        elif mutation == 'missing_qwen':
            del invalid['rows'][0]['probability']
            assert verify_predictions(invalid, rows, require_qwen=False)
        else:
            invalid['masks'] = MASKS[:-1]
        refuses(verify_predictions, invalid, rows)
    dependencies = ('torch', 'numpy', 'PIL')
    if all(importlib.util.find_spec(name) is not None for name in dependencies):
        import numpy as np
        from PIL import Image
        from drone_presence.adapters import trident_test
        row = rows[1]
        source = by_clip[row['clip_id']]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            paths = row['rgb_paths'] + [source['rf_files'][index - 1]['path'] for index in row['rf_indices']]
            for path in paths:
                target = temporary / path
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (8, 8), (42, 84, 126)).save(target)
            target = temporary / row['audio_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            signal = (np.arange(441000, dtype=np.int64) % 32000 - 16000).astype('<i2')
            with wave.open(str(target), 'wb') as wav:
                wav.setparams((1, 2, 44100, 441000, 'NONE', 'not compressed'))
                wav.writeframes(signal.tobytes())
            original_load = trident_test.load
            trident_test.load = lambda root: ([source], [row], contract)
            try:
                actual = list(trident_test.windows(temporary, [row]))[0]
                assert len(actual['rgb']) == 8 and len(actual['rf']) == 4
                assert actual['audio'].shape == (44100,)
                assert np.array_equal(actual['audio'], signal[row['audio_start_sample']:row['audio_end_sample']].astype(np.float32) / 32768.)
                refuses(lambda: list(trident_test.windows(temporary, [{**row, 'role': 'validation'}])))
                refuses(lambda: list(trident_test.windows(temporary, [{**row, 'audio_start_sample': 0}])))
            finally:
                trident_test.load = original_load
        print('PASS: adapter decodes 8 RGB and 4 RF frames, exact one-second WAV crop, modified-row guard')
    else:
        print('SKIP: adapter payload check requires torch, numpy and Pillow')
    after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in DEVELOPMENT_ARTIFACTS}
    assert before == after
    print('PASS: 53 test recordings, 1961 exact shared-sampler windows, 13727 sample-mask pairs, original 24642 development rows unchanged, immutable metadata and scoring refusal checks')


if __name__ == '__main__':
    main()
