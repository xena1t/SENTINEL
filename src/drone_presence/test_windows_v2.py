"""Isolated run-0 observed-test windows and the sealed inference boundary."""
import hashlib
import json
import math
import wave
from collections import Counter
from pathlib import Path

from .windows import AUDIT, SOURCE_FILES, source_records, window_record, sha256

DIRECTORY = Path('data/windows/test-v2')
DEVELOPMENT_ARTIFACTS = {
    'data/windows/contract.json': '282e39e76553d8948eebc2be8d9623c28c76c1461c0f59a0dbb9ee6e3c5153eb',
    'data/windows/sources.json': '871036aa5ebec635a12861cb7a4395b9fead23db4f680bdd0f2308614d137dac',
    'data/windows/manifest.jsonl': '385863a7ad30b71c2691c644a6521b8d94ea37568dd50e87a0e8373bcab73494',
}
MASKS = ['100', '010', '001', '110', '101', '011', '111']
RELEASE = {
    'rf-cnn-best': {'sha256': '74503b7c625a261443104eeb13933bb2347ffca82efd2635d1030fdb03e75f28', 'epoch': 25},
    'fusion-best': {'sha256': '643b797bccb356e11c170bb449ff027d270e27e3588897daab0b763ca64cecf6', 'epoch': 4},
    'qwen-dropout-connector-best': {'sha256': 'c3a6a2ca6e447c17ee370bb3bb1d6c4d6869753d5cc8d9e4e90450125fe60ea0', 'epoch': 15},
}
CODE_FILES = (
    'scripts/infer/run_pipeline.py', 'scripts/infer/evaluate_external.py',
    'src/drone_presence/windows.py', 'src/drone_presence/test_windows_v2.py',
    'src/drone_presence/adapters/trident_test.py', 'src/drone_presence/adapters/trident.py',
    'src/drone_presence/pipeline.py', 'src/drone_presence/composite_windows.py',
    'src/drone_presence/model.py', 'src/drone_presence/qwen_alignment.py',
    'src/drone_presence/fusion_baseline.py', 'scripts/train/train_fusion.py',
)


def json_bytes(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def build(root):
    root = Path(root)
    sources = source_records(root, roles=('test',))
    if len(sources) != 53 or Counter(s['label'] for s in sources) != {0: 25, 1: 28}:
        raise ValueError('Expected exactly 53 observed-test recordings, 28 Drone and 25 Background')
    rows = []
    for source in sources:
        for end in range(120, 1201, 30):
            row = window_record(source, 120, end, allow_test=True)
            if (len(row['rgb_indices']), len(row['rf_indices']), row['audio_end_sample'] - row['audio_start_sample']) != (8, 4, 44100):
                raise ValueError('Test window input shape changed')
            row.update(sample_id=row['window_id'], kind='real', presence_label=row['label'],
                       slice='real_test', rgb_clip_id=source['clip_id'], rf_audio_clip_id=source['clip_id'],
                       rgb_paths=[source['rgb_files'][i - 1]['path'] for i in row['rgb_indices']],
                       audio_path=source['audio']['path'])
            rows.append(row)
    payloads = {'sources.json': json_bytes(sources),
                'manifest.jsonl': ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows).encode()}
    contract = {'schema': 'trident-test-window-v2', 'scope': 'run0_observed_test_only',
                'window_recipe': 'trident-window-v2.1', 'clips': 53, 'windows': 1961,
                'clip_labels': {'Background': 25, 'Drone': 28},
                'duration_tick': 120, 'step_tick': 30, 'ticks_per_second': 120,
                'rgb_frames': 8, 'rf_frames': 4, 'audio_samples': 44100,
                'labels': 'inherited_clip_presence', 'visibility_labels': False, 'composites': False,
                'timing_status': 'nominally_aligned_offline_window_classification_not_verified_streaming',
                'source_sha256': {str(AUDIT / name): sha256(root / AUDIT / name) for name in SOURCE_FILES},
                'artifact_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
    payloads['contract.json'] = json_bytes(contract)
    return sources, rows, contract, payloads


def prepare(root):
    root = Path(root)
    _, _, contract, payloads = build(root)
    directory = root / DIRECTORY
    for name, data in payloads.items():
        path = directory / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f'Immutable test artifact differs: {path}')
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        path = directory / name
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(data)
    return contract


def load(root):
    root = Path(root)
    sources, rows, contract, payloads = build(root)
    for name, data in payloads.items():
        if (root / DIRECTORY / name).read_bytes() != data:
            raise ValueError(f'Test metadata differs from pinned source reconstruction: {name}')
    return sources, rows, contract


def expected_protocol(root, qwen_load='4bit'):
    root = Path(root)
    load(root)
    for name, digest in DEVELOPMENT_ARTIFACTS.items():
        if sha256(root / name) != digest:
            raise ValueError(f'Original development artifact changed: {name}')
    return {'schema': 'trident-test-v2-protocol-v1', 'dataset': 'trident-test',
            'contract_sha256': sha256(root / DIRECTORY / 'contract.json'),
            'masks': MASKS, 'threshold': .5,
            'metrics': ['macro_f1', 'accuracy', 'recall', 'false_positive_rate', 'auroc', 'tp', 'tn', 'fp', 'fn'],
            'bootstrap': {'replicates': 1000, 'seed': 4422, 'unit': 'recording',
                          'method': 'percentile_recording_resample_with_replacement',
                          'confidence': .95, 'metrics': ['macro_f1', 'accuracy', 'recall', 'false_positive_rate', 'auroc']},
            'qwen_load': qwen_load,
            'qwen_revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
            'clap_revision': 'c47d441165daa21986ead0850660917636a81775',
            'checkpoints': RELEASE, 'models': ['learned_fusion', 'qwen_dropout'],
            'test_role': 'run0_observed_test', 'labels': 'inherited_clip_presence',
            'visibility_labels': False, 'composites': False,
            'development_artifact_sha256': DEVELOPMENT_ARTIFACTS,
            'code_sha256': {name: sha256(root / name) for name in CODE_FILES}}


def verify_protocol(root, protocol_path, qwen_load='4bit'):
    root, protocol_path = Path(root), Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    for key, expected in expected_protocol(root, '4bit').items():
        if protocol.get(key) != expected:
            raise ValueError(f'Sealed test protocol differs: {key}')
    if qwen_load != protocol['qwen_load']:
        raise ValueError('Qwen precision differs from sealed test protocol')
    for name, release in RELEASE.items():
        path = root / 'checkpoints/version-2' / f'{name}.pt'
        if sha256(path) != release['sha256']:
            raise ValueError(f'Release checkpoint differs before model loading: {name}')
    versions = json.loads((root / 'models/model-paths.json').read_text())
    for key in ('qwen_revision', 'clap_revision'):
        if versions[key] != protocol[key]:
            raise ValueError(f'Local encoder revision differs: {key}')
    for path_key, revision_key in (('qwen_path', 'qwen_revision'), ('clap_path', 'clap_revision')):
        path = Path(versions[path_key])
        if not path.is_absolute():
            path = root / path
        revision = protocol[revision_key]
        candidates = (path.absolute(), path.resolve())
        if not any('snapshots' in candidate.parts and revision in candidate.parts for candidate in candidates):
            raise ValueError(f'Encoder path is outside its pinned Hugging Face snapshot: {path_key}')
        if not path.exists():
            raise ValueError(f'Pinned encoder is missing: {path}')
    return {'protocol_sha256': sha256(protocol_path), 'contract_sha256': protocol['contract_sha256'],
            'code_sha256': protocol['code_sha256']}


def verify_media(root, sources):
    from PIL import Image
    root = Path(root)
    checked = Counter()
    identity = hashlib.sha256()
    for source in sources:
        for modality in ('rgb', 'rf'):
            for ref in source[modality + '_files']:
                path = root / ref['path']
                data = path.read_bytes()
                if len(data) != ref['bytes'] or hashlib.md5(data).hexdigest() != ref['md5']:
                    raise ValueError(f'Test media size or MD5 changed: {path}')
                with Image.open(path) as image:
                    image.load()
                    image.convert('RGB').load()
                identity.update(json_bytes([ref['path'], hashlib.sha256(data).hexdigest()]))
                checked[modality] += 1
        ref = source['audio']
        path = root / ref['path']
        if path.stat().st_size != ref['bytes'] or sha256(path) != ref['sha256']:
            raise ValueError(f'Test audio identity changed: {path}')
        with wave.open(str(path)) as wav:
            if (wav.getframerate(), wav.getnframes(), wav.getnchannels(), wav.getsampwidth()) != (44100, 441000, 1, 2):
                raise ValueError(f'Test audio format changed: {path}')
            if hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest() != ref['pcm_sha256']:
                raise ValueError(f'Test audio PCM changed: {path}')
        identity.update(json_bytes([ref['path'], ref['sha256']]))
        checked['audio'] += 1
    if checked != {'rgb': 15900, 'rf': 2120, 'audio': 53}:
        raise ValueError('Incomplete test media verification')
    return {'status': 'PASS', 'files': dict(checked), 'payload_sha256': identity.hexdigest()}


def verify_predictions(report, expected_rows, *, require_qwen=True):
    if report.get('dataset') != 'trident-test' or report.get('masks') != MASKS or report.get('windows') != 1961 or report.get('limit') != 0:
        raise ValueError('Test predictions must cover the sealed complete dataset and all seven masks')
    expected = {(r['sample_id'], mask): (r['presence_label'], r['clip_id'], r['slice'])
                for r in expected_rows for mask in MASKS}
    seen = set()
    for row in report['rows']:
        key = (row['sample_id'], row['mask'])
        if key in seen or key not in expected or (row['label'], row['recording'], row['slice']) != expected[key]:
            raise ValueError('Duplicate, unknown or relabelled test prediction')
        seen.add(key)
        for name in ('fusion_probability', 'probability') if require_qwen else ('fusion_probability',):
            value = row.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Missing or invalid test probability: {name}')
    if seen != set(expected):
        raise ValueError('Incomplete sample-mask test prediction coverage')
    if report.get('test_seal', {}).get('media', {}).get('status') != 'PASS':
        raise ValueError('Test predictions lack a successful media seal')
    return True
