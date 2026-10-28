"""Download and verify TRIDENT media for the 53 validation clips only (pipeline parity gate).

Every RGB frame, RF frame and WAV of each validation clip is fetched from the file IDs pinned in
data/windows/sources.json. Images are checked by size, source MD5 and decode; WAVs by size, SHA-256,
format and PCM SHA-256. Train and observed-test clips are refused.
"""
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/audit'))
sys.path.insert(0, str(ROOT / 'scripts/windows'))
from drone_presence.windows import sha256
from download_trident_images import fetch
from prepare_development_data import verify_audio

DIRECTORY = ROOT / 'data/windows'
REPORT = ROOT / 'data/trident/validation-integrity.json'
EXPECTED_CLIPS = 53


def validation_sources():
    contract = json.loads((DIRECTORY / 'contract.json').read_text())
    for name, expected in contract['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Pinned source changed: {name}')
    if sha256(DIRECTORY / 'sources.json') != contract['artifact_sha256']['sources.json']:
        raise ValueError('Pinned sources.json changed')
    sources = json.loads((DIRECTORY / 'sources.json').read_text())
    if any(s['role'] not in ('train', 'validation') for s in sources):
        raise ValueError('Observed-test clip in sources; refusing')
    chosen = [s for s in sources if s['role'] == 'validation']
    if len(chosen) != EXPECTED_CLIPS:
        raise ValueError(f'Expected {EXPECTED_CLIPS} validation clips, found {len(chosen)}')
    return chosen


def image_rows(sources):
    rows = []
    for source in sources:
        for modality in ('rgb', 'rf'):
            for frame in source[modality + '_files']:
                rows.append({'clip_id': source['clip_id'], 'modality': modality, 'id': frame['id'],
                             'name': frame['name'], 'bytes': frame['bytes'], 'md5': frame['md5'],
                             'path': frame['path']})
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate media path')
    return rows


def fetch_audio(source, verify_only=False):
    audio = source['audio']
    path = ROOT / audio['path']
    try:
        if not path.exists():
            if verify_only:
                raise FileNotFoundError('WAV missing')
            path.parent.mkdir(parents=True, exist_ok=True)
            last_error = None
            for attempt in range(5):
                try:
                    request = urllib.request.Request(
                        'https://drive.usercontent.google.com/download?id=' + audio['id'] + '&export=download',
                        headers={'User-Agent': 'TRIDENT-research-audit/1.0'})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        data = response.read(audio['bytes'] + 1)
                    if len(data) != audio['bytes'] or hashlib.sha256(data).hexdigest() != audio['sha256']:
                        raise ValueError('Size or SHA-256 mismatch')
                    temporary = path.with_suffix(path.suffix + '.part')
                    temporary.write_bytes(data)
                    temporary.replace(path)
                    break
                except Exception as exc:
                    last_error = f'{type(exc).__name__}: {exc}'
                    if attempt < 4:
                        time.sleep(min(2 ** attempt, 16))
            else:
                raise RuntimeError(last_error)
        return verify_audio(source)
    except Exception as exc:
        return {'clip_id': source['clip_id'], 'path': audio['path'], 'status': 'failed',
                'error': f'{type(exc).__name__}: {exc}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        parser.error('--workers must be between 1 and 24')
    sources = validation_sources()
    rows = image_rows(sources)
    print(json.dumps({'clips': len(sources), 'images': dict(Counter(r['modality'] for r in rows)),
                      'image_bytes': sum(r['bytes'] for r in rows),
                      'audio_bytes': sum(s['audio']['bytes'] for s in sources)}), flush=True)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        audio = list(pool.map(lambda s: fetch_audio(s, args.verify_only), sources))
        results = []
        for result in pool.map(lambda r: fetch(r, args.verify_only), rows):
            results.append(result)
            if len(results) % 500 == 0:
                print(json.dumps({'completed': len(results), 'total': len(rows),
                                  'statuses': dict(Counter(r['status'] for r in results)),
                                  'elapsed_s': round(time.monotonic() - started)}), flush=True)
    failures = [r for r in results + audio if r['status'] == 'failed']
    report = {'schema': 'trident-validation-media-v1', 'status': 'FAIL' if failures else 'PASS',
              'sources_sha256': sha256(DIRECTORY / 'sources.json'), 'clips': len(sources),
              'image_statuses': dict(Counter(r['status'] for r in results)),
              'audio_statuses': dict(Counter(r['status'] for r in audio)),
              'elapsed_s': round(time.monotonic() - started), 'verify_only': args.verify_only,
              'failures': failures}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'failures'} | {'failures': len(failures)}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
