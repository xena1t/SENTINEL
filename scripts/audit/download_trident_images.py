"""Fetch the presence-plan image selection and verify payloads before promotion."""
import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'data/audit/trident'
DEST = ROOT / 'data/trident/images'


def selection():
    clips = json.loads((AUDIT / 'curated-paired-manifest.json').read_text())
    rows = []
    for clip in clips:
        for modality, key in [('rgb', 'rgb_files'), ('rf', 'rf_files')]:
            files = sorted(clip[key], key=lambda f: int(re.search(r'_frame_(\d+)', f['name'])[1]))
            assert len(files) == (300 if modality == 'rgb' else 40)
            chosen = [files[round(i * (len(files) - 1) / 7)] for i in range(8)] if modality == 'rgb' else files
            for file in chosen:
                path = DEST / modality / clip['clip_id'] / file['name']
                rows.append({'clip_id': clip['clip_id'], 'modality': modality, 'id': file['id'],
                             'name': file['name'], 'bytes': int(file['size']), 'md5': file['md5Checksum'],
                             'path': str(path.relative_to(ROOT))})
    assert len(rows) == 13200 and len({r['path'] for r in rows}) == 13200
    return rows


def inspect(data, row):
    if len(data) != row['bytes'] or hashlib.md5(data).hexdigest() != row['md5']:
        raise ValueError('Size or source MD5 mismatch')
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError('Invalid dimensions')
        fmt, mode = image.format, image.mode
        image.convert('RGB').load()
    return {'width': width, 'height': height, 'format': fmt, 'mode': mode}


def fetch(row, verify_only=False):
    path = ROOT / row['path']
    try:
        if path.exists():
            return {**row, **inspect(path.read_bytes(), row), 'status': 'verified_existing'}
        if verify_only:
            raise FileNotFoundError('Payload missing')
        path.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    'https://drive.usercontent.google.com/download?id=' + row['id'] + '&export=download',
                    headers={'User-Agent': 'TRIDENT-research-audit/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read(row['bytes'] + 1)
                info = inspect(data, row)
                temporary = path.with_suffix(path.suffix + '.part')
                temporary.write_bytes(data)
                temporary.replace(path)
                return {**row, **info, 'status': 'downloaded_verified'}
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'
                print(json.dumps({'event': 'download_attempt_error', 'path': row['path'],
                                  'attempt': attempt + 1, 'http_status': getattr(exc, 'code', None),
                                  'error': last_error}), file=sys.stderr, flush=True)
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 16))
        raise RuntimeError(last_error)
    except Exception as exc:
        return {**row, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    rows = selection()
    selected = {'rule': 'Numeric frame order; RGB indices round(i*299/7), i=0..7; every RF frame',
                'source_manifest_sha256': hashlib.sha256((AUDIT / 'curated-paired-manifest.json').read_bytes()).hexdigest(),
                'counts': dict(Counter(r['modality'] for r in rows)),
                'expected_bytes': sum(r['bytes'] for r in rows), 'files': rows}
    manifest = AUDIT / 'selected-image-download-manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == selected, 'Selection changed; review before replacing manifest'
    else:
        manifest.write_text(json.dumps(selected, indent=2))
    if args.limit:
        rows = rows[:args.limit]
    print(json.dumps({'selected': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'workers': args.workers}), flush=True)
    started = time.monotonic()
    results = []
    with (AUDIT / 'image-download-progress.jsonl').open('a') as log:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, row, args.verify_only) for row in rows]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                log.write(json.dumps(result) + '\n')
                log.flush()
                if len(results) % 100 == 0 or result['status'] == 'failed':
                    print(json.dumps({'completed': len(results), 'total': len(rows),
                                      'statuses': dict(Counter(r['status'] for r in results)),
                                      'elapsed_s': round(time.monotonic() - started)}), flush=True)
    report = {'scope': 'partial' if args.limit else 'full_selection', 'verify_only': args.verify_only,
              'selected_count': len(rows), 'statuses': dict(Counter(r['status'] for r in results)),
              'elapsed_s': round(time.monotonic() - started), 'files': sorted(results, key=lambda r: r['path'])}
    name = 'selected-image-integrity-partial.json' if args.limit else 'selected-image-integrity.json'
    (AUDIT / name).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)
    if any(r['status'] == 'failed' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
