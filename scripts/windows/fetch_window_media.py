"""Fetch the media needed by rgb-visibility-v1 one-second windows, verified against pinned sources."""
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
sys.path.insert(0, str(ROOT / 'scripts/audit'))
from download_trident_images import fetch

DIRECTORY = ROOT / 'data/windows'


def selection():
    contract = json.loads((DIRECTORY / 'contract.json').read_text())
    for name, expected in contract['artifact_sha256'].items():
        if hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Pinned window artifact changed: {name}')
    sources = {s['clip_id']: s for s in json.loads((DIRECTORY / 'sources.json').read_text())}
    rows = [json.loads(line) for line in (DIRECTORY / 'manifest.jsonl').read_text().splitlines()]
    selected = {}
    for row in rows:
        if row['duration_tick'] != 120:
            continue
        if row['role'] not in ('train', 'validation') or sources[row['clip_id']]['role'] != row['role']:
            raise ValueError('Observed-test or inconsistent role refused')
        source = sources[row['clip_id']]
        for index in row['rgb_indices']:
            frame = source['rgb_files'][index - 1]
            selected[frame['path']] = {**frame, 'clip_id': row['clip_id'], 'modality': 'rgb'}
    # All 40 RF frames per clip: the RF CNN refit uses whole training clips.
    for source in sources.values():
        for frame in source['rf_files']:
            selected[frame['path']] = {**frame, 'clip_id': source['clip_id'], 'modality': 'rf'}
    return sorted(selected.values(), key=lambda r: (r['modality'], r['path'])), list(sources.values())


def fetch_audio(source):
    audio = source['audio']
    path = ROOT / audio['path']
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                url = 'https://drive.usercontent.google.com/download?id=' + audio['id'] + '&export=download'
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read(audio['bytes'] + 1)
                if len(data) != audio['bytes'] or hashlib.sha256(data).hexdigest() != audio['sha256']:
                    raise ValueError('WAV size or SHA-256 mismatch')
                temporary = path.with_suffix('.part')
                temporary.write_bytes(data)
                temporary.replace(path)
                break
            except Exception:
                if attempt == 4:
                    return {'path': audio['path'], 'status': 'failed'}
                time.sleep(min(2 ** attempt, 16))
    if hashlib.sha256(path.read_bytes()).hexdigest() != audio['sha256']:
        return {'path': audio['path'], 'status': 'failed'}
    return {'path': audio['path'], 'status': 'verified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=4)
    args = parser.parse_args()
    files, sources = selection()
    counts, started = Counter(), time.monotonic()
    pending, failures = files, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        audio = list(pool.map(fetch_audio, sources))
        # Throttled files get further rounds after a pause instead of failing the job.
        for round_number in range(1, args.rounds + 1):
            failures = []
            for number, result in enumerate(pool.map(fetch, pending), 1):
                counts[result['status']] += 1
                if result['status'] == 'failed':
                    failures.append(result)
                if number % 1000 == 0:
                    print(json.dumps({'round': round_number, 'done': number, 'total': len(pending), 'statuses': dict(counts),
                                      'elapsed_s': round(time.monotonic() - started)}), flush=True)
            if not failures:
                break
            print(json.dumps({'round': round_number, 'failed': len(failures), 'retrying': round_number < args.rounds}), flush=True)
            pending = [{k: v for k, v in r.items() if k not in ('status', 'error')} for r in failures]
            if round_number < args.rounds:
                time.sleep(300)
        audio = [r if r['status'] == 'verified' else fetch_audio(next(s for s in sources if s['audio']['path'] == r['path'])) for r in audio]
    failures = [{'path': r['path'], 'error': r.get('error')} for r in failures]
    audio_failures = [r for r in audio if r['status'] != 'verified']
    report = {'status': 'PASS' if not failures and not audio_failures else 'FAIL',
              'images': len(files), 'image_counts': dict(Counter(r['modality'] for r in files)),
              'statuses': dict(counts), 'audio_verified': len(audio) - len(audio_failures),
              'failures': failures + audio_failures, 'elapsed_s': round(time.monotonic() - started)}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'failures'}), flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
