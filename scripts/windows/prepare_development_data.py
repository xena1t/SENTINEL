"""Prepare the exact development-window media union with resumable verified downloads."""
import argparse
import fcntl
import os
from collections import Counter
import concurrent.futures
import json
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/audit'))
from drone_presence.windows import all_windows, source_records, sha256
from download_trident_images import fetch

DIRECTORY = ROOT / 'data/windows'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def selection():
    contract = json.loads((DIRECTORY / 'contract.json').read_text())
    for name, expected in contract['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Pinned source changed: {name}')
    for name, expected in contract['artifact_sha256'].items():
        if sha256(DIRECTORY / name) != expected:
            raise ValueError(f'Pinned window artifact changed: {name}')
    sources = json.loads((DIRECTORY / 'sources.json').read_text())
    if sources != source_records(ROOT):
        raise ValueError('Sources differ from curated development assignments')
    windows = [json.loads(line) for line in (DIRECTORY / 'manifest.jsonl').read_text().splitlines()]
    if windows != list(all_windows(sources)):
        raise ValueError('Windows differ from the complete frozen development grid')
    by_clip = {s['clip_id']: s for s in sources}
    selected = {}
    for row in windows:
        if row['role'] not in ('train', 'validation'):
            raise ValueError('Observed-test processing refused')
        source = by_clip[row['clip_id']]
        for modality in ('rgb', 'rf'):
            for index in row[modality + '_indices']:
                frame = source[modality + '_files'][index - 1]
                record = {**frame, 'clip_id': row['clip_id'], 'role': row['role'], 'modality': modality}
                if frame['path'] in selected and selected[frame['path']] != record:
                    raise ValueError('Conflicting media identity')
                selected[frame['path']] = record
    files = sorted(selected.values(), key=lambda row: (row['modality'], row['path']))
    manifest = {'schema': 'development-media-v1', 'scope': 'run0_train_validation_only',
                'contract_sha256': sha256(DIRECTORY / 'contract.json'),
                'window_manifest_sha256': sha256(DIRECTORY / 'manifest.jsonl'),
                'source_manifest_sha256': sha256(DIRECTORY / 'sources.json'),
                'window_count': len(windows), 'clip_count': len(sources),
                'image_counts': dict(Counter(r['modality'] for r in files)),
                'image_expected_bytes': sum(r['bytes'] for r in files),
                'audio_expected_bytes': sum(s['audio']['bytes'] for s in sources), 'files': files}
    path = DIRECTORY / 'development-download-manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Development selection changed; existing manifest retained')
    if not path.exists():
        write_json(path, manifest)
    return manifest, sources


def verify_audio(source):
    audio = source['audio']
    path = ROOT / audio['path']
    if path.stat().st_size != audio['bytes'] or sha256(path) != audio['sha256']:
        raise ValueError(f'WAV identity differs: {path}')
    import hashlib
    with wave.open(str(path)) as wav:
        if (wav.getframerate(), wav.getnframes(), wav.getnchannels(), wav.getsampwidth()) != (
                audio['sample_rate'], audio['samples'], audio['channels'], audio['sample_width']):
            raise ValueError(f'WAV format differs: {path}')
        if hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest() != audio['pcm_sha256']:
            raise ValueError(f'PCM identity differs: {path}')
    return {'clip_id': source['clip_id'], 'role': source['role'], 'path': audio['path'],
            'status': 'verified', 'sha256': audio['sha256'], 'pcm_sha256': audio['pcm_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        parser.error('--workers must be between 1 and 24')
    manifest, sources = selection()
    files = manifest['files']
    missing = [r for r in files if not (ROOT / r['path']).exists()]
    inventory = {k: v for k, v in manifest.items() if k != 'files'}
    inventory.update(missing_image_count=len(missing), missing_image_bytes=sum(r['bytes'] for r in missing),
                     missing_image_counts=dict(Counter(r['modality'] for r in missing)))
    write_json(DIRECTORY / 'development-inventory.json', inventory)
    print(json.dumps(inventory), flush=True)
    if args.inventory_only:
        return
    audio = [verify_audio(s) for s in sources]
    write_json(DIRECTORY / 'development-audio-integrity.json', {'status': 'PASS', 'clips': audio})
    started = time.monotonic()
    results = []
    counts = Counter()
    failures = []
    report_path = DIRECTORY / 'development-integrity.json'
    def checkpoint(done=False):
        report = {'schema': 'development-media-integrity-v1',
                  'status': 'PASS' if done and not failures else 'FAIL' if done else 'RUNNING',
                  'selection_sha256': sha256(DIRECTORY / 'development-download-manifest.json'),
                  'selected_images': len(files), 'completed_images': len(results),
                  'remaining_images': len(files) - len(results), 'statuses': dict(counts),
                  'verified_audio_clips': len(audio), 'elapsed_seconds': round(time.monotonic() - started),
                  'failures': failures, 'verify_only': args.verify_only}
        write_json(report_path, report)
        print(json.dumps(report), flush=True)
    with (DIRECTORY / 'development-download-progress.jsonl').open('a') as log:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, row, args.verify_only or row['modality'] == 'rf') for row in files]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                counts[result['status']] += 1
                if result['status'] == 'failed':
                    failures.append(result)
                log.write(json.dumps(result) + '\n')
                log.flush()
                if len(results) % 100 == 0:
                    checkpoint()
    checkpoint(done=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    with (DIRECTORY / 'development-download.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
        main()
