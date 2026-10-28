"""Download every pinned source image and WAV for the isolated 53 test recordings."""
import argparse
import concurrent.futures
import fcntl
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from drone_presence.test_windows_v2 import DIRECTORY, load, verify_media
from drone_presence.windows import sha256
from download_trident_validation import image_rows, fetch_audio
from download_trident_images import fetch


def main(a):
    if not 1 <= a.workers <= 24:
        raise ValueError('Workers must be between 1 and 24')
    sources, _, _ = load(ROOT)
    images = image_rows(sources)
    manifest = {'schema': 'trident-test-media-v1', 'sources_sha256': sha256(ROOT / DIRECTORY / 'sources.json'),
                'images': dict(Counter(r['modality'] for r in images)),
                'image_bytes': sum(r['bytes'] for r in images),
                'audio_bytes': sum(s['audio']['bytes'] for s in sources), 'audio_files': len(sources)}
    print(json.dumps(manifest), flush=True)
    if a.inventory_only:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        audio = list(pool.map(lambda source: fetch_audio(source, a.verify_only), sources))
        results = []
        for result in pool.map(lambda row: fetch(row, a.verify_only), images):
            results.append(result)
            if len(results) % 500 == 0:
                progress = {'status': 'RUNNING', 'completed': len(results), 'total': len(images),
                            'statuses': dict(Counter(r['status'] for r in results))}
                print(json.dumps(progress), flush=True)
                with (ROOT / DIRECTORY / 'download-progress.jsonl').open('a') as stream:
                    stream.write(json.dumps(progress) + '\n')
    failures = [row for row in audio + results if row['status'] == 'failed']
    report = {**manifest, 'status': 'FAIL' if failures else 'PASS', 'failures': failures,
              'image_statuses': dict(Counter(r['status'] for r in results)),
              'audio_statuses': dict(Counter(r['status'] for r in audio))}
    if not failures:
        report['verification'] = verify_media(ROOT, sources)
    (ROOT / DIRECTORY / 'media-integrity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'failures'}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    lock_path = ROOT / DIRECTORY / 'download.lock'
    if not lock_path.parent.exists():
        parser.error('Build test metadata with prepare_trident_test_v2.py first')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another test media downloader holds the lock')
        main(args)
