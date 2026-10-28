"""Fetch the ten-second development clips with the exact original eight camera frames."""
import argparse
import concurrent.futures
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/audit'))
sys.path.insert(0, str(ROOT / 'scripts/windows'))
from download_trident_images import fetch, selection
from fetch_window_media import fetch_audio

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--inventory-only', action='store_true')
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
if not 1 <= args.workers <= 24:
    parser.error('--workers must be between 1 and 24')
sources = json.loads((ROOT / 'data/windows/sources.json').read_text())
if len(sources) != 222 or {source['role'] for source in sources} != {'train', 'validation'}:
    raise ValueError('Expected the 222 development clips only')
ids = {source['clip_id'] for source in sources}
images = [row for row in selection() if row['clip_id'] in ids]
manifest = {row['clip_id']: row for row in json.loads((ROOT / 'data/trident/local-presence-manifest.json').read_text())}
expected = {path for clip in ids for key in ('rgb_paths', 'rf_paths') for path in manifest[clip][key]}
if {row['path'] for row in images} != expected:
    raise ValueError('Historical image selection differs from the training manifest')
report = {'images': len(images), 'image_counts': dict(Counter(row['modality'] for row in images)),
          'audio': len(sources), 'status': 'INVENTORY'}
if not args.inventory_only:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        audio = list(pool.map(fetch_audio, sources))
        results = list(pool.map(fetch, images))
    failures = [row for row in audio + results if row['status'] == 'failed']
    report.update(status='FAIL' if failures else 'PASS', failures=failures)
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
if report['status'] == 'FAIL':
    raise SystemExit(1)
