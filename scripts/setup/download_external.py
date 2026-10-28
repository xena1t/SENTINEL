"""Fetch RFVS or Halmstad into the directory layout consumed by the inference adapters."""
import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'rfvs': {'drive_id': '1ANz2rfjkWlMiWKkTb31dbqQhCApGPFtK',
             'archive': 'RFVS.zip', 'prefixes': ('images/test/', 'RF_raw/test/', 'labels/test/')},
    'halmstad': {'url': 'https://zenodo.org/records/5500576/files/Drone-detection-dataset-v1.0.0.zip?download=1',
                 'archive': 'Drone-detection-dataset-v1.0.0.zip',
                 'prefixes': ('DroneDetectionThesis-Drone-detection-dataset-e7a6eaf/Data/Video_V/',
                              'DroneDetectionThesis-Drone-detection-dataset-e7a6eaf/Data/Audio/')},
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset', choices=sorted(SOURCES))
parser.add_argument('--archive', type=Path, help='Use an existing ZIP instead of downloading it')
args = parser.parse_args()
source = SOURCES[args.dataset]
destination = ROOT / 'data/external' / args.dataset
destination.mkdir(parents=True, exist_ok=True)
archive = args.archive.resolve() if args.archive else destination / source['archive']
if not archive.exists():
    if args.dataset == 'rfvs':
        import gdown
        if not gdown.download(id=source['drive_id'], output=str(archive), resume=True):
            raise RuntimeError('RFVS download did not complete')
    else:
        partial = archive.with_suffix('.part')
        with urllib.request.urlopen(source['url'], timeout=120) as response, partial.open('wb') as output:
            shutil.copyfileobj(response, output)
        partial.replace(archive)

with zipfile.ZipFile(archive) as zipped:
    if args.dataset == 'rfvs':
        expected = json.loads((ROOT / 'data/audit/rfvs/archive-manifest.json').read_text())
        actual = [{'name': item.filename, 'size': item.file_size, 'crc': item.CRC} for item in zipped.infolist()]
        if sorted(actual, key=lambda item: item['name']) != sorted(
                ({key: row[key] for key in ('name', 'size', 'crc')} for row in expected),
                key=lambda item: item['name']):
            raise ValueError('RFVS archive differs from the released archive manifest')
    selected = [item for item in zipped.infolist()
                if not item.is_dir() and item.filename.startswith(source['prefixes'])]
    expected_counts = (580, 580, 580) if args.dataset == 'rfvs' else (285, 90)
    counts = tuple(sum(item.filename.startswith(prefix) for item in selected) for prefix in source['prefixes'])
    if counts != expected_counts:
        raise ValueError(f'Unexpected {args.dataset} source counts: {counts}')
    for item in selected:
        relative = PurePosixPath(item.filename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe archive path: {item.filename}')
        path = destination.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError(f'Archive target escapes destination: {item.filename}')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.part')
        with zipped.open(item) as payload, temporary.open('wb') as output:
            shutil.copyfileobj(payload, output)
        temporary.replace(path)
print(json.dumps({'dataset': args.dataset, 'files': len(selected), 'counts': counts, 'destination': str(destination)}))
