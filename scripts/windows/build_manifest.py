"""Build the train/validation nominal-window contract and fixed development probe."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from drone_presence.windows import (AUDIT, SOURCE_FILES, VERSION, all_windows,
                                    probe_rgb_files, probe_selection, sha256, source_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'data/windows'
    output.mkdir(parents=True, exist_ok=True)
    sources = source_records(root)
    windows = list(all_windows(sources))
    selected, probe = probe_selection(sources)
    (output / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in windows))
    (output / 'probe.json').write_text(json.dumps({'seed': 4422, 'selection': '4 train and 2 validation clips per class, distinct folder proxies; seeded group shuffle and clip choice',
                                                'clip_ids': [r['clip_id'] for r in selected], 'windows': probe}, indent=2) + '\n')
    rgb = probe_rgb_files(sources, probe)
    (output / 'probe-rgb-download-manifest.json').write_text(json.dumps({'manifest_version': VERSION, 'files': rgb,
                                                                     'count': len(rgb), 'bytes': sum(r['bytes'] for r in rgb)}, indent=2) + '\n')
    metadata = {'manifest_version': VERSION, 'scope': 'run0_train_validation_only',
                'timing_status': 'nominally_aligned_offline_window_classification_not_verified_streaming',
                'ticks_per_second': 120, 'clip_ticks': 1200, 'step_ticks': 30,
                'duration_ticks': [30, 120, 240, 1200], 'interval': '[start_tick,end_tick)',
                'rgb_timestamp': '(one_based_index-1)/30 seconds; 300 references',
                'rgb_sampling': 'up to 8 distinct eligible indices, endpoints included, exact nearest integer with half ties upward',
                'rf_support': 'assumed [(index-1)/4,index/4) seconds; retain only complete bins wholly contained; acquisition and causal support unverified',
                'audio_bounds': 'original-rate half-open sample range; slice waveform before features; no padding or resampling in manifest',
                'probe_positions': 'first, lower-middle, last valid endpoint, separately at 0.25 and 1 second',
                'label_provenance': 'inherited_clip_label_weak_window_supervision; no instantaneous visibility or onset inference',
                'encoder_recipe': 'unselected_phase01_no_features',
                'clip_count': len(sources), 'clip_role_counts': dict(Counter(r['role'] for r in sources)),
                'window_count': len(windows), 'window_role_counts': dict(Counter(r['role'] for r in windows)),
                'window_duration_counts': dict(Counter(str(r['duration_tick']) for r in windows)),
                'probe_clip_count': len(selected), 'probe_window_count': len(probe), 'probe_rgb_count': len(rgb),
                'source_sha256': {str(AUDIT / name): sha256(root / AUDIT / name) for name in SOURCE_FILES},
                'artifact_sha256': {name: sha256(output / name) for name in ('sources.json', 'manifest.jsonl', 'probe.json', 'probe-rgb-download-manifest.json')}}
    (output / 'contract.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({key: value for key, value in metadata.items() if key.endswith('count') or key.endswith('counts')}))


if __name__ == '__main__':
    main()
