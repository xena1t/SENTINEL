"""Build per-window RGB-visibility labels and two-way cross-recording composite pairs (rgb-visibility-v1).

Inputs (all local, no media reads):
  data/windows/manifest.jsonl                         frozen trident-window-v2.1 window grid
  research/windows/visibility-labels/labels.json      per-frame RGB visibility for 129 drone clips
Outputs (data/windows/composite-v1/):
  window-labels.jsonl      every 1 s development window with presence + RGB visibility labels
  composite-manifest.jsonl RGB window from one recording paired with RF+audio from another
  summary.json             counts, input hashes and rules
"""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
WINDOWS = ROOT / 'data/windows/manifest.jsonl'
LABELS = ROOT / 'research/windows/visibility-labels/labels.json'
OUT = ROOT / 'data/windows/composite-v1'
VERSION = 'rgb-visibility-v1'
SEED = 20261022
DURATION_TICKS = 120  # 1 s windows; the manifest uses 120 ticks per second
TICKS_PER_SECOND = 120
ROLES = ('train', 'validation')


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_windows() -> list[dict]:
    rows = [json.loads(line) for line in WINDOWS.read_text().splitlines() if line.strip()]
    windows = [r for r in rows if r['duration_tick'] == DURATION_TICKS]
    if any(r['role'] not in ROLES for r in windows):
        raise ValueError('Window manifest contains a non-development role')
    return windows


def rgb_label(window: dict, clips: dict) -> tuple[str, str]:
    """Background clips inherit not_visible; drone clips use the frame-derived 1 s window label."""
    if window['label'] == 0:
        return 'not_visible', 'inherited_background_clip_label_unscanned'
    clip = clips.get(window['clip_id'])
    if clip is None:
        raise ValueError(f"Missing visibility labels for {window['clip_id']}")
    start_s = round(window['start_tick'] / TICKS_PER_SECOND, 2)
    matches = [w for w in clip['windows_1s_hop_0.25s'] if w['start_s'] == start_s]
    if len(matches) != 1:
        raise ValueError(f"No unique visibility window for {window['window_id']}")
    return matches[0]['rgb_label'], 'frame_level_visual_annotation_ai_assisted'


def window_label_rows(windows: list[dict], clips: dict) -> list[dict]:
    rows = []
    for w in windows:
        visibility, provenance = rgb_label(w, clips)
        rows.append({
            'window_id': w['window_id'], 'clip_id': w['clip_id'], 'group_proxy': w['group_proxy'],
            'role': w['role'], 'start_s': w['start_tick'] / TICKS_PER_SECOND,
            'presence_label': w['label'], 'rgb_visibility': visibility,
            'rgb_visibility_provenance': provenance,
            'use_for_rgb_visibility_loss': visibility in ('visible', 'not_visible'),
            'label_version': VERSION,
        })
    return rows


def cycle(items: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Draw count items, reusing each as evenly as possible (shuffled full passes)."""
    drawn = []
    while len(drawn) < count:
        batch = list(items)
        rng.shuffle(batch)
        drawn.extend(batch)
    return drawn[:count]


def composites_for_role(rows: list[dict], role: str, rng: random.Random) -> list[dict]:
    pool = defaultdict(list)
    for r in rows:
        if r['role'] != role:
            continue
        if r['presence_label'] == 0:
            pool['background'].append(r)
        else:
            pool['drone_any'].append(r)
            if r['rgb_visibility'] in ('visible', 'not_visible'):
                pool[f"drone_{r['rgb_visibility']}"].append(r)
    for key in pool.values():
        key.sort(key=lambda r: r['window_id'])
    negatives = len(pool['drone_not_visible'])
    half = negatives // 2
    plan = [
        ('A_drone_scene_not_visible__background_rf_audio', 0, pool['drone_not_visible'], pool['background'], negatives),
        ('B_background_scene__drone_rf_audio', 1, pool['background'], pool['drone_any'], half),
        ('C_drone_visible__background_rf_audio', 1, pool['drone_visible'], pool['background'], negatives - half),
    ]
    out = []
    for kind, label, rgb_pool, sensor_pool, count in plan:
        rgb = cycle(rgb_pool, count, rng)
        sensors = cycle(sensor_pool, count, rng)
        for i, (a, b) in enumerate(zip(rgb, sensors)):
            if a['clip_id'] == b['clip_id'] or a['presence_label'] == b['presence_label']:
                raise ValueError('Composite must combine two different recordings of different classes')
            out.append({
                'composite_id': f'{VERSION}:{role}:{kind[0]}:{i:05d}', 'type': kind, 'role': role,
                'presence_label': label, 'rgb_window_id': a['window_id'], 'rgb_clip_id': a['clip_id'],
                'rgb_visibility': a['rgb_visibility'], 'rf_audio_window_id': b['window_id'],
                'rf_audio_clip_id': b['clip_id'], 'rf_audio_presence': b['presence_label'],
                'use': 'train' if role == 'train' else 'diagnostic_only', 'label_version': VERSION,
            })
    return out


def real_window_counts(rows: list[dict], role: str) -> dict[str, int]:
    counts = Counter((r['presence_label'], r['rgb_visibility']) for r in rows if r['role'] == role)
    return {f'presence={p}|rgb={v}': n for (p, v), n in sorted(counts.items())}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))


def main() -> None:
    clips = json.loads(LABELS.read_text())['clips']
    windows = load_windows()
    rows = window_label_rows(windows, clips)
    rng = random.Random(SEED)
    composites = [c for role in ROLES for c in composites_for_role(rows, role, rng)]
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / 'window-labels.jsonl', rows)
    write_jsonl(OUT / 'composite-manifest.jsonl', composites)
    summary = {
        'label_version': VERSION, 'seed': SEED, 'window_duration_s': 1.0, 'hop_s': 0.25,
        'inputs_sha256': {'window_manifest': sha256(WINDOWS), 'visibility_labels': sha256(LABELS)},
        'outputs_sha256': {n: sha256(OUT / n) for n in ('window-labels.jsonl', 'composite-manifest.jsonl')},
        'real_windows': {role: real_window_counts(rows, role) for role in ROLES},
        'composites': {role: dict(Counter(c['type'] for c in composites if c['role'] == role)) for role in ROLES},
        'rules': [
            'Only run0 train and validation clips; the 53 observed-test clips are never read.',
            'Pairs combine two different recordings from the same role only.',
            'RGB windows labelled ambiguous (uncertain frames or within 0.25 s of an entry/exit) are never used as composite RGB.',
            'Composite negatives (A) are balanced by composite positives (B + C) so mixing sensors does not predict the label.',
            'Validation composites are diagnostic only; model selection uses real validation windows.',
        ],
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'real_windows': summary['real_windows'], 'composites': summary['composites']}, indent=1))


if __name__ == '__main__':
    main()
