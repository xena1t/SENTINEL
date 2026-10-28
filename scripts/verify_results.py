"""Recompute the saved V1, V2 and external results from prediction probabilities."""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASKS = ('100', '010', '001', '110', '101', '011', '111')


def read(path):
    return json.loads((ROOT / path).read_text())


def score(rows, key):
    values = sorted((float(row[key]), int(row['label'])) for row in rows)
    tp = sum(p >= .5 and label == 1 for p, label in values)
    tn = sum(p < .5 and label == 0 for p, label in values)
    fp = sum(p >= .5 and label == 0 for p, label in values)
    fn = sum(p < .5 and label == 1 for p, label in values)
    positives, negatives = tp + fn, tn + fp
    positive_rank_sum, start = 0, 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end][0] == values[start][0]:
            end += 1
        positive_rank_sum += sum(label for _, label in values[start:end]) * (start + end + 1) / 2
        start = end
    return {
        'n': len(values), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': (tp + tn) / len(values),
        'macro_f1': (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2,
        'recall': tp / positives, 'false_positive_rate': fp / negatives,
        'auroc': (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives),
    }


def check(tag, rows, key, expected):
    result = score(rows, key)
    result['max_abs_saved_metric_error'] = max(
        abs(value - expected[name]) for name, value in result.items() if name in expected)
    assert result['max_abs_saved_metric_error'] < 1e-12, tag
    return {'tag': tag, **result}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, help='Optionally save the recomputed metrics as JSON')
args = parser.parse_args()
checks = []

for name in ('fusion', 'qwen'):
    report = read(f'research/test-v2/{name}-predictions.json')
    saved = read(f'research/test-v2/{name}-metrics.json')['masks']
    model, key = ('fusion', 'fusion_probability') if name == 'fusion' else ('qwen_dropout', 'probability')
    for mask in report['masks']:
        rows = [row for row in report['rows'] if row['mask'] == mask]
        checks.append(check(f'test_v2/{model}/{mask}', rows, key, saved[mask][model]))

for cohort in ('rfvs', 'halmstad-video', 'halmstad-audio'):
    report = read(f'research/external/{cohort}/predictions.json')
    saved = read(f'research/external/{cohort}/metrics.json')['masks']
    for mask in report['masks']:
        rows = [row for row in report['rows'] if row['mask'] == mask]
        for model, key in (('fusion', 'fusion_probability'), ('qwen_dropout', 'probability')):
            checks.append(check(f'external/{cohort}/{model}/{mask}', rows, key, saved[mask][model]))

base = 'research/windows/rgb-visibility-v1-220956/raw'
predictions = read(f'{base}/fusion/validation-predictions.json')
saved = read(f'{base}/fusion/validation-report.json')['real_validation']
for mask in MASKS:
    rows = [row for row in predictions if row['mask'] == mask and row['set'] == 'real_validation']
    for model, key in (('fusion', 'fusion_probability'), ('late_fusion', 'late_probability')):
        checks.append(check(f'validation_v2/{model}/{mask}', rows, key, saved[mask][model]))

for arm in ('control', 'dropout'):
    directory = f'{base}/alignment-{arm}'
    saved = read(f'{directory}/result.json')['real_validation']
    for mask in MASKS:
        rows = read(f'{directory}/predictions-{mask}.json')
        checks.append(check(f'validation_v2/qwen_{arm}/{mask}', rows, 'probability', saved[mask]))
    native = read(f'{directory}/camera-only.json')
    checks.append(check(f'validation_v2/camera_qwen_{arm}', native['predictions'], 'probability', native['metrics']))

predictions = read('research/evaluation/test-218024/predictions.json')
saved = read('research/evaluation/test-218024/metrics.json')
for model, mask in sorted({(row['model'], row['mask']) for row in predictions}):
    rows = [row for row in predictions if (row['model'], row['mask']) == (model, mask)]
    checks.append(check(f'test_v1/{model}/{mask}', rows, 'probability', saved[f'{model}:{mask}']))

assert len(checks) == 94
for cohort in ('test_v2', 'validation_v2', 'external', 'test_v1'):
    selected = [row for row in checks if row['tag'].startswith(cohort + '/')]
    print(cohort, 'conditions', len(selected), 'max_error', max(row['max_abs_saved_metric_error'] for row in selected))

datasets = (
    ('real', 'data/windows/manifest.jsonl', 8214),
    ('labels', 'data/windows/composite-v1/window-labels.jsonl', 8214),
    ('composites', 'data/windows/composite-v1/composite-manifest.jsonl', 2918),
    ('test', 'data/windows/test-v2/manifest.jsonl', 1961),
)
for name, path, expected_count in datasets:
    rows = [json.loads(line) for line in (ROOT / path).read_text().splitlines()]
    if name == 'real':
        rows = [row for row in rows if row['duration_tick'] == 120]
    assert len(rows) == expected_count, name
    print(name, len(rows), dict(Counter(row['role'] for row in rows)))

for model in ('late_fusion', 'learned_fusion', 'stable_control', 'stable_dropout'):
    assert next(row for row in checks if row['tag'] == f'test_v1/{model}/111')['accuracy'] == 1
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2) + '\n')
print('PASS: 94 saved conditions, split/window/composite counts and four perfect 10-second test models')
