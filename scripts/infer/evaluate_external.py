"""Metrics for one run_pipeline.py predictions JSON, per mask and model, as fixed in research/external-test-protocol.md.

Adds a parity section for TRIDENT: row-by-row comparison with the saved Version 2 validation predictions.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/train'))
from train_fusion import metrics
from drone_presence.test_windows_v2 import load as load_test, verify_predictions, expected_protocol
from drone_presence.windows import sha256

SAVED = ROOT / 'research/windows/rgb-visibility-v1-220956/raw'
MODELS = {'fusion': 'fusion_probability', 'qwen_dropout': 'probability'}


def grouped_bootstrap(rows, key, replicates=1000, seed=4422, *, full_metrics=False):
    by_recording = defaultdict(list)
    for i, r in enumerate(rows):
        by_recording[r['recording']].append(i)
    recordings = sorted(by_recording)
    y = np.asarray([r['label'] for r in rows])
    probabilities = np.asarray([r[key] for r in rows])
    pred = probabilities >= .5
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(replicates):
        idx = np.concatenate([by_recording[recordings[j]] for j in rng.integers(len(recordings), size=len(recordings))])
        if full_metrics:
            draw = metrics(y[idx], probabilities[idx])
            for name in ('macro_f1', 'accuracy', 'recall', 'false_positive_rate', 'auroc'):
                if draw[name] is not None:
                    values[name].append(draw[name])
            continue
        values['accuracy'].append(float(np.mean(pred[idx] == y[idx])))
        positives, negatives = idx[y[idx] == 1], idx[y[idx] == 0]
        if len(positives):
            values['recall'].append(float(pred[positives].mean()))
        if len(negatives):
            values['false_positive_rate'].append(float(pred[negatives].mean()))
    interval = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in values.items()}
    result = {'recordings': len(recordings), 'replicates': replicates, 'seed': seed, 'ci95': interval}
    if full_metrics:
        result.update(method='percentile_recording_resample_with_replacement',
                      valid_replicates={key: len(value) for key, value in values.items()})
    return result


def parity(rows):
    fusion = {(r['sample_id'], r['mask']): r['fusion_probability']
              for r in json.loads((SAVED / 'fusion/validation-predictions.json').read_text()) if r['set'] == 'real_validation'}
    result = {}
    for mask in sorted({r['mask'] for r in rows}):
        current = [r for r in rows if r['mask'] == mask]
        qwen_path = SAVED / f'alignment-dropout/predictions-{mask}.json'
        qwen = {r['sample_id']: r['probability'] for r in json.loads(qwen_path.read_text())} if qwen_path.exists() else {}
        entry = {}
        for name, saved, key in (('fusion', {k[0]: v for k, v in fusion.items() if k[1] == mask}, 'fusion_probability'),
                                 ('qwen_dropout', qwen, 'probability')):
            pairs = [(r[key], saved[r['sample_id']]) for r in current if key in r and r['sample_id'] in saved]
            if not pairs:
                continue
            new, old = np.asarray(pairs).T
            entry[name] = {'rows': len(pairs), 'max_abs_difference': float(np.abs(new - old).max()),
                           'mean_abs_difference': float(np.abs(new - old).mean()),
                           'decision_changes': int(((new >= .5) != (old >= .5)).sum()),
                           'saved_metrics': metrics([r['label'] for r in current if r['sample_id'] in saved], old)}
        result[mask] = entry
    return result


def main(a):
    report = json.loads(a.predictions.read_text())
    rows = report['rows']
    if report['dataset'] == 'trident-test':
        _, expected_rows, _ = load_test(ROOT)
        if report.get('phase') not in ('learned_fusion', 'learned_fusion_and_qwen_dropout'):
            raise ValueError('Unknown sealed test prediction phase')
        require_qwen = report['phase'] == 'learned_fusion_and_qwen_dropout'
        verify_predictions(report, expected_rows, require_qwen=require_qwen)
        protocol = json.loads(a.protocol.read_text())
        for key, value in expected_protocol(ROOT).items():
            if protocol.get(key) != value:
                raise ValueError(f'Sealed test protocol differs before scoring: {key}')
        seal = report['test_seal']
        if seal.get('protocol_sha256') != sha256(a.protocol) or seal.get('contract_sha256') != protocol['contract_sha256'] or seal.get('code_sha256') != protocol['code_sha256']:
            raise ValueError('Prediction seal differs from the scoring protocol')
        provenance = report['provenance']
        names = protocol['checkpoints'] if require_qwen else ('rf-cnn-best', 'fusion-best')
        expected_checkpoints = {name: protocol['checkpoints'][name] for name in names}
        if provenance['checkpoints'] != expected_checkpoints or any(provenance[key] != protocol[key] for key in ('qwen_revision', 'clap_revision', 'qwen_load')):
            raise ValueError('Prediction models differ from the scoring protocol')
        if a.output.exists():
            raise ValueError('Test metrics output exists; refusing overwrite')
    output = {'dataset': report['dataset'], 'windows': report['windows'], 'limit': report['limit'],
              'provenance': report['provenance'], 'code_sha256': report['code_sha256'], 'masks': {}}
    for mask in report['masks']:
        current = [r for r in rows if r['mask'] == mask]
        output['masks'][mask] = {}
        for name, key in MODELS.items():
            if key not in current[0]:
                continue
            labels = [r['label'] for r in current]
            output['masks'][mask][name] = {**metrics(labels, [r[key] for r in current]),
                                           'bootstrap': grouped_bootstrap(current, key, full_metrics=report['dataset'] == 'trident-test')}
    if report['dataset'] == 'trident-test':
        output['test_seal'] = report['test_seal']
        output['threshold'] = .5
        output['phase'] = report['phase']
    if report['dataset'] == 'trident':
        output['parity'] = parity(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x' if report['dataset'] == 'trident-test' else 'w') as stream:
        stream.write(json.dumps(output, indent=2) + '\n')
    for mask, models in output['masks'].items():
        print(mask, {n: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
                         if k in ('macro_f1', 'recall', 'false_positive_rate', 'auroc')} for n, m in models.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT / 'research/test-v2/protocol.json')
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
