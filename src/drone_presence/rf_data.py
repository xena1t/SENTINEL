"""Clip-level RF data; publisher directories never determine evaluation roles."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def partition_records(root, run=0):
    root = Path(root)
    manifest = json.loads((root / 'data/trident/local-presence-manifest.json').read_text())
    records = {r['clip_id']: r for r in manifest}
    if len(records) != len(manifest):
        raise ValueError('Duplicate manifest clip')
    with (root / 'data/audit/trident/curated-evaluation-runs.csv').open() as f:
        assignments = [r for r in csv.DictReader(f) if int(r['run']) == run]
    if len({r['clip_id'] for r in assignments}) != len(assignments) or {r['clip_id'] for r in assignments} != set(records):
        raise ValueError('Split must assign every clip exactly once')
    parts = {role: [] for role in ('train', 'validation', 'test')}
    groups = {}
    for assignment in assignments:
        record = records[assignment['clip_id']]
        role = assignment['role']
        if record['label'] != {'Drone': 1, 'Background': 0}[assignment['label']]:
            raise ValueError('Split/manifest labels disagree')
        group = assignment['candidate_group']
        if groups.setdefault(group, role) != role:
            raise ValueError('Group proxy crosses partitions')
        if len(record['rf_paths']) != 40 or len(set(record['rf_paths'])) != 40:
            raise ValueError('Expected 40 distinct RF paths per clip')
        parts[role].append(record)
    if any({r['label'] for r in rows} != {0, 1} for rows in parts.values()):
        raise ValueError('Every partition must contain both classes')
    return parts


class RFClips(Dataset):
    def __init__(self, cache, records):
        self.cache = Path(cache)
        self.records = records
        index = json.loads((self.cache / 'index.json').read_text())
        self.paths = index['clips']
        if index['shape'] != [40, 3, 224, 224]:
            raise ValueError('Unsupported preprocessing shape')
        for record in records:
            if record['clip_id'] not in self.paths:
                raise ValueError('Requested clip absent from cache')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        array = np.load(self.cache / self.paths[record['clip_id']], allow_pickle=False)
        if array.shape != (40, 3, 224, 224) or array.dtype != np.uint8:
            raise ValueError('Corrupt RF cache')
        return torch.from_numpy(array).float().div_(255), torch.tensor(record['label'], dtype=torch.float32), record['clip_id']
