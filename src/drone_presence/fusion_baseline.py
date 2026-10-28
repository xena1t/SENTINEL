"""Frozen-feature presence classifier with explicit sensor availability."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset
from .model import SensorFusion

SENSOR_MASKS = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]],dtype=torch.bool)

class FeatureClips(Dataset):
    def __init__(self, directory, records):
        self.directory=Path(directory);self.records=records
        self.index=json.loads((self.directory/'index.json').read_text())
        if not self.index['complete']:raise ValueError('Feature extraction is incomplete')
        for r in records:
            if r['clip_id'] not in self.index['clips']:raise ValueError('Missing clip features')
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        features=torch.load(self.directory/self.index['clips'][r['clip_id']],map_location='cpu',weights_only=True)
        streams=[features[k].float() for k in ('rgb','rf','audio')]
        for x,shape in zip(streams,[(128,4096),(40,128),(2,1024)]):
            if tuple(x.shape)!=shape or not torch.isfinite(x).all():raise ValueError('Invalid cached features')
        return streams,torch.tensor(r['label'],dtype=torch.float32),r['clip_id']

class FusionClassifier(nn.Module):
    def __init__(self):
        super().__init__();self.fusion=SensorFusion(4096,None);self.head=nn.Linear(512,1)
    def forward(self,streams,available):
        return self.head(self.fusion.encode(streams,available).mean(1)).squeeze(-1)

class UnimodalHeads(nn.Module):
    def __init__(self):
        super().__init__();self.heads=nn.ModuleList([nn.Linear(d,1) for d in (4096,128,1024)])
    def forward(self,streams):
        return torch.stack([head(x.mean(1)).squeeze(-1) for head,x in zip(self.heads,streams)],dim=1)
