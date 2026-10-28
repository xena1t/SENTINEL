"""Masked sensor fusion and scoped integration into native Qwen visual processing."""
from contextlib import contextmanager

import torch
from torch import nn


class RFEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),nn.Conv2d(64,128,3,2,1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())

    def forward(self, images):
        return self.net(images)


class FusionBlock(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        self.cross = nn.MultiheadAttention(width,8,batch_first=True)
        self.self_attention = nn.MultiheadAttention(width,8,batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(width,2048),nn.GELU(),nn.Linear(2048,width))

    def forward(self, latent, tokens, invalid):
        latent = self.norms[0](latent + self.cross(latent,tokens,tokens,key_padding_mask=invalid,need_weights=False)[0])
        latent = self.norms[1](latent + self.self_attention(latent,latent,latent,need_weights=False)[0])
        return self.norms[2](latent + self.ffn(latent))


class SensorFusion(nn.Module):
    def __init__(self, rgb_dim, lm_dim):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(rgb_dim,512),nn.Linear(128,512),nn.Linear(1024,512)])
        self.modality = nn.Parameter(torch.randn(3,512)*.02)
        self.queries = nn.Parameter(torch.randn(1,16,512)*.02)
        self.blocks = nn.ModuleList([FusionBlock(),FusionBlock()])
        self.connector = nn.Identity() if lm_dim is None else nn.Sequential(nn.LayerNorm(512),nn.Linear(512,512),nn.GELU(),nn.Linear(512,lm_dim))

    def encode(self, streams, available):
        if available.shape != (streams[0].shape[0],3) or not available.any(dim=1).all():
            raise ValueError('Each row requires at least one available sensor')
        projected=[]; invalid=[]
        for i,(features,projection) in enumerate(zip(streams,self.projections)):
            # Suppression precedes projection so hidden NaNs cannot enter attention values.
            valid=available[:,i,None,None]
            features=torch.where(valid,features,torch.zeros_like(features))
            projected.append(projection(features)+self.modality[i])
            invalid.append((~available[:,i,None]).expand(-1,features.shape[1]))
        tokens=torch.cat(projected,dim=1); mask=torch.cat(invalid,dim=1)
        latent=self.queries.expand(tokens.shape[0],-1,-1)
        for block in self.blocks: latent=block(latent,tokens,mask)
        return latent

    def forward(self, streams, available):
        return self.connector(self.encode(streams, available))


@contextmanager
def sensor_embedding_hook(model, positions, tokens):
    embedding=model.get_input_embeddings()
    if getattr(embedding,'_sensor_hook_active',False):
        raise RuntimeError('Overlapping sensor forwards are unsupported')
    def replace(module,args,output):
        if positions.shape != tokens.shape[:2] or output.shape[0] != tokens.shape[0]:
            raise ValueError('Sensor token shape mismatch')
        if positions.min()<0 or positions.max()>=output.shape[1]:
            raise ValueError('Sensor position outside input sequence')
        if not all(torch.unique(row).numel()==row.numel() for row in positions):
            raise ValueError('Duplicate sensor positions')
        return output.scatter(1,positions[:,:,None].expand(-1,-1,output.shape[-1]),tokens.to(output))
    embedding._sensor_hook_active=True
    handle=embedding.register_forward_hook(replace)
    try: yield
    finally:
        handle.remove()
        embedding._sensor_hook_active=False


def conditioned_forward(model, inputs, tokens=None, positions=None):
    if tokens is None:
        return model(**inputs,use_cache=False)
    with sensor_embedding_hook(model,positions,tokens):
        return model(**inputs,use_cache=False)
