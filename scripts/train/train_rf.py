"""RF presence baseline with a mandatory small-subset overfit gate."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from drone_presence.rf_data import RFClips, partition_records
from drone_presence.model import RFEncoder

class RFClassifier(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=RFEncoder();self.head=nn.Linear(128,1)
    def forward(self,x):
        b,t,c,h,w=x.shape
        return self.head(self.encoder(x.reshape(b*t,c,h,w)).reshape(b,t,-1).mean(1)).squeeze(-1)

@torch.no_grad()
def evaluate(model,loader,device,criterion):
    model.eval();labels=[];scores=[];ids=[];total=0.
    for x,y,clip_ids in loader:
        x,y=x.to(device),y.to(device);logits=model(x)
        total+=float(criterion(logits,y))*len(y)
        labels.extend(y.cpu().tolist());scores.extend(logits.sigmoid().cpu().tolist());ids.extend(clip_ids)
    predicted=np.asarray(scores)>=.5
    tn,fp,fn,tp=confusion_matrix(labels,predicted,labels=[0,1]).ravel()
    return dict(loss=total/len(labels),accuracy=accuracy_score(labels,predicted),macro_f1=f1_score(labels,predicted,average='macro',zero_division=0),recall=tp/max(tp+fn,1),false_positive_rate=fp/max(fp+tn,1),auroc=roc_auc_score(labels,scores),average_precision=average_precision_score(labels,scores)),[dict(clip_id=i,label=int(y),probability=p) for i,y,p in zip(ids,labels,scores)]

def main(a):
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    device=a.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    parts=partition_records(a.root)
    training=parts['train'];validation=parts['validation']
    split_hash=hashlib.sha256((a.root/'data/audit/trident/curated-evaluation-runs.csv').read_bytes()).hexdigest()
    cache_hash=hashlib.sha256((a.cache/'index.json').read_bytes()).hexdigest()
    if a.mode=='overfit':
        rng=random.Random(a.seed);training=[]
        for label in (0,1):
            candidates=sorted([r for r in parts['train'] if r['label']==label],key=lambda r:r['clip_id'])
            training.extend(rng.sample(candidates,4))
        validation=training
    else:
        gate=json.loads(a.gate.read_text())
        if gate['status']!='PASS' or gate['split_sha256']!=split_hash or gate['cache_index_sha256']!=cache_hash:
            raise RuntimeError('A passing overfit check on the same data is required')
    a.output.mkdir(parents=True,exist_ok=True)
    def loader(records,shuffle=False):
        return DataLoader(RFClips(a.cache,records),batch_size=a.batch_size,shuffle=shuffle,num_workers=2,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(a.seed))
    train_loader=loader(training,True);eval_loader=loader(validation)
    model=RFClassifier().to(device)
    positives=sum(r['label'] for r in training)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(training)-positives)/positives,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0 if a.mode=='overfit' else 1e-4)
    best=(-1.,float('-inf'));stale=0;consecutive=0;started=time.monotonic()
    config=dict(mode=a.mode,seed=a.seed,lr=a.lr,batch_size=a.batch_size,max_epochs=a.epochs,split_sha256=split_hash,cache_index_sha256=cache_hash,train_ids=[r['clip_id'] for r in training],validation_ids=[r['clip_id'] for r in validation],gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,threshold=.5,architecture='RFEncoder mean across 40 frames + linear clip head',split_limit='folder proxy, not verified independent session')
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    initial,_=evaluate(model,eval_loader,device,criterion)
    with (a.output/'history.jsonl').open('w') as history:
        for epoch in range(1,a.epochs+1):
            model.train()
            for x,y,_ in train_loader:
                optimizer.zero_grad(set_to_none=True);loss=criterion(model(x.to(device)),y.to(device))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            metrics,predictions=evaluate(model,eval_loader,device,criterion)
            row=dict(epoch=epoch,**metrics);history.write(json.dumps(row)+'\n');history.flush();print(json.dumps(row),flush=True)
            key=(metrics['macro_f1'],-metrics['loss'])
            if key>best:
                best=key;stale=0
                torch.save(dict(model=model.state_dict(),encoder=model.encoder.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,metrics=metrics,config=config),a.output/'best.pt')
                (a.output/'best-predictions.json').write_text(json.dumps(predictions,indent=2))
            else:stale+=1
            consecutive=consecutive+1 if metrics['accuracy']==1. and metrics['loss']<.08 else 0
            if a.mode=='overfit' and consecutive>=3:break
            if a.mode=='full' and stale>=15:break
    status='PASS' if a.mode=='full' or consecutive>=3 else 'FAIL'
    report=dict(status=status,**config,initial_metrics=initial,final_metrics=metrics,epochs_completed=epoch,elapsed_s=time.monotonic()-started,peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3 if device=='cuda' else 0,acceptance='8 balanced training clips, accuracy 1.0 and BCE <0.08 for 3 consecutive evaluations' if a.mode=='overfit' else 'completed validation-selected baseline; no test evaluation')
    (a.output/'result.json').write_text(json.dumps(report,indent=2))
    if status!='PASS':raise RuntimeError('Overfit acceptance failed; do not launch full training')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['overfit','full'],required=True);p.add_argument('--gate',type=Path);p.add_argument('--epochs',type=int,default=100);p.add_argument('--lr',type=float,default=.001);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--seed',type=int,default=4422);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--threads',type=int,default=4);main(p.parse_args())
