"""Train masked intermediate fusion on frozen cached features; test fold excluded."""
import argparse,hashlib,json,random,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,roc_auc_score,confusion_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from drone_presence.rf_data import partition_records
from drone_presence.fusion_baseline import FeatureClips,FusionClassifier,UnimodalHeads,SENSOR_MASKS


def metrics(y,p):
    pred=np.asarray(p)>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),accuracy=float(np.mean(pred==np.asarray(y))),recall=float(tp/(tp+fn)) if tp+fn else None,false_positive_rate=float(fp/(fp+tn)) if fp+tn else None,auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn))

@torch.no_grad()
def predictions(model,loader,device,mask=None):
    model.eval();ys=[];ps=[];ids=[]
    for streams,y,clips in loader:
        streams=[x.to(device) for x in streams]
        if mask is None:logits=model(streams)
        else:logits=model(streams,mask.to(device).expand(len(y),-1))
        ys.extend(y.tolist());ps.extend(logits.sigmoid().cpu().tolist());ids.extend(clips)
    return ys,np.asarray(ps),ids


def main(a):
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4)
    parts=partition_records(a.root);train=parts['train'];val=parts['validation']
    index=json.loads((a.features/'index.json').read_text())
    smoke=json.loads((a.features/'smoke.json').read_text())
    if smoke['status']!='PASS' or smoke['feature_index_sha256']!=hashlib.sha256((a.features/'index.json').read_bytes()).hexdigest():raise RuntimeError('Real-feature smoke must pass on this cache before training')
    split_hash=hashlib.sha256((a.root/'data/audit/trident/curated-evaluation-runs.csv').read_bytes()).hexdigest()
    assert index['split_sha256']==split_hash
    assert not {r['clip_id'] for r in parts['test']} & set(index['clips'])
    loader=lambda rows,shuffle:DataLoader(FeatureClips(a.features,rows),batch_size=8,shuffle=shuffle,num_workers=2)
    training=loader(train,True);validation=loader(val,False);device=a.device
    a.output.mkdir(parents=True,exist_ok=True)
    config=dict(seed=a.seed,epochs=a.epochs,dropout=a.dropout,split_sha256=split_hash,feature_provenance=index,train_ids=[r['clip_id'] for r in train],validation_ids=[r['clip_id'] for r in val],selection='full-sensor validation macro-F1; fixed threshold 0.5; no test evaluation')
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    positive=sum(r['label'] for r in train)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train)-positive)/positive,device=device))
    heads=UnimodalHeads().to(device);optimizer=torch.optim.AdamW(heads.parameters(),lr=.001,weight_decay=1e-4)
    best=[-1.]*3
    # Select each unimodal head independently; late fusion uses those saved heads.
    for epoch in range(1,a.epochs+1):
        heads.train()
        for streams,y,_ in training:
            optimizer.zero_grad(set_to_none=True);logits=heads([x.to(device) for x in streams]);loss=criterion(logits,y.to(device)[:,None].expand(-1,3))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite unimodal loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(heads.parameters(),5,error_if_nonfinite=True);optimizer.step()
        y,p,_=predictions(heads,validation,device)
        for i in range(3):
            score=metrics(y,p[:,i])['macro_f1']
            if score>best[i]:
                best[i]=score;torch.save(dict(state=heads.heads[i].state_dict(),epoch=epoch,metrics=metrics(y,p[:,i])),a.output/f'head-{i}.pt')
    for i in range(3):heads.heads[i].load_state_dict(torch.load(a.output/f'head-{i}.pt',map_location=device,weights_only=True)['state'])
    y,unimodal,ids=predictions(heads,validation,device)
    model=FusionClassifier().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    best=-1.;stale=0
    with (a.output/'history.jsonl').open('w') as history:
        for epoch in range(1,a.epochs+1):
            model.train()
            for streams,target,_ in training:
                streams=[x.to(device) for x in streams];b=len(target)
                mask=torch.ones((b,3),dtype=torch.bool,device=device)
                dropped=torch.rand(b,device=device)<a.dropout
                mask[dropped]=SENSOR_MASKS[:6].to(device)[torch.randint(6,(int(dropped.sum()),),device=device)]
                optimizer.zero_grad(set_to_none=True);loss=criterion(model(streams,mask),target.to(device))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite fusion loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            labels,p,_=predictions(model,validation,device,SENSOR_MASKS[-1]);result=metrics(labels,p)
            row=dict(epoch=epoch,**result);history.write(json.dumps(row)+'\n');history.flush();print(json.dumps(row),flush=True)
            if result['macro_f1']>best:
                best=result['macro_f1'];stale=0;torch.save(dict(state=model.state_dict(),epoch=epoch,metrics=result),a.output/'best.pt')
            else:stale+=1
            if stale>=15:break
    selected=torch.load(a.output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(selected['state'])
    report=dict(selected_epoch=selected['epoch'],unimodal={name:metrics(y,unimodal[:,i]) for i,name in enumerate(['rgb','rf','audio'])},masks={})
    output=[]
    for mask in SENSOR_MASKS:
        key=''.join(str(int(v)) for v in mask)
        labels,p,clip_ids=predictions(model,validation,device,mask)
        late=unimodal[:,mask.numpy()].mean(1)
        report['masks'][key]=dict(fusion=metrics(labels,p),late_fusion=metrics(y,late),groups={})
        group_by_id={r['clip_id']:r['group_proxy'] for r in val}
        for group in sorted(set(group_by_id.values())):
            indices=[i for i,c in enumerate(clip_ids) if group_by_id[c]==group]
            report['masks'][key]['groups'][group]=metrics([labels[i] for i in indices],p[indices])
        for clip,label,prob,avg in zip(clip_ids,labels,p,late):output.append(dict(clip_id=clip,label=int(label),mask=key,fusion_probability=float(prob),late_probability=float(avg)))
    (a.output/'validation-report.json').write_text(json.dumps(report,indent=2))
    (a.output/'validation-predictions.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','features','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--dropout',type=float,default=.5);p.add_argument('--seed',type=int,default=4422);p.add_argument('--device',default='cuda',choices=['cuda','cpu']);a=p.parse_args()
    if not 0<=a.dropout<=1:p.error('dropout must be between 0 and 1')
    main(a)
