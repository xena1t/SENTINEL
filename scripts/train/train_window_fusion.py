"""rgb-visibility-v1 masked fusion on frozen one-second features; selection on real validation windows only."""
import argparse,hashlib,json,random,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,WeightedRandomSampler
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from drone_presence.fusion_baseline import FusionClassifier,UnimodalHeads,SENSOR_MASKS
from drone_presence.composite_windows import load_samples,partition,sampling_weights,expected_positive_rate,WindowFeatures,sha256,SLICES
from train_fusion import metrics,predictions


def slice_metrics(rows,p):
    """Per-slice counts and accuracy; single-class slices have no macro-F1 or AUROC."""
    out={}
    for name in sorted({r['slice'] for r in rows}):
        idx=[i for i,r in enumerate(rows) if r['slice']==name];y=np.asarray([rows[i]['presence_label'] for i in idx]);pred=np.asarray(p)[idx]>=.5
        out[name]=dict(count=len(idx),accuracy=float(np.mean(pred==y)),predicted_positive=int(pred.sum()))
    return out


def grouped_interval(rows,p,replicates=1000,seed=4422):
    """Recording-clustered bootstrap: overlapping windows of one recording are resampled together."""
    by_clip=defaultdict(list)
    for i,r in enumerate(rows):by_clip[r['rgb_clip_id']].append(i)
    clips=sorted(by_clip);rng=np.random.default_rng(seed);y=np.asarray([r['presence_label'] for r in rows]);pred=np.asarray(p)>=.5
    accuracy=[]
    for _ in range(replicates):
        idx=np.concatenate([by_clip[clips[j]] for j in rng.integers(len(clips),size=len(clips))])
        accuracy.append(float(np.mean(pred[idx]==y[idx])))
    return dict(recordings=len(clips),replicates=replicates,seed=seed,accuracy_95=[float(np.percentile(accuracy,2.5)),float(np.percentile(accuracy,97.5))])


def main(a):
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4)
    samples=load_samples(a.root);train,val,diagnostic=partition(samples)
    if a.mode=='overfit':
        rng=random.Random(a.seed);pick=lambda pool,n:rng.sample(sorted(pool,key=lambda r:r['sample_id']),n)
        train=(pick([r for r in train if r['kind']=='real' and r['presence_label']==0],2)+pick([r for r in train if r['kind']=='real' and r['presence_label']==1],2)
               +pick([r for r in train if r['composite_type']=='A'],2)+pick([r for r in train if r['composite_type']=='B'],1)+pick([r for r in train if r['composite_type']=='C'],1))
        val=train;diagnostic=[]
    index_hash=sha256(a.features/'index.json');index=json.loads((a.features/'index.json').read_text())
    weights=sampling_weights(train,a.composite_share) if a.mode=='full' else torch.ones(len(train),dtype=torch.double)
    draws=a.draws or len(train)*(2 if a.mode=='full' else 1)
    generator=torch.Generator().manual_seed(a.seed)
    training=DataLoader(WindowFeatures(a.features,train),batch_size=a.batch_size,sampler=WeightedRandomSampler(weights,draws,replacement=True,generator=generator),num_workers=4,persistent_workers=True)
    evaluation=lambda rows:DataLoader(WindowFeatures(a.features,rows),batch_size=64,shuffle=False,num_workers=4)
    validation=evaluation(val);device=a.device
    a.output.mkdir(parents=True,exist_ok=True)
    positive=expected_positive_rate(train,weights)
    config=dict(mode=a.mode,seed=a.seed,epochs=a.epochs,dropout=a.dropout,lr=a.lr,batch_size=a.batch_size,draws_per_epoch=draws,
                sampler=dict(composite_share=a.composite_share,type_ratio='A:B:C=2:1:1',expected_positive_rate=positive) if a.mode=='full' else 'uniform overfit subset',
                features_index_sha256=index_hash,feature_provenance={k:v for k,v in index.items() if k!='clips'},
                code_sha256={p:sha256(Path(__file__).resolve().parents[2]/p) for p in ('scripts/train/train_window_fusion.py','src/drone_presence/composite_windows.py','src/drone_presence/fusion_baseline.py','src/drone_presence/model.py')},
                counts=dict(train=len(train),validation_real=len(val),validation_composite=len(diagnostic)),
                selection='full-sensor real-validation macro-F1; fixed threshold 0.5; composites and test never used for selection',
                dropout_recipe='after assembling each sample: all sensors with probability 1-dropout, else one of six incomplete masks uniformly')
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-positive)/positive,device=device))
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
    model=FusionClassifier().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    best=-1.;stale=0;consecutive=0
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
                best=result['macro_f1'];stale=0;torch.save(dict(state=model.state_dict(),epoch=epoch,metrics=result,config=config),a.output/'best.pt')
            else:stale+=1
            consecutive=consecutive+1 if result['accuracy']==1. else 0
            if a.mode=='overfit' and consecutive>=3:break
            if a.mode=='full' and stale>=a.patience:break
    if a.mode=='overfit':
        status='PASS' if consecutive>=3 else 'FAIL'
        (a.output/'result.json').write_text(json.dumps(dict(status=status,epochs=epoch,final=result,config=config),indent=2))
        if status!='PASS':raise RuntimeError('Fusion overfit gate failed')
        return
    selected=torch.load(a.output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(selected['state'])
    report=dict(selected_epoch=selected['epoch'],best_sha256=sha256(a.output/'best.pt'),config=config,real_validation={},composite_diagnostic={})
    output=[]
    for rows,name in ((val,'real_validation'),(diagnostic,'composite_diagnostic')):
        loader=evaluation(rows);_,unimodal,ids=predictions(heads,loader,device);assert ids==[r['sample_id'] for r in rows]
        report[name]['unimodal']={sensor:metrics([r['presence_label'] for r in rows],unimodal[:,i]) for i,sensor in enumerate(['rgb','rf','audio'])}
        for mask in SENSOR_MASKS:
            key=''.join(str(int(v)) for v in mask)
            labels,p,ids=predictions(model,loader,device,mask);late=unimodal[:,mask.numpy()].mean(1)
            entry=dict(fusion=metrics(labels,p),late_fusion=metrics(labels,late),fusion_slices=slice_metrics(rows,p),late_slices=slice_metrics(rows,late))
            if name=='real_validation':entry['fusion_recording_bootstrap']=grouped_interval(rows,p)
            report[name][key]=entry
            for r,prob,avg in zip(rows,p,late):output.append(dict(sample_id=r['sample_id'],set=name,slice=r['slice'],label=r['presence_label'],mask=key,fusion_probability=float(prob),late_probability=float(avg)))
    (a.output/'validation-report.json').write_text(json.dumps(report,indent=2))
    (a.output/'validation-predictions.json').write_text(json.dumps(output))
    print(json.dumps({k:report['real_validation'][k]['fusion'] for k in report['real_validation'] if k!='unimodal'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','features','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=['overfit','full'],default='full');p.add_argument('--epochs',type=int,default=60);p.add_argument('--patience',type=int,default=15)
    p.add_argument('--dropout',type=float,default=.5);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--composite-share',type=float,default=.5);p.add_argument('--draws',type=int,default=0);p.add_argument('--seed',type=int,default=4422);p.add_argument('--device',default='cuda',choices=['cuda','cpu']);a=p.parse_args()
    if not 0<=a.dropout<=1 or not 0<a.composite_share<1:p.error('dropout must be in [0,1] and composite share in (0,1)')
    main(a)
