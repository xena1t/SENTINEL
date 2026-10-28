"""Frozen-Qwen adapter alignment, gated by a balanced eight-clip overfit test."""
import argparse,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from drone_presence.model import SensorFusion
from drone_presence.rf_data import partition_records
from drone_presence.fusion_baseline import FeatureClips,SENSOR_MASKS
from drone_presence.qwen_alignment import AnswerInputs,forward_score,score_yes_no
from train_fusion import metrics


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(model,adapter,builder,dataset,condition='111',seed=4422):
    adapter.eval();predictions=[];donors=list(range(len(dataset)))
    if condition=='shuffled_rf_audio':
        rng=random.Random(seed)
        while True:
            rng.shuffle(donors)
            if all(i!=j for i,j in enumerate(donors)):break
    mask=[1,1,1] if condition=='shuffled_rf_audio' else ([1,0,0] if condition=='camera_only' else [int(x) for x in condition])
    available=torch.tensor([mask],dtype=torch.bool,device='cuda')
    for i in range(len(dataset)):
        streams,label,clip=dataset[i]
        donor=None
        if condition=='shuffled_rf_audio':
            others,_,donor=dataset[donors[i]];streams=[streams[0],others[1],others[2]]
        streams=[x.unsqueeze(0).cuda() for x in streams]
        probability,scores=score_yes_no(model,adapter,builder,clip,streams,available,enabled=condition!='camera_only')
        predictions.append(dict(clip_id=clip,label=int(label),probability=probability,no_log_probability=scores[0],yes_log_probability=scores[1],donor_clip_id=donor))
    result=metrics([r['label'] for r in predictions],[r['probability'] for r in predictions])
    result['conditional_nll']=float(np.mean([np.logaddexp(r['no_log_probability'],r['yes_log_probability'])-r['yes_log_probability' if r['label'] else 'no_log_probability'] for r in predictions]))
    result['answer_nll']=float(np.mean([-r['yes_log_probability' if r['label'] else 'no_log_probability'] for r in predictions]))
    return result,predictions


def main(a):
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    a.output.mkdir(parents=True,exist_ok=True)
    parts=partition_records(a.root);rows=parts['train'];validation=parts['validation']
    provenance=dict(split_sha256=digest(a.root/'data/audit/trident/curated-evaluation-runs.csv'),feature_index_sha256=digest(a.features/'index.json'),fusion_checkpoint_sha256=digest(a.fusion_checkpoint),models_sha256=digest(a.models))
    feature_index=json.loads((a.features/'index.json').read_text())
    assert feature_index['complete'] and feature_index['split_sha256']==provenance['split_sha256']
    assert not set(feature_index['clips']) & {r['clip_id'] for r in parts['test']}
    if a.mode in ('overfit','smoke'):
        rng=random.Random(a.seed);rows=[]
        for label in (0,1):rows.extend(rng.sample(sorted([r for r in parts['train'] if r['label']==label],key=lambda r:r['clip_id']),4))
        validation=rows
    else:
        gate=json.loads(a.gate.read_text())
        if gate['status']!='PASS' or gate['provenance']!=provenance:raise RuntimeError('Matching overfit PASS required')
    train=FeatureClips(a.features,rows);val=FeatureClips(a.features,validation)
    versions=json.loads(a.models.read_text())
    model=Qwen3VLForConditionalGeneration.from_pretrained(versions['qwen_path'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False).cuda()
    processor=AutoProcessor.from_pretrained(versions['qwen_path'])
    builder=AnswerInputs(processor,a.root,rows+validation,model.config.vision_end_token_id)
    adapter=SensorFusion(4096,model.config.text_config.hidden_size).cuda()
    checkpoint=torch.load(a.fusion_checkpoint,map_location='cpu',weights_only=True)
    state={k.removeprefix('fusion.'):v for k,v in checkpoint['state'].items() if k.startswith('fusion.')}
    transferred=adapter.load_state_dict(state,strict=False)
    assert not transferred.unexpected_keys and transferred.missing_keys and all(k.startswith('connector.') for k in transferred.missing_keys)
    if a.freeze_fusion:
        adapter.requires_grad_(False)
        adapter.connector.requires_grad_(True)
    frozen_before={n:p.detach().cpu().clone() for n,p in adapter.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4)
    total_steps=math.ceil(len(train)/a.accumulation)*a.epochs
    warmup_steps=math.ceil(total_steps*a.warmup_fraction)
    def lr_scale(step):
        if step<warmup_steps:return (step+1)/max(1,warmup_steps)
        progress=(step-warmup_steps)/max(1,total_steps-warmup_steps)
        return .1+.9*.5*(1+math.cos(math.pi*min(1.,progress)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale) if a.warmup_fraction else None
    mask_rng=random.Random(a.seed+1)
    incomplete=['100','010','001','110','101','011']
    config=dict(mode=a.mode,seed=a.seed,lr=a.lr,epochs=a.epochs,accumulation=a.accumulation,provenance=provenance,train_ids=[r['clip_id'] for r in rows],validation_ids=[r['clip_id'] for r in validation],fusion_initialization_epoch=checkpoint['epoch'],gpu=torch.cuda.get_device_name(0),objective='lexical answer token NLL; no EOS or prompt loss',scoring='softmax of summed no/yes answer log probabilities; fixed 0.5 threshold; not calibrated',frozen='Qwen, camera/RF/audio encoders',dropout=a.modality_dropout,freeze_fusion=a.freeze_fusion,warmup_fraction=a.warmup_fraction,scheduler='warmup_cosine_to_10pct' if scheduler else 'constant',trainable_parameters=sum(p.numel() for p in adapter.parameters() if p.requires_grad),selection='full-sensor macro-F1 then conditional NLL',patience=a.patience)
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    if a.mode=='full':
        baseline,pred=evaluate(model,adapter,builder,val,'camera_only')
        (a.output/'camera-only.json').write_text(json.dumps(dict(metrics=baseline,predictions=pred),indent=2))
    initial,_=evaluate(model,adapter,builder,val)
    print('Initial',json.dumps(initial),flush=True)
    best=(-1.,float('-inf'));stale=0;passes=0;gradient_verified=False;started=time.monotonic()
    with (a.output/'history.jsonl').open('w') as history:
        for epoch in range(1,a.epochs+1):
            adapter.train();mask_counts={k:0 for k in ['111']+incomplete};train_losses=[];order=torch.randperm(len(train)).tolist();optimizer.zero_grad(set_to_none=True)
            for step,i in enumerate(order):
                streams,label,clip=train[i];streams=[x.unsqueeze(0).cuda() for x in streams]
                mask=(['111']+incomplete)[step%7] if a.mode=='smoke' else (mask_rng.choice(incomplete) if mask_rng.random()<a.modality_dropout else '111')
                mask_counts[mask]+=1
                available=torch.tensor([[int(x) for x in mask]],dtype=torch.bool,device='cuda')
                tokens=adapter(streams,available)
                answer='yes' if int(label) else 'no';inputs,positions=builder.build(clip,answer,camera=bool(available[0,0]))
                denominator=min(a.accumulation,len(order)-(step//a.accumulation)*a.accumulation)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=-forward_score(model,inputs,positions,tokens).mean()/len(builder.answers[answer])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite answer loss')
                train_losses.append(float(loss.detach()))
                (loss/denominator).backward()
                del tokens,inputs,loss
                if (step+1)%a.accumulation==0 or step+1==len(order):
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True)
                    if not gradient_verified:
                        for prefix in (('connector',) if a.freeze_fusion else ('projections.0','projections.1','projections.2','queries','connector')):
                            assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in adapter.named_parameters() if n.startswith(prefix)),prefix
                        assert all(p.grad is None for p in model.parameters())
                        assert all(p.grad is None for p in adapter.parameters() if not p.requires_grad)
                        before=adapter.connector[-1].weight.detach().clone()
                    optimizer.step()
                    if scheduler:scheduler.step()
                    if not gradient_verified:
                        assert not torch.equal(before,adapter.connector[-1].weight);gradient_verified=True;del before
                    optimizer.zero_grad(set_to_none=True)
            result,pred=evaluate(model,adapter,builder,val)
            entry=dict(epoch=epoch,train_loss=float(np.mean(train_losses)),mask_counts=mask_counts,lr=optimizer.param_groups[0]['lr'],**result);history.write(json.dumps(entry)+'\n');history.flush();print(json.dumps(entry),flush=True)
            key=(result['macro_f1'],-result['conditional_nll'])
            if key>best:
                best=key;stale=0;torch.save(dict(adapter=adapter.state_dict(),epoch=epoch,metrics=result,config=config,optimizer=optimizer.state_dict()),a.output/'best.pt')
                (a.output/'best-predictions.json').write_text(json.dumps(pred,indent=2))
            else:stale+=1
            passes=passes+1 if result['accuracy']==1. and result['conditional_nll']<.08 and result['answer_nll']<.15 else 0
            if a.mode=='overfit' and passes>=3:break
            if a.mode=='full' and stale>=a.patience:break
    for name,param in adapter.named_parameters():
        if name in frozen_before:assert torch.equal(param.detach().cpu(),frozen_before[name]),name
    if a.mode=='smoke':assert all(mask_counts.values()) and gradient_verified
    status='PASS' if a.mode in ('full','smoke') or (passes>=3 and gradient_verified) else 'FAIL'
    selected=torch.load(a.output/'best.pt',map_location='cpu',weights_only=True)
    report=dict(status=status,provenance=provenance,initial_metrics=initial,selected_epoch=selected['epoch'],selected_metrics=selected['metrics'],final_metrics=result,epochs_completed=epoch,gradient_verified=gradient_verified,frozen_fusion_unchanged=a.freeze_fusion,elapsed_s=time.monotonic()-started,peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
    if a.mode=='full':
        adapter.load_state_dict(selected['adapter']);conditions={}
        for condition in ['shuffled_rf_audio','100','010','001','110','101','011']:
            score,pred=evaluate(model,adapter,builder,val,condition)
            conditions[condition]=score;(a.output/f'predictions-{condition}.json').write_text(json.dumps(pred,indent=2))
        report['conditions']=conditions
    (a.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if status!='PASS':raise RuntimeError('Overfit gate failed; do not submit full alignment')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','features','fusion-checkpoint','models','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gate',type=Path);p.add_argument('--mode',choices=['overfit','full','smoke'],required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--accumulation',type=int,default=4);p.add_argument('--seed',type=int,default=4422)
    p.add_argument('--freeze-fusion',action='store_true');p.add_argument('--modality-dropout',type=float,default=0.);p.add_argument('--warmup-fraction',type=float,default=0.);p.add_argument('--patience',type=int,default=5)
    args=p.parse_args()
    if not 0<=args.modality_dropout<=1 or not 0<=args.warmup_fraction<1 or args.accumulation<1 or args.epochs<1: p.error('Invalid training configuration')
    main(args)
