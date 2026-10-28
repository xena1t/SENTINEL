"""rgb-visibility-v1 frozen-Qwen connector alignment on one-second windows; real-validation selection only."""
import argparse,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from drone_presence.model import SensorFusion,conditioned_forward
from drone_presence.fusion_baseline import SENSOR_MASKS
from drone_presence.qwen_alignment import AnswerInputs,forward_score,normalized_yes_probability
from drone_presence.composite_windows import load_samples,partition,sampling_weights,WindowFeatures,sha256
from train_fusion import metrics
from train_window_fusion import slice_metrics,grouped_interval

INCOMPLETE=['100','010','001','110','101','011']


@torch.no_grad()
def score(model,adapter,builder,sample_id,streams,available,enabled=True):
    """Both one-token answers share the prompt, so one forward yields the no and yes log probabilities."""
    camera=bool(available[0,0]);tokens=adapter(streams,available) if enabled else None
    inputs,positions=builder.build(sample_id,'no',camera=camera,sensors=enabled)
    labels=inputs.pop('labels')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=conditioned_forward(model,inputs,tokens,positions).logits
    answer=(labels[0]!=-100).nonzero().flatten();assert len(answer)==1
    log_probs=F.log_softmax(logits[0,answer[0]-1].float(),-1)
    no,yes=log_probs[builder.answers['no'][0]],log_probs[builder.answers['yes'][0]]
    return float(normalized_yes_probability(no,yes)),[float(no),float(yes)]


def evaluate(model,adapter,builder,dataset,rows,condition='111'):
    adapter.eval();predictions=[]
    mask=[1,0,0] if condition=='camera_only' else [int(x) for x in condition]
    available=torch.tensor([mask],dtype=torch.bool,device='cuda')
    for i,r in enumerate(rows):
        streams,label,sample_id=dataset[i];assert sample_id==r['sample_id']
        probability,scores=score(model,adapter,builder,sample_id,[x.unsqueeze(0).cuda() for x in streams],available,enabled=condition!='camera_only')
        predictions.append(dict(sample_id=sample_id,slice=r['slice'],label=int(label),probability=probability,no_log_probability=scores[0],yes_log_probability=scores[1]))
    result=metrics([p['label'] for p in predictions],[p['probability'] for p in predictions])
    result['conditional_nll']=float(np.mean([np.logaddexp(p['no_log_probability'],p['yes_log_probability'])-p['yes_log_probability' if p['label'] else 'no_log_probability'] for p in predictions]))
    result['answer_nll']=float(np.mean([-p['yes_log_probability' if p['label'] else 'no_log_probability'] for p in predictions]))
    result['slices']=slice_metrics(rows,[p['probability'] for p in predictions])
    return result,predictions


def main(a):
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    a.output.mkdir(parents=True,exist_ok=True)
    samples=load_samples(a.root);train,val,diagnostic=partition(samples)
    provenance=dict(features_index_sha256=sha256(a.features/'index.json'),fusion_checkpoint_sha256=sha256(a.fusion_checkpoint),models_sha256=sha256(a.models),
                    composite_summary_sha256=sha256(a.root/'data/windows/composite-v1/summary.json'))
    if a.mode in ('overfit','smoke'):
        rng=random.Random(a.seed);pick=lambda pool,n:rng.sample(sorted(pool,key=lambda r:r['sample_id']),n)
        train=(pick([r for r in train if r['kind']=='real' and r['presence_label']==0],2)+pick([r for r in train if r['kind']=='real' and r['presence_label']==1],2)
               +pick([r for r in train if r['composite_type']=='A'],2)+pick([r for r in train if r['composite_type']=='B'],1)+pick([r for r in train if r['composite_type']=='C'],1))
        val=train;diagnostic=[]
    else:
        gate=json.loads(a.gate.read_text())
        if gate['status']!='PASS' or gate['provenance']!=provenance:raise RuntimeError('Matching overfit PASS required')
    train_set=WindowFeatures(a.features,train);val_set=WindowFeatures(a.features,val)
    versions=json.loads(a.models.read_text())
    model=Qwen3VLForConditionalGeneration.from_pretrained(versions['qwen_path'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False).cuda()
    processor=AutoProcessor.from_pretrained(versions['qwen_path'])
    builder=AnswerInputs(processor,a.root,[dict(clip_id=r['sample_id'],rgb_paths=r['rgb_paths']) for r in train+val+diagnostic],model.config.vision_end_token_id)
    if any(len(ids)!=1 for ids in builder.answers.values()):raise RuntimeError('Single-forward scoring needs one-token answers')
    # Same seed before construction: both arms get an identical fresh connector.
    torch.manual_seed(a.seed);adapter=SensorFusion(4096,model.config.text_config.hidden_size).cuda()
    connector_init=sum(float(p.double().sum()) for p in adapter.connector.parameters())
    checkpoint=torch.load(a.fusion_checkpoint,map_location='cpu',weights_only=True)
    state={k.removeprefix('fusion.'):v for k,v in checkpoint['state'].items() if k.startswith('fusion.')}
    transferred=adapter.load_state_dict(state,strict=False)
    assert not transferred.unexpected_keys and transferred.missing_keys and all(k.startswith('connector.') for k in transferred.missing_keys)
    adapter.requires_grad_(False);adapter.connector.requires_grad_(True)
    frozen_before={n:p.detach().cpu().clone() for n,p in adapter.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4)
    draws=len(train) if a.mode in ('overfit','smoke') else a.draws
    total_steps=math.ceil(draws/a.accumulation)*a.epochs;warmup_steps=math.ceil(total_steps*a.warmup_fraction)
    def lr_scale(step):
        if step<warmup_steps:return (step+1)/max(1,warmup_steps)
        progress=(step-warmup_steps)/max(1,total_steps-warmup_steps)
        return .1+.9*.5*(1+math.cos(math.pi*min(1.,progress)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale) if a.warmup_fraction else None
    weights=sampling_weights(train,a.composite_share) if a.mode=='full' else torch.ones(len(train),dtype=torch.double)
    generator=torch.Generator().manual_seed(a.seed);mask_rng=random.Random(a.seed+1)
    config=dict(mode=a.mode,seed=a.seed,lr=a.lr,epochs=a.epochs,draws_per_epoch=draws,accumulation=a.accumulation,provenance=provenance,
                counts=dict(train=len(train),validation_real=len(val),validation_composite=len(diagnostic)),
                sampler=dict(composite_share=a.composite_share,type_ratio='A:B:C=2:1:1') if a.mode=='full' else 'uniform overfit subset',
                fusion_initialization_epoch=checkpoint['epoch'],connector_init_sum=connector_init,gpu=torch.cuda.get_device_name(0),
                objective='lexical answer token NLL; no EOS or prompt loss',scoring='softmax of no/yes answer log probabilities from one shared prompt forward; fixed 0.5 threshold; not calibrated',
                frozen='Qwen, camera/RF/audio encoders, fusion projections/queries/blocks',dropout=a.modality_dropout,warmup_fraction=a.warmup_fraction,
                scheduler='warmup_cosine_to_10pct' if scheduler else 'constant',trainable_parameters=sum(p.numel() for p in adapter.parameters() if p.requires_grad),
                selection='full-sensor real-validation macro-F1 then conditional NLL; composites and test never used',patience=a.patience,
                code_sha256={p:sha256(Path(__file__).resolve().parents[2]/p) for p in ('scripts/train/train_window_alignment.py','src/drone_presence/composite_windows.py','src/drone_presence/qwen_alignment.py','src/drone_presence/model.py')})
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    if a.mode=='smoke':
        # All sensors missing must be refused before any Qwen forward.
        streams,_,_=train_set[0];none=torch.zeros((1,3),dtype=torch.bool,device='cuda')
        try:adapter([x.unsqueeze(0).cuda() for x in streams],none);raise AssertionError('All-missing input accepted')
        except ValueError:pass
    if a.mode=='full':
        baseline,pred=evaluate(model,adapter,builder,val_set,val,'camera_only')
        (a.output/'camera-only.json').write_text(json.dumps(dict(metrics=baseline,predictions=pred),indent=2))
    initial,_=evaluate(model,adapter,builder,val_set,val)
    print('Initial',json.dumps({k:v for k,v in initial.items() if k!='slices'}),flush=True)
    best=(-1.,float('-inf'));stale=0;passes=0;gradient_verified=False;started=time.monotonic()
    with (a.output/'history.jsonl').open('w') as history:
        for epoch in range(1,a.epochs+1):
            adapter.train();mask_counts={k:0 for k in ['111']+INCOMPLETE};train_losses=[];optimizer.zero_grad(set_to_none=True)
            order=torch.multinomial(weights,draws,replacement=True,generator=generator).tolist() if a.mode=='full' else torch.randperm(len(train),generator=generator).tolist()
            for step,i in enumerate(order):
                streams,label,sample_id=train_set[i];streams=[x.unsqueeze(0).cuda() for x in streams]
                # Dropout is applied after the composite is assembled; the presence target is kept.
                mask=(['111']+INCOMPLETE)[step%7] if a.mode=='smoke' else (mask_rng.choice(INCOMPLETE) if mask_rng.random()<a.modality_dropout else '111')
                mask_counts[mask]+=1
                available=torch.tensor([[int(x) for x in mask]],dtype=torch.bool,device='cuda')
                tokens=adapter(streams,available)
                answer='yes' if int(label) else 'no';inputs,positions=builder.build(sample_id,answer,camera=bool(available[0,0]))
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
                        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in adapter.connector.parameters())
                        assert all(p.grad is None for p in model.parameters())
                        assert all(p.grad is None for p in adapter.parameters() if not p.requires_grad)
                        before=adapter.connector[-1].weight.detach().clone()
                    optimizer.step()
                    if scheduler:scheduler.step()
                    if not gradient_verified:
                        assert not torch.equal(before,adapter.connector[-1].weight);gradient_verified=True;del before
                    optimizer.zero_grad(set_to_none=True)
                if a.mode=='full' and (step+1)%500==0:print(f'epoch {epoch} step {step+1}/{len(order)} loss {np.mean(train_losses[-500:]):.4f}',flush=True)
            result,pred=evaluate(model,adapter,builder,val_set,val)
            entry=dict(epoch=epoch,train_loss=float(np.mean(train_losses)),mask_counts=mask_counts,lr=optimizer.param_groups[0]['lr'],**result);history.write(json.dumps(entry)+'\n');history.flush();print(json.dumps(entry),flush=True)
            key=(result['macro_f1'],-result['conditional_nll'])
            if key>best:
                best=key;stale=0;torch.save(dict(adapter=adapter.state_dict(),epoch=epoch,metrics=result,config=config,optimizer=optimizer.state_dict()),a.output/'best.pt')
                (a.output/'best-predictions.json').write_text(json.dumps(pred))
            else:stale+=1
            passes=passes+1 if result['accuracy']==1. and result['conditional_nll']<.08 and result['answer_nll']<.15 else 0
            if a.mode=='overfit' and passes>=3:break
            if a.mode=='full' and stale>=a.patience:break
    for name,param in adapter.named_parameters():
        if name in frozen_before:assert torch.equal(param.detach().cpu(),frozen_before[name]),name
    if a.mode=='smoke':assert all(mask_counts.values()) and gradient_verified
    status='PASS' if a.mode in ('full','smoke') or (passes>=3 and gradient_verified) else 'FAIL'
    selected=torch.load(a.output/'best.pt',map_location='cpu',weights_only=True)
    report=dict(status=status,provenance=provenance,initial_metrics=initial,selected_epoch=selected['epoch'],best_sha256=sha256(a.output/'best.pt'),selected_metrics=selected['metrics'],final_metrics=result,
                epochs_completed=epoch,gradient_verified=gradient_verified,frozen_fusion_unchanged=True,elapsed_s=time.monotonic()-started,peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
    if a.mode=='full':
        adapter.load_state_dict(selected['adapter']);conditions={}
        for condition in INCOMPLETE+['111']:
            score_,pred=evaluate(model,adapter,builder,val_set,val,condition)
            score_['recording_bootstrap']=grouped_interval(val,[p['probability'] for p in pred])
            conditions[condition]=score_;(a.output/f'predictions-{condition}.json').write_text(json.dumps(pred))
        report['real_validation']=conditions
        diagnostic_set=WindowFeatures(a.features,diagnostic)
        score_,pred=evaluate(model,adapter,builder,diagnostic_set,diagnostic,'111')
        report['composite_diagnostic_full_sensor']=score_;(a.output/'predictions-composites-111.json').write_text(json.dumps(pred))
    (a.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('real_validation','initial_metrics','final_metrics','selected_metrics')}),flush=True)
    if status!='PASS':raise RuntimeError('Overfit gate failed; do not submit full alignment')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','features','fusion-checkpoint','models','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gate',type=Path);p.add_argument('--mode',choices=['overfit','full','smoke'],required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--accumulation',type=int,default=8);p.add_argument('--seed',type=int,default=4422)
    p.add_argument('--modality-dropout',type=float,default=0.);p.add_argument('--warmup-fraction',type=float,default=.1);p.add_argument('--patience',type=int,default=8)
    p.add_argument('--draws',type=int,default=2000);p.add_argument('--composite-share',type=float,default=.5)
    args=p.parse_args()
    if not 0<=args.modality_dropout<=1 or not 0<=args.warmup_fraction<1 or args.accumulation<1 or args.epochs<1 or args.draws<1: p.error('Invalid training configuration')
    main(args)
