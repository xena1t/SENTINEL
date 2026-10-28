"""Causal answer scoring, gradient flow, real tokenizer boundaries and camera removal."""
import json,sys
from pathlib import Path
import torch
from transformers import AutoProcessor,AutoConfig
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from drone_presence.qwen_alignment import answer_log_probability,normalized_yes_probability,AnswerInputs
from drone_presence.rf_data import partition_records

torch.set_num_threads(4)
logits=torch.tensor([[[1.,2.,3.],[3.,1.,2.],[2.,3.,1.],[1.,1.,1.]]],requires_grad=True)
labels=torch.tensor([[-100,-100,0,1]])
score=answer_log_probability(logits,labels)
expected=logits[0,1].log_softmax(0)[0]+logits[0,2].log_softmax(0)[1]
torch.testing.assert_close(score[0],expected);(-score.sum()).backward()
assert logits.grad[0,0].abs().sum()==0 and logits.grad[0,3].abs().sum()==0 and logits.grad[0,1:3].abs().sum()>0
p=normalized_yes_probability(torch.tensor([-2.]),torch.tensor([-1.]))
torch.testing.assert_close(p,torch.tensor([0.7310586]))
try:answer_log_probability(logits,torch.full_like(labels,-100))
except ValueError:pass
else:raise AssertionError('Empty answers accepted')
root=Path(sys.argv[1]);versions=json.loads(Path(sys.argv[2]).read_text());rows=partition_records(root)['train'][:1]
processor=AutoProcessor.from_pretrained(versions['qwen_path']);config=AutoConfig.from_pretrained(versions['qwen_path'])
builder=AnswerInputs(processor,root,rows,config.vision_end_token_id)
for camera in (False,True):
 for enabled in (False,True):
  for answer in ('yes','no'):
   inputs,positions=builder.build(rows[0]['clip_id'],answer,camera,enabled,'cpu')
   assert (inputs['labels']!=-100).sum()==len(builder.answers[answer])
   assert bool((inputs['input_ids']==config.image_token_id).any())==camera
   assert ('pixel_values' in inputs)==camera
   if enabled:assert positions.shape==(1,16) and (inputs['labels'][0,positions[0]]==-100).all()
   else:assert positions is None
native,_=builder.build(rows[0]['clip_id'],'yes',True,False,'cpu')
active,pos=builder.build(rows[0]['clip_id'],'yes',True,True,'cpu')
start=int(pos[0,0]);torch.testing.assert_close(torch.cat([active['input_ids'][:,:start],active['input_ids'][:,start+16:]],dim=1),native['input_ids'])
print('PASS: causal multi-token sum, answer-only gradients, normalized yes/no scores, tokenizer boundary, all prompt modes, identity slot removal')
