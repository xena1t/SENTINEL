"""Evaluate an alignment checkpoint on validation only, using fixed yes/no candidates."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from train_alignment import evaluate,digest
from drone_presence.rf_data import partition_records
from drone_presence.fusion_baseline import FeatureClips
from drone_presence.model import SensorFusion
from drone_presence.qwen_alignment import AnswerInputs
p=argparse.ArgumentParser()
for name in ['root','features','models','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--condition',default='111',choices=['111','100','010','001','110','101','011','camera_only','shuffled_rf_audio']);a=p.parse_args()
checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True);provenance=checkpoint['config']['provenance']
assert provenance['feature_index_sha256']==digest(a.features/'index.json') and provenance['models_sha256']==digest(a.models)
assert provenance['split_sha256']==digest(a.root/'data/audit/trident/curated-evaluation-runs.csv')
versions=json.loads(a.models.read_text());rows=partition_records(a.root)['validation']
model=Qwen3VLForConditionalGeneration.from_pretrained(versions['qwen_path'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False).cuda()
processor=AutoProcessor.from_pretrained(versions['qwen_path']);adapter=SensorFusion(4096,model.config.text_config.hidden_size).cuda();adapter.load_state_dict(checkpoint['adapter'],strict=True)
builder=AnswerInputs(processor,a.root,rows,model.config.vision_end_token_id)
result,pred=evaluate(model,adapter,builder,FeatureClips(a.features,rows),a.condition)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(condition=a.condition,metrics=result,predictions=pred),indent=2));print(json.dumps(result))
