"""Extract frozen train/validation features; never use test images or audio."""
import argparse,gc,hashlib,json,sys,wave
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from drone_presence.rf_data import partition_records,RFClips
from drone_presence.model import RFEncoder

def main(a):
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    parts=partition_records(a.root);rows=parts['train']+parts['validation']
    a.output.mkdir(parents=True,exist_ok=True)
    versions=json.loads(a.models.read_text())
    split_hash=hashlib.sha256((a.root/'data/audit/trident/curated-evaluation-runs.csv').read_bytes()).hexdigest()
    # This is our own locally trained checkpoint, including NumPy scalar metric metadata.
    checkpoint=torch.load(a.rf_checkpoint,map_location='cpu',weights_only=False)
    assert checkpoint['config']['split_sha256']==split_hash
    assert set(checkpoint['config']['train_ids'])=={r['clip_id'] for r in parts['train']}
    rf_model=RFEncoder();rf_model.load_state_dict(checkpoint['encoder'],strict=True)
    rf_model.eval().requires_grad_(False).cuda()
    from msclap.models.clap import AudioEncoder
    audio_model=AudioEncoder(audioenc_name='HTSAT',d_in=768,d_out=1024,sample_rate=44100,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=8000,classes_num=527)
    state=torch.load(versions['clap_path'],map_location='cpu',weights_only=True)['model']
    audio_model.load_state_dict({k.removeprefix('audio_encoder.'):v for k,v in state.items() if k.startswith('audio_encoder.')},strict=True)
    audio_model.eval().requires_grad_(False).cuda();del state
    dataset=RFClips(a.rf_cache,rows);paths={}
    index=dict(complete=False,clips=paths,split_sha256=split_hash,models=versions,rf_checkpoint_sha256=hashlib.sha256(a.rf_checkpoint.read_bytes()).hexdigest(),rf_checkpoint_epoch=checkpoint['epoch'],visual_pooling='16 tokens per image, 8 images, native Qwen 448 pixel budget',audio_windows=[[0,7],[3,10]])
    (a.output/'index.json').write_text(json.dumps(index,indent=2))
    with torch.no_grad():
        for i,row in enumerate(rows):
            x,_,clip=dataset[i]
            with wave.open(str(a.root/row['audio_path'])) as wav:
                assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),wav.getnframes())==(44100,1,2,441000)
                signal=np.frombuffer(wav.readframes(441000),dtype='<i2').astype(np.float32)/32768.
            audio=torch.from_numpy(np.stack([signal[:308700],signal[132300:441000]])).cuda()
            features=dict(rf=rf_model(x.cuda()).cpu(),audio=audio_model(audio)[0].cpu())
            name=hashlib.sha256(clip.encode()).hexdigest()+'.pt';paths[clip]=name;torch.save(features,a.output/name)
    del rf_model,audio_model,checkpoint;gc.collect();torch.cuda.empty_cache()
    model=Qwen3VLForConditionalGeneration.from_pretrained(versions['qwen_path'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False).cuda()
    processor=AutoProcessor.from_pretrained(versions['qwen_path'])
    with torch.no_grad():
        for i,row in enumerate(rows):
            images=[]
            for path in row['rgb_paths']:
                with Image.open(a.root/path) as im:images.append(im.convert('RGB').copy())
            inputs=processor.image_processor(images=images,return_tensors='pt',size={'longest_edge':448*448,'shortest_edge':32*32}).to('cuda')
            visual,_=model.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'])
            rgb=torch.cat([F.adaptive_avg_pool1d(x.float().T.unsqueeze(0),16).transpose(1,2) for x in visual],dim=1)[0].cpu()
            assert rgb.shape==(128,4096) and torch.isfinite(rgb).all()
            path=a.output/paths[row['clip_id']];features=torch.load(path,weights_only=True);features['rgb']=rgb;torch.save(features,path)
            print(f'visual {i+1}/{len(rows)}',flush=True)
    index['complete']=True;(a.output/'index.json').write_text(json.dumps(index,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','rf-cache','rf-checkpoint','models','output'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
