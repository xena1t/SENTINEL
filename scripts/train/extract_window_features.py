"""Frozen one-second features for rgb-visibility-v1; train/validation clips only, never test media."""
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
from drone_presence.composite_windows import load_samples,sha256,clap_input

CLAP_SAMPLES=7*44100


def clip_plan(samples):
    """Per clip: RGB frame indices and audio crop starts referenced by any sample."""
    plan={}
    for s in samples:
        plan.setdefault(s['rgb_clip_id'],dict(rgb=set(),audio=set()))['rgb'].update(s['rgb_indices'])
        plan.setdefault(s['rf_audio_clip_id'],dict(rgb=set(),audio=set()))['audio'].add(s['audio_start_sample'])
    return {c:dict(rgb=sorted(v['rgb']),audio=sorted(v['audio'])) for c,v in plan.items()}


def main(a):
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    samples=load_samples(a.root);plan=clip_plan(samples)
    parts=partition_records(a.root);development={r['clip_id']:r for r in parts['train']+parts['validation']}
    assert set(plan)<=set(development) and not set(plan)&{r['clip_id'] for r in parts['test']}
    versions=json.loads(a.models.read_text())
    split_hash=sha256(a.root/'data/audit/trident/curated-evaluation-runs.csv')
    # Our own locally trained checkpoint; it carries NumPy scalar metric metadata.
    checkpoint=torch.load(a.rf_checkpoint,map_location='cpu',weights_only=False)
    assert checkpoint['config']['split_sha256']==split_hash and checkpoint['config']['mode']=='full'
    assert set(checkpoint['config']['train_ids'])=={r['clip_id'] for r in parts['train']}
    a.output.mkdir(parents=True,exist_ok=True)
    index=dict(complete=False,label_version='rgb-visibility-v1',clips={},split_sha256=split_hash,models=versions,
               window_manifest_sha256=sha256(a.root/'data/windows/manifest.jsonl'),sources_sha256=sha256(a.root/'data/windows/sources.json'),
               implementation_sha256=sha256(__file__),rf_checkpoint_sha256=sha256(a.rf_checkpoint),rf_checkpoint_epoch=checkpoint['epoch'],
               recipe=dict(rgb='Qwen3-VL-8B native visual features, 448x448 pixel budget, adaptive-average-pooled to 16 tokens per frame (as extract_fusion_features.py)',
                           rf='trained 128-wide RFEncoder per RF frame, RGB decode bilinear224 /255 (as prepare_rf.py)',
                           audio='exact one-second 44,100-sample PCM crop, then right zero padding to 7 s, CLAP 2023 HTSAT audio encoder; no neighbouring audio'))
    (a.output/'index.json').write_text(json.dumps(index,indent=2))
    folders={c:a.output/hashlib.sha256(c.encode()).hexdigest()[:24] for c in plan}
    rf_model=RFEncoder();rf_model.load_state_dict(checkpoint['encoder'],strict=True);rf_model.eval().requires_grad_(False).cuda()
    from msclap.models.clap import AudioEncoder
    audio_model=AudioEncoder(audioenc_name='HTSAT',d_in=768,d_out=1024,sample_rate=44100,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=8000,classes_num=527)
    state=torch.load(versions['clap_path'],map_location='cpu',weights_only=True)['model']
    audio_model.load_state_dict({k.removeprefix('audio_encoder.'):v for k,v in state.items() if k.startswith('audio_encoder.')},strict=True)
    audio_model.eval().requires_grad_(False).cuda();del state
    rf_clips=RFClips(a.rf_cache,[development[c] for c in plan])
    with torch.no_grad():
        for i,clip in enumerate(sorted(plan)):
            folder=folders[clip];folder.mkdir(exist_ok=True)
            x,_,clip_id=rf_clips[rf_clips.records.index(development[clip])];assert clip_id==clip
            np.save(folder/'rf.npy',rf_model(x.cuda()).float().cpu().numpy())
            source=development[clip]
            with wave.open(str(a.root/source['audio_path'])) as wav:
                assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),wav.getnframes())==(44100,1,2,441000)
                signal=np.frombuffer(wav.readframes(441000),dtype='<i2').astype(np.float32)/32768.
            starts=plan[clip]['audio']
            crops=np.stack([clap_input(signal,start,CLAP_SAMPLES) for start in starts]) if starts else np.zeros((0,CLAP_SAMPLES),dtype=np.float32)
            audio=torch.cat([audio_model(torch.from_numpy(chunk).cuda())[0].float().cpu() for chunk in np.array_split(crops,max(1,len(crops)//16))]) if starts else torch.zeros(0,1024)
            np.save(folder/'audio.npy',audio.numpy()[:,None,:])
            (folder/'meta.json').write_text(json.dumps(dict(clip_id=clip,rgb_indices=plan[clip]['rgb'],audio_starts=starts)))
            if (i+1)%20==0:print(f'rf/audio {i+1}/{len(plan)}',flush=True)
    del rf_model,audio_model,checkpoint;gc.collect();torch.cuda.empty_cache()
    model=Qwen3VLForConditionalGeneration.from_pretrained(versions['qwen_path'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False).cuda()
    processor=AutoProcessor.from_pretrained(versions['qwen_path'])
    sources={s['clip_id']:s for s in json.loads((a.root/'data/windows/sources.json').read_text())}
    with torch.no_grad():
        for i,clip in enumerate(sorted(plan)):
            indices=plan[clip]['rgb'];pooled=[]
            for start in range(0,len(indices),16):
                images=[]
                for j in indices[start:start+16]:
                    with Image.open(a.root/sources[clip]['rgb_files'][j-1]['path']) as im:images.append(im.convert('RGB').copy())
                inputs=processor.image_processor(images=images,return_tensors='pt',size={'longest_edge':448*448,'shortest_edge':32*32}).to('cuda')
                visual,_=model.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'])
                pooled.extend(F.adaptive_avg_pool1d(x.float().T.unsqueeze(0),16).transpose(1,2)[0].cpu() for x in visual)
            rgb=torch.stack(pooled) if pooled else torch.zeros(0,16,4096)
            assert rgb.shape==(len(indices),16,4096) and torch.isfinite(rgb).all()
            np.save(folders[clip]/'rgb.npy',rgb.numpy())
            index['clips'][clip]=folders[clip].name
            print(f'visual {i+1}/{len(plan)}',flush=True)
    index['complete']=True;(a.output/'index.json').write_text(json.dumps(index,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','rf-cache','rf-checkpoint','models','output'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
