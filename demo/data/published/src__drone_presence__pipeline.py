"""Version 2 inference on one-second windows with any non-empty subset of RGB, RF and audio.

The recipe copies the training code exactly and is checked against it by the parity gate:
- RGB: scripts/train/extract_window_features.py (Qwen3-VL visual tower, 448x448 budget, 16 pooled tokens per frame)
- RF: scripts/train/prepare_rf.py (RGB decode, bilinear 224, /255) then the trained RFEncoder
- audio: composite_windows.clap_input (one second, right zero padding to 7 s) then CLAP 2023 HTSAT
- Qwen score: scripts/train/train_window_alignment.py score() (one forward, softmax over no/yes)
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .composite_windows import AUDIO_SAMPLES, RGB_FRAMES, RF_FRAMES, clap_input
from .fusion_baseline import FusionClassifier
from .model import RFEncoder, SensorFusion, conditioned_forward
from .qwen_alignment import QUESTION, AnswerInputs, normalized_yes_probability

SENSORS = ('rgb', 'rf', 'audio')
SHAPES = {'rgb': (RGB_FRAMES * 16, 4096), 'rf': (RF_FRAMES, 128), 'audio': (1, 1024)}
CLAP_SAMPLES = 7 * 44100
IMAGE_SIZE = {'longest_edge': 448 * 448, 'shortest_edge': 32 * 32}
QWEN_LOADS = ('bf16', '4bit', 'offload')
# Our own checkpoints store NumPy scalars in their metric metadata; allow only those types.
SAFE_GLOBALS = [np.core.multiarray.scalar, np.dtype, type(np.dtype('float64')), type(np.dtype('float32')),
                type(np.dtype('int64'))]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path):
    with torch.serialization.safe_globals(SAFE_GLOBALS):
        return torch.load(path, map_location='cpu', weights_only=True)


def mask_string(available):
    return ''.join(str(int(bool(x))) for x in available)


class ImageAnswerInputs(AnswerInputs):
    """AnswerInputs that takes the window's PIL frames directly instead of TRIDENT paths."""

    def __init__(self, processor, vision_end_token_id):
        super().__init__(processor, '.', [], vision_end_token_id)
        self.images = None

    def prompt(self, clip_id, camera):
        images = list(self.images) if camera else []
        content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': QUESTION}]
        text = self.processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                                  add_generation_prompt=True)
        kwargs = dict(text=[text], return_tensors='pt')
        if images:
            kwargs.update(images=images, images_kwargs={'size': IMAGE_SIZE})
        return dict(self.processor(**kwargs))


class Pipeline:
    def __init__(self, root, qwen_load='4bit', use_qwen=True, device='cuda',
                 checkpoints='checkpoints/version-2', models='models/model-paths.json'):
        if qwen_load not in QWEN_LOADS:
            raise ValueError(f'qwen_load must be one of {QWEN_LOADS}')
        self.root, self.device, self.use_qwen, self.qwen_load = Path(root), device, use_qwen, qwen_load
        checkpoints = self.root / checkpoints
        versions = json.loads((self.root / models).read_text())
        files = {name: checkpoints / f'{name}.pt' for name in ('rf-cnn-best', 'fusion-best', 'qwen-dropout-connector-best')}
        self.provenance = {'qwen_revision': versions['qwen_revision'], 'clap_revision': versions['clap_revision'],
                           'qwen_load': qwen_load, 'checkpoints': {}}

        rf = load_checkpoint(files['rf-cnn-best'])
        self.rf_encoder = RFEncoder()
        self.rf_encoder.load_state_dict(rf['encoder'], strict=True)
        self.rf_encoder.eval().requires_grad_(False).to(device)

        from msclap.models.clap import AudioEncoder
        self.audio_encoder = AudioEncoder(audioenc_name='HTSAT', d_in=768, d_out=1024, sample_rate=44100, window_size=1024,
                                          hop_size=320, mel_bins=64, fmin=50, fmax=8000, classes_num=527)
        state = torch.load(versions['clap_path'], map_location='cpu', weights_only=True)['model']
        self.audio_encoder.load_state_dict({k.removeprefix('audio_encoder.'): v for k, v in state.items()
                                            if k.startswith('audio_encoder.')}, strict=True)
        self.audio_encoder.eval().requires_grad_(False).to(device)
        del state

        fusion = load_checkpoint(files['fusion-best'])
        self.fusion = FusionClassifier()
        self.fusion.load_state_dict(fusion['state'], strict=True)
        self.fusion.eval().requires_grad_(False).to(device)

        self.qwen = self._load_qwen(versions['qwen_path'])
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(versions['qwen_path'])
        epochs = {'rf-cnn-best': rf['epoch'], 'fusion-best': fusion['epoch']}
        if use_qwen:
            connector = load_checkpoint(files['qwen-dropout-connector-best'])
            self.adapter = SensorFusion(4096, self.qwen.config.text_config.hidden_size)
            self.adapter.load_state_dict(connector['adapter'], strict=True)
            self.adapter.eval().requires_grad_(False).to(device)
            self.builder = ImageAnswerInputs(self.processor, self.qwen.config.vision_end_token_id)
            if any(len(ids) != 1 for ids in self.builder.answers.values()):
                raise RuntimeError('Single-forward scoring needs one-token answers')
            epochs['qwen-dropout-connector-best'] = connector['epoch']
        for name, path in files.items():
            if name in epochs:
                self.provenance['checkpoints'][name] = {'sha256': sha256(path), 'epoch': int(epochs[name])}

    def _load_qwen(self, path):
        from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        common = dict(torch_dtype=torch.bfloat16, attn_implementation='sdpa')
        if self.qwen_load == 'bf16':
            model = Qwen3VLForConditionalGeneration.from_pretrained(path, **common).to(self.device)
        elif self.qwen_load == '4bit':
            # The vision tower stays bf16: the RGB fusion features come from it.
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                              bnb_4bit_compute_dtype=torch.bfloat16,
                                              llm_int8_skip_modules=['visual', 'lm_head'])
            model = Qwen3VLForConditionalGeneration.from_pretrained(path, quantization_config=quantization,
                                                                    device_map={'': 0}, **common)
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(path, device_map='auto',
                                                                    max_memory={0: '10GiB', 'cpu': '26GiB'}, **common)
        return model.eval().requires_grad_(False)

    @torch.no_grad()
    def encode_rgb(self, images):
        if len(images) != RGB_FRAMES:
            raise ValueError(f'RGB needs {RGB_FRAMES} frames')
        images = [im.convert('RGB') for im in images]
        inputs = self.processor.image_processor(images=images, return_tensors='pt', size=IMAGE_SIZE).to(self.device)
        visual, _ = self.qwen.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'])
        pooled = [F.adaptive_avg_pool1d(x.float().T.unsqueeze(0), 16).transpose(1, 2)[0] for x in visual]
        return torch.stack(pooled).reshape(SHAPES['rgb']).float().cpu()

    @torch.no_grad()
    def encode_rf(self, images):
        if len(images) != RF_FRAMES:
            raise ValueError(f'RF needs {RF_FRAMES} spectrogram images')
        arrays = [np.asarray(im.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)).transpose(2, 0, 1) for im in images]
        x = torch.from_numpy(np.stack(arrays)).float().div_(255).to(self.device)
        return self.rf_encoder(x).float().cpu()

    @torch.no_grad()
    def encode_audio(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
        if signal.shape != (AUDIO_SAMPLES,):
            raise ValueError(f'Audio needs {AUDIO_SAMPLES} mono samples at 44.1 kHz')
        crop = torch.from_numpy(clap_input(signal, 0, CLAP_SAMPLES)[None]).to(self.device)
        return self.audio_encoder(crop)[0].float().cpu()

    def encode(self, window):
        """Features for each sensor present in the window; absent sensors are None."""
        encoders = {'rgb': self.encode_rgb, 'rf': self.encode_rf, 'audio': self.encode_audio}
        return {s: None if window.get(s) is None else encoders[s](window[s]) for s in SENSORS}

    @torch.no_grad()
    def predict(self, window, features=None, mask=None):
        """window: {'id', 'rgb': 8 PIL frames, 'rf': 4 PIL spectrograms, 'audio': 44,100 floats}; missing = None.

        mask (e.g. '011') may remove sensors that are present; it cannot add a missing one.
        """
        features = features if features is not None else self.encode(window)
        present = [features[s] is not None for s in SENSORS]
        available = present if mask is None else [bool(int(c)) for c in mask]
        if len(available) != 3 or not any(available):
            raise ValueError('At least one sensor must be available')
        if any(a and not p for a, p in zip(available, present)):
            raise ValueError('Mask requests a sensor that the window does not have')
        streams = []
        for sensor in SENSORS:
            x = features[sensor] if features[sensor] is not None else torch.zeros(SHAPES[sensor])
            if tuple(x.shape) != SHAPES[sensor] or not torch.isfinite(x).all():
                raise ValueError(f'Invalid {sensor} features')
            streams.append(x.unsqueeze(0).to(self.device))
        flags = torch.tensor([available], dtype=torch.bool, device=self.device)
        result = {'id': window.get('id'), 'mask': mask_string(available),
                  'fusion_probability': float(self.fusion(streams, flags).sigmoid()[0])}
        if self.use_qwen:
            result.update(self._score_qwen(window, streams, flags))
        return result

    def _score_qwen(self, window, streams, flags):
        camera = bool(flags[0, 0])
        if camera and (window.get('rgb') is None or len(window['rgb']) != RGB_FRAMES):
            raise ValueError('Qwen with RGB needs the 8 native frames')
        self.builder.images = [im.convert('RGB') for im in window['rgb']] if camera else None
        tokens = self.adapter(streams, flags)
        inputs, positions = self.builder.build(window.get('id'), 'no', camera=camera, sensors=True, device=self.device)
        labels = inputs.pop('labels')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = conditioned_forward(self.qwen, inputs, tokens, positions).logits
        answer = (labels[0] != -100).nonzero().flatten()
        if len(answer) != 1:
            raise RuntimeError('Expected one answer token')
        log_probs = F.log_softmax(logits[0, answer[0] - 1].float(), -1)
        no, yes = log_probs[self.builder.answers['no'][0]], log_probs[self.builder.answers['yes'][0]]
        return {'qwen_probability': float(normalized_yes_probability(no, yes)),
                'no_log_probability': float(no), 'yes_log_probability': float(yes)}
