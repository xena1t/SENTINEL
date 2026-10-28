"""Native Qwen image path plus sensor tokens and lexical yes/no sequence scoring."""
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from .model import conditioned_forward

QUESTION = 'Is a drone present during this observation? Answer yes or no.'


def answer_log_probability(logits, labels):
    """Sum causal log probabilities over answer tokens only (no prompt or EOS)."""
    shifted_labels = labels[:, 1:]
    selected = shifted_labels != -100
    if not selected.any(dim=1).all():
        raise ValueError('Each example needs answer tokens')
    targets = shifted_labels.masked_fill(~selected, 0)
    # Gather only answer positions before computing a float32 vocabulary softmax.
    scores = torch.zeros(labels.shape[0], device=logits.device, dtype=torch.float32)
    for row in range(labels.shape[0]):
        token_logits = logits[row, :-1][selected[row]].float()
        scores[row] = F.log_softmax(token_logits, dim=-1).gather(1, targets[row][selected[row], None]).sum()
    return scores


def normalized_yes_probability(no_score, yes_score):
    """Conditional probability within the two fixed candidate strings, not calibration."""
    return torch.softmax(torch.stack([no_score, yes_score], dim=-1), dim=-1)[..., 1]


class AnswerInputs:
    def __init__(self, processor, root, records, vision_end_token_id):
        self.processor = processor
        self.root = Path(root)
        self.records = {r['clip_id']: r for r in records}
        self.vision_end_token_id = vision_end_token_id
        self.slot_ids = processor.tokenizer.encode('~', add_special_tokens=False)
        if len(self.slot_ids) != 1:
            raise ValueError('Expected one ordinary placeholder token')
        self.answers = {word: processor.tokenizer.encode(word, add_special_tokens=False) for word in ('no', 'yes')}
        if any(not ids for ids in self.answers.values()):
            raise ValueError('Empty answer token sequence')

    @lru_cache(maxsize=8)
    def prompt(self, clip_id, camera):
        images = []
        if camera:
            for path in self.records[clip_id]['rgb_paths']:
                with Image.open(self.root / path) as im:
                    images.append(im.convert('RGB').copy())
        content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': QUESTION}]
        text = self.processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        kwargs = dict(text=[text], return_tensors='pt')
        if images:
            kwargs.update(images=images, images_kwargs={'size': {'longest_edge': 448*448, 'shortest_edge': 32*32}})
        values = dict(self.processor(**kwargs))
        # Ensure independently encoded candidate tokens match their prompt boundary.
        for word, ids in self.answers.items():
            prefix = self.processor.tokenizer.encode(text, add_special_tokens=False)
            combined = self.processor.tokenizer.encode(text + word, add_special_tokens=False)
            if combined != prefix + ids:
                raise ValueError('Answer tokenization changes at prompt boundary')
        return values

    def build(self, clip_id, answer, camera=True, sensors=True, device='cuda'):
        prompt = self.prompt(clip_id, camera)
        result = dict(prompt)
        ids = prompt['input_ids']
        labels = torch.full_like(ids, -100)
        positions = None
        if sensors:
            ends = (ids[0] == self.vision_end_token_id).nonzero().flatten()
            # Without camera put slots after the user role header, before the question.
            if camera:
                if len(ends) != 8:
                    raise ValueError('Expected eight native image blocks')
                start = int(ends[-1]) + 1
            else:
                header = self.processor.tokenizer.encode('<|im_start|>user\n', add_special_tokens=False)
                sequence = ids[0].tolist()
                matches = [i+len(header) for i in range(len(sequence)-len(header)+1) if sequence[i:i+len(header)] == header]
                if len(matches) != 1:
                    raise ValueError('Cannot locate user header for sensor slots')
                start = matches[0]
            slots = torch.full((1, 16), self.slot_ids[0], dtype=ids.dtype)
            ids = torch.cat([ids[:, :start], slots, ids[:, start:]], dim=1)
            labels = torch.full_like(ids, -100)
            positions = torch.arange(start, start+16).unsqueeze(0).to(device)
        answer_ids = torch.tensor([self.answers[answer]], dtype=ids.dtype)
        result['input_ids'] = torch.cat([ids, answer_ids], dim=1)
        result['labels'] = torch.cat([labels, answer_ids], dim=1)
        result['attention_mask'] = torch.ones_like(result['input_ids'])
        if not camera and ('pixel_values' in result or 'image_grid_thw' in result):
            raise ValueError('Missing camera must remove native visual input')
        return {key: value.to(device) for key, value in result.items()}, positions


def forward_score(model, inputs, positions=None, sensor_tokens=None):
    labels = inputs['labels']
    model_inputs = {key: value for key, value in inputs.items() if key != 'labels'}
    output = conditioned_forward(model, model_inputs, sensor_tokens, positions)
    return answer_log_probability(output.logits, labels)


@torch.no_grad()
def score_yes_no(model, adapter, builder, clip_id, streams, available, device='cuda', enabled=True):
    camera = bool(available[0, 0])
    tokens = adapter(streams, available) if enabled else None
    scores = []
    for answer in ('no', 'yes'):
        inputs, positions = builder.build(clip_id, answer, camera=camera, sensors=enabled, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            scores.append(forward_score(model, inputs, positions, tokens))
    return float(normalized_yes_probability(scores[0], scores[1])[0]), [float(x[0]) for x in scores]
