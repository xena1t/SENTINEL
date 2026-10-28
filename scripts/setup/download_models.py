"""Download the pretrained models at the revisions used for both reported studies."""
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
QWEN_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
CLAP_REVISION = 'c47d441165daa21986ead0850660917636a81775'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cache', type=Path, default=ROOT / 'models/hf')
args = parser.parse_args()
cache = args.cache.resolve()
qwen = snapshot_download('Qwen/Qwen3-VL-8B-Instruct', revision=QWEN_REVISION,
                         cache_dir=cache, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
clap = hf_hub_download('microsoft/msclap', 'CLAP_weights_2023.pth',
                       revision=CLAP_REVISION, cache_dir=cache)
config = {'qwen_path': qwen, 'qwen_revision': QWEN_REVISION,
          'clap_path': clap, 'clap_revision': CLAP_REVISION}
destination = ROOT / 'models/model-paths.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(config, indent=2) + '\n')
print(destination)
