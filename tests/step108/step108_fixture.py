"""Create a tiny head-size-32 target/draft pair compatible with frozen Triton."""
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from minivllm.testing.tiny_models import tiny_eagle3_dir, tiny_qwen3_config, _weights


def create_pair(root):
    root = Path(root)
    target, draft = root/'target', root/'draft'
    target.mkdir(parents=True, exist_ok=True)
    draft.mkdir(parents=True, exist_ok=True)
    target_cfg = tiny_qwen3_config('tiny_gqa')
    target_cfg['head_dim'] = 32
    (target/'config.json').write_text(json.dumps(target_cfg)+'\n')
    save_file(_weights(target_cfg, seed=0), str(target/'model.safetensors'))
    source = Path(tiny_eagle3_dir('tiny_gqa'))
    draft_cfg = json.loads((source/'config.json').read_text())
    draft_cfg['head_dim'] = 32
    weights = load_file(str(source/'model.safetensors'))
    g = torch.Generator().manual_seed(108)
    def random(shape):
        return torch.randn(shape, generator=g) * .02
    hidden = draft_cfg['hidden_size']
    heads = draft_cfg['num_attention_heads']
    kv = draft_cfg['num_key_value_heads']
    for name, width in (('q', heads * 32), ('k', kv * 32), ('v', kv * 32)):
        weights[f'midlayer.self_attn.{name}_proj.weight'] = random((width, 2*hidden))
    weights['midlayer.self_attn.o_proj.weight'] = random((hidden, heads*32))
    weights['midlayer.self_attn.q_norm.weight'] = torch.ones(32)
    weights['midlayer.self_attn.k_norm.weight'] = torch.ones(32)
    (draft/'config.json').write_text(json.dumps(draft_cfg)+'\n')
    save_file(weights, str(draft/'model.safetensors'))
    return target, draft
