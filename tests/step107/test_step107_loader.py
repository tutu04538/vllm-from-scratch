"""Packed EAGLE3 parameters must not silently accept incomplete checkpoints."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from minivllm.models.qwen3_eagle3 import Eagle3ForCausalLM
from minivllm.testing.tiny_models import tiny_eagle3_dir


def test_packed_checkpoint_missing_and_duplicate_shards():
    path = Path(tiny_eagle3_dir())
    config = json.loads((path / 'config.json').read_text())
    weights = list(sorted(load_file(str(path / 'model.safetensors')).items()))
    key = 'midlayer.self_attn.k_proj.weight'
    assert any(name == key for name, _ in weights)
    with pytest.raises(ValueError, match='缺少'):
        Eagle3ForCausalLM(config).load_weights((n, t) for n, t in weights if n != key)
    repeated = weights + [next((n, t) for n, t in weights if n == key)]
    with pytest.raises(ValueError, match='重复'):
        Eagle3ForCausalLM(config).load_weights(iter(repeated))
    repeated_map = weights + [('d2t', torch.zeros(config['vocab_size'], dtype=torch.long))] * 2
    with pytest.raises(ValueError, match='重复'):
        Eagle3ForCausalLM(config).load_weights(iter(repeated_map))
    with pytest.raises((ValueError, KeyError)):
        Eagle3ForCausalLM(config).load_weights(iter(weights + [('unexpected.weight', torch.zeros(1))]))
    with pytest.raises(ValueError, match='缺少|没有被加载'):
        Eagle3ForCausalLM(config).load_weights(iter([next(x for x in weights if x[0] == 'lm_head.weight')]))


@pytest.mark.parametrize('draft_tp', [1, 2])
def test_draft_tp_must_match_target(draft_tp):
    from minivllm.config import ModelConfig, ParallelConfig, SpeculativeConfig, VllmConfig
    model = ModelConfig('fixture', hf_config={})
    spec = SpeculativeConfig(method='eagle3', num_speculative_tokens=2,
                             draft_model_config=model, draft_tensor_parallel_size=draft_tp)
    with pytest.raises(NotImplementedError, match='draft TP 必须等于 target TP'):
        VllmConfig(model, speculative_config=spec,
                   parallel_config=ParallelConfig(tensor_parallel_size=4))
