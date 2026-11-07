"""模型层面的缺失和重复 shard 不能被 packed 参数覆盖检查漏掉。"""
import pytest
from minivllm.model_loader.weight_utils import iter_weights
from minivllm.models.qwen3 import Qwen3Model
from minivllm.testing.tiny_models import tiny_qwen3_config,tiny_qwen3_dir


def test_qwen_loader_detects_missing_k_and_duplicate_gate():
    config=tiny_qwen3_config('tiny_gqa')
    path=tiny_qwen3_dir('tiny_gqa')
    weights=[(n.removeprefix('model.'),t) for n,t in iter_weights(path) if n.startswith('model.')]
    missing=[(n,t) for n,t in weights if n!='layers.0.self_attn.k_proj.weight']
    with pytest.raises(ValueError,match='缺少'):
        Qwen3Model(config).load_weights(iter(missing))
    duplicate=[]
    for name,tensor in weights:
        duplicate.append((name,tensor))
        if name=='layers.0.mlp.gate_proj.weight':
            duplicate.append((name,tensor))
    with pytest.raises(ValueError,match='重复'):
        Qwen3Model(config).load_weights(iter(duplicate))
