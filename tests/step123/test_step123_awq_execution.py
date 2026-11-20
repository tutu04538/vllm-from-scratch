"""AWQ execution must not bypass the explicitly supported runtime boundary."""
from types import SimpleNamespace

import pytest
import torch

from minivllm.config import CUDAGraphMode, VllmConfig
from minivllm.layers.quantization.awq import AutoAWQMarlinLinearMethod
from minivllm.model_loader.base_loader import BaseModelLoader
from minivllm import ModelConfig
from minivllm.testing.tiny_models import tiny_qwen3_config

AWQ = {'quant_method': 'awq', 'bits': 4, 'group_size': 128,
       'zero_point': True, 'version': 'gemm'}


def runtime(**changes):
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config={'quantization_config': AWQ},
            _text_hf_config=lambda: {}, quantization=None, dtype='bfloat16'),
        use_v2_model_runner=True, device_config=SimpleNamespace(device='cuda'),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE))
    for key, value in changes.items():
        parent, _, attr = key.partition('__')
        if attr:
            setattr(getattr(cfg, parent), attr, value)
        else:
            setattr(cfg, parent, value)
    return cfg


def test_supported_boundary_and_unquantized_bypass():
    VllmConfig._validate_quantized_execution(runtime())
    cfg = runtime(use_v2_model_runner=False)
    cfg.model_config.hf_config = {}
    VllmConfig._validate_quantized_execution(cfg)


@pytest.mark.parametrize('changes', [
    {'use_v2_model_runner': False}, {'device_config__device': 'cpu'},
    {'parallel_config__tensor_parallel_size': 2}, {'speculative_config': object()},
    {'compilation_config__cudagraph_mode': CUDAGraphMode.FULL},
    {'model_config__dtype': 'float16'},
])
def test_unsupported_quantized_combinations_rejected(changes):
    with pytest.raises(NotImplementedError, match='量化模型当前仅支持'):
        VllmConfig._validate_quantized_execution(runtime(**changes))


def test_quantized_cpu_execution_rejected_but_representation_remains_available():
    hf = {**tiny_qwen3_config('tiny_gqa'), 'quantization_config': AWQ}
    cfg = ModelConfig(model='unused', hf_config=hf, dtype='bfloat16')
    with pytest.raises(NotImplementedError, match='量化执行尚未开放'):
        BaseModelLoader()._initialize_model(cfg, 'cpu')
    # Use a group-compatible Qwen configuration to exercise representation construction.
    hf.update(hidden_size=256, intermediate_size=512, num_attention_heads=4,
              num_key_value_heads=2, head_dim=64)
    model = BaseModelLoader()._initialize_model(cfg, 'cpu', representation_only=True)
    methods = [m.quant_method for m in model.modules()
               if isinstance(getattr(m, 'quant_method', None), AutoAWQMarlinLinearMethod)]
    assert methods and all(not m.supports_execution for m in methods)
