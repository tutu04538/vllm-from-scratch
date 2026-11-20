"""Validate the AWQ TP entry boundary before starting worker processes."""
import pytest
import torch

from minivllm.config import (CompilationConfig, DeviceConfig, ModelConfig,
                             ParallelConfig, VllmConfig)
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir


@pytest.mark.parametrize('tp', [1, 2])
def test_awq_v2_eager_tp_configuration(monkeypatch, tp):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER', '1')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    hf = tiny_qwen3_config('tiny_gqa')
    hf['quantization_config'] = {'quant_method': 'awq', 'bits': 4,
                                  'group_size': 128, 'zero_point': True}
    cfg = VllmConfig(ModelConfig(tiny_qwen3_dir('tiny_gqa'), hf_config=hf,
                                  dtype='bfloat16', max_model_len=64),
                     device_config=DeviceConfig('cuda'),
                     parallel_config=ParallelConfig(tensor_parallel_size=tp),
                     compilation_config=CompilationConfig(cudagraph_mode='none'),
                     attention_backend='triton')
    assert cfg.parallel_config.tensor_parallel_size == tp


def test_awq_tp4_requires_real_multi_gpu_validation(monkeypatch):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER', '1')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    hf = tiny_qwen3_config('tiny_gqa')
    hf['quantization_config'] = {'quant_method': 'awq', 'bits': 4,
                                  'group_size': 128, 'zero_point': True}
    with pytest.raises(NotImplementedError, match='TP4.*验收尚未完成'):
        VllmConfig(ModelConfig(tiny_qwen3_dir('tiny_gqa'), hf_config=hf,
                               dtype='bfloat16', max_model_len=64),
                   device_config=DeviceConfig('cuda'),
                   parallel_config=ParallelConfig(tensor_parallel_size=4),
                   compilation_config=CompilationConfig(cudagraph_mode='none'),
                   attention_backend='triton')


def test_unknown_quantization_is_rejected_before_spawning(monkeypatch):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER', '1')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    hf = tiny_qwen3_config('tiny_gqa')
    hf['quantization_config'] = {'quant_method': 'unknown'}
    with pytest.raises(NotImplementedError, match='不支持量化方法'):
        VllmConfig(ModelConfig(tiny_qwen3_dir('tiny_gqa'), hf_config=hf,
                               dtype='bfloat16', max_model_len=64),
                   device_config=DeviceConfig('cuda'),
                   parallel_config=ParallelConfig(tensor_parallel_size=2),
                   compilation_config=CompilationConfig(cudagraph_mode='none'),
                   attention_backend='triton')


def test_awq_tp_graph_still_rejected_before_worker(monkeypatch):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER', '1')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    hf = tiny_qwen3_config('tiny_gqa')
    hf['quantization_config'] = {'quant_method': 'awq', 'bits': 4,
                                  'group_size': 128, 'zero_point': True}
    with pytest.raises(NotImplementedError, match='量化模型当前仅支持'):
        VllmConfig(ModelConfig(tiny_qwen3_dir('tiny_gqa'), hf_config=hf,
                               dtype='bfloat16', max_model_len=64),
                   device_config=DeviceConfig('cuda'),
                   parallel_config=ParallelConfig(tensor_parallel_size=2),
                   compilation_config=CompilationConfig(cudagraph_mode='full_decode_only'),
                   attention_backend='triton')
