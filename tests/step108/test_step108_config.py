"""Unsupported TP speculation combinations fail before worker startup."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minivllm.config import (CompilationConfig, DeviceConfig, ModelConfig, ParallelConfig,
                             SpeculativeConfig, VllmConfig)
from minivllm.engine.core import EngineCore
from minivllm.testing.tiny_models import tiny_eagle3_dir, tiny_qwen3_config, tiny_qwen3_dir


def config(monkeypatch, *, mode='none', **spec_args):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER', '1')
    target = ModelConfig(tiny_qwen3_dir('tiny_gqa'), hf_config=tiny_qwen3_config('tiny_gqa'), max_model_len=64)
    draft = Path(tiny_eagle3_dir())
    draft_cfg = ModelConfig(str(draft), max_model_len=64,
                            hf_config=json.loads((draft/'config.json').read_text()))
    spec = SpeculativeConfig(method='eagle3', num_speculative_tokens=2, draft_model_config=draft_cfg,
                             **spec_args)
    return VllmConfig(model_config=target, device_config=DeviceConfig('cuda'),
                      speculative_config=spec, parallel_config=ParallelConfig(tensor_parallel_size=2),
                      compilation_config=CompilationConfig(cudagraph_mode=mode), attention_backend='triton')


def test_eager_and_full_decode_config(monkeypatch):
    assert config(monkeypatch).effective_draft_tensor_parallel_size == 2
    assert config(monkeypatch, mode='full_decode_only').effective_draft_tensor_parallel_size == 2


@pytest.mark.parametrize('args', [
    {'parallel_drafting': True},
    {'num_speculative_tokens_per_batch_size': [(1, 4, 1)]},
    {'rejection_sample_method': 'block'},
])
def test_unsupported_tp_speculation_is_not_silently_changed(monkeypatch, args):
    with pytest.raises((ValueError, NotImplementedError)):
        config(monkeypatch, **args)


def test_tp_speculation_rejects_structured_output_request(monkeypatch):
    cfg = config(monkeypatch)
    core = object.__new__(EngineCore)
    core.vllm_config = cfg
    core.speculative_config = cfg.speculative_config
    with pytest.raises(NotImplementedError, match='结构化输出'):
        core._validate_sampling_params(SimpleNamespace(num_logprobs=None, structured_outputs=object()))
