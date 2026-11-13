"""Configuration boundaries must reject before workers, weights or NCCL."""
import json
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from minivllm.config import (CompilationConfig, DeviceConfig, ModelConfig, ParallelConfig,
                             SchedulerConfig, SpeculativeConfig, VllmConfig)
from minivllm.executor.factory import get_executor_class
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir, tiny_eagle3_dir
from minivllm.worker.worker import Worker


def model():
    return ModelConfig(tiny_qwen3_dir('tiny_gqa'), hf_config=tiny_qwen3_config('tiny_gqa'), max_model_len=64)


def eagle():
    path=Path(tiny_eagle3_dir('tiny_gqa'))
    return SpeculativeConfig(method='eagle3', num_speculative_tokens=2,
        draft_model_config=ModelConfig(str(path), hf_config=json.loads((path/'config.json').read_text()), max_model_len=64))


@pytest.mark.parametrize('pp,dp',[(2,1),(4,1),(1,2),(1,4),(2,2),(4,4)])
@pytest.mark.parametrize('mode',['none','full_decode_only'])
@pytest.mark.parametrize('speculative',[False,True])
def test_pp_dp_rejected_without_worker_weights_or_process_group(monkeypatch,pp,dp,mode,speculative):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','1')
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append('started');raise AssertionError('must reject before execution')
    monkeypatch.setattr(torch.cuda,'device_count',forbidden)
    monkeypatch.setattr(torch.distributed,'init_process_group',forbidden)
    monkeypatch.setattr(Worker,'load_model',forbidden)
    from minivllm.executor.multiproc_executor import MultiprocExecutor
    monkeypatch.setattr(MultiprocExecutor,'__init__',forbidden)
    with pytest.raises(NotImplementedError,match='PP|DP'):
        VllmConfig(model(),device_config=DeviceConfig('cuda'),
          parallel_config=ParallelConfig(tensor_parallel_size=1,pipeline_parallel_size=pp,data_parallel_size=dp),
          compilation_config=CompilationConfig(cudagraph_mode=mode),
          speculative_config=eagle() if speculative else None,attention_backend='triton')
    assert calls==[]


@pytest.mark.parametrize('tp',[0,-1,3,5])
def test_invalid_or_out_of_scope_tp(monkeypatch,tp):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','1')
    with pytest.raises((ValueError,NotImplementedError)):
        VllmConfig(model(),parallel_config=ParallelConfig(tensor_parallel_size=tp))


def test_mismatched_eagle_model_rejected_before_worker(monkeypatch):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','1');monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    spec=eagle();spec.draft_model_config.hf_config['target_hidden_size']=model()._text_hf_config()['hidden_size']+1
    with pytest.raises(ValueError,match='hidden_size.*不匹配'):
        VllmConfig(model(),device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2),
                   speculative_config=spec,compilation_config=CompilationConfig(cudagraph_mode='none'),attention_backend='triton')


def test_draft_tp_and_async_are_explicitly_rejected(monkeypatch):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','1');monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    spec=replace(eagle(),draft_tensor_parallel_size=1)
    with pytest.raises(NotImplementedError,match='draft TP'):
        VllmConfig(model(),device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2),speculative_config=spec)
    with pytest.raises(NotImplementedError,match='async_scheduling'):
        VllmConfig(model(),device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2),scheduler_config=SchedulerConfig(async_scheduling=True))


@pytest.mark.parametrize('kw',[{'nnodes':2},{'prefill_context_parallel_size':2},
    {'decode_context_parallel_size':2},{'enable_expert_parallel':True},
    {'distributed_executor_backend':'ray'}])
def test_other_parallel_axes_rejected(kw):
    with pytest.raises(NotImplementedError):ParallelConfig(**kw)


def test_factory_rejects_insufficient_gpu_without_launch(monkeypatch):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','1');monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    cfg=VllmConfig(model(),device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=4),
                   compilation_config=CompilationConfig(cudagraph_mode='none'),attention_backend='triton')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    with pytest.raises(ValueError,match='GPU'):get_executor_class(cfg)


def test_single_gpu_legacy_configuration_remains_available(monkeypatch):
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','0')
    cfg=VllmConfig(model(),compilation_config=CompilationConfig(cudagraph_mode='none'))
    assert cfg.parallel_config.world_size==1


@pytest.mark.parametrize('kind',['quantized','wrong_model','torch_backend','graph_prefix'])
def test_production_executor_rejects_known_model_boundary_before_spawning(monkeypatch,kind):
    from minivllm import CacheConfig
    from minivllm.executor.multiproc_executor import MultiprocExecutor
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','1');monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append('started');raise AssertionError('must reject before spawning')
    monkeypatch.setattr('minivllm.executor.multiproc_executor.WorkerProc.make_worker_process',forbidden)
    hf=tiny_qwen3_config('tiny_gqa')
    if kind=='quantized':hf['quantization_config']={'quant_method':'test'}
    if kind=='wrong_model':hf['model_type']='other'
    cfg=VllmConfig(ModelConfig(tiny_qwen3_dir('tiny_gqa'),hf_config=hf,max_model_len=64),
        device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2),
        cache_config=CacheConfig(enable_prefix_caching=kind=='graph_prefix'),
        compilation_config=CompilationConfig(cudagraph_mode='full_decode_only' if kind=='graph_prefix' else 'none'),
        attention_backend='torch' if kind=='torch_backend' else 'triton')
    with pytest.raises(NotImplementedError):MultiprocExecutor(cfg)
    assert not calls


@pytest.mark.parametrize('tp',[2,4])
@pytest.mark.parametrize('mode',['none','full_decode_only'])
@pytest.mark.parametrize('speculative',[False,True])
def test_supported_tp_configuration_passes_factory_and_worker_checks(monkeypatch,tp,mode,speculative):
    from minivllm.executor.multiproc_executor import MultiprocExecutor
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','1');monkeypatch.setattr(torch.cuda,'device_count',lambda:4)
    cfg=VllmConfig(model(),device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=tp),
                   compilation_config=CompilationConfig(cudagraph_mode=mode),
                   speculative_config=eagle() if speculative else None,attention_backend='triton')
    assert get_executor_class(cfg) is MultiprocExecutor
    Worker(cfg)._validate_tp_model()
