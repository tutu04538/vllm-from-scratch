"""83-B：shutdown 所有权、幂等、关闭后调用与图注册生命周期（CPU 可跑）。"""

from __future__ import annotations

import gc
import weakref

import pytest

from minivllm import SamplingParams
from minivllm.compilation import CUDAGraphEntry, CUDAGraphWrapper
from minivllm.config import (CUDAGraphMode, CompilationConfig, VllmConfig)
from minivllm.engine import LLMEngine
from minivllm.executor import UniProcExecutor
from minivllm.worker import Worker
from spec83_helpers import make_fake_engine


def _config():
    from minivllm import (CacheConfig, DeviceConfig, ModelConfig,
                          SchedulerConfig)
    return VllmConfig(
        model_config=ModelConfig(model="tests/step83/fake-model",
                                 hf_config={"vocab_size": 32}, max_model_len=32),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=8),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(cudagraph_mode="none"),
    )


def test_repeated_shutdown_and_calls_after_close_are_explicit():
    engine, runner, worker = make_fake_engine(tokens={"a": [7, 8]})
    engine.add_request("a", [1, 2, 3, 4],
                       SamplingParams(max_tokens=2, eos_token_id=999))
    engine.step()
    engine.shutdown()
    engine.shutdown()                      # 幂等
    worker.shutdown()                      # Worker 也幂等
    core = engine.engine_core.engine_core
    assert core._shutdown
    assert core.model_executor is None
    with pytest.raises(RuntimeError, match="已经 shutdown"):
        engine.add_request("b", [1], SamplingParams(max_tokens=1))
    with pytest.raises(RuntimeError, match="已经 shutdown"):
        engine.step()
    with pytest.raises(RuntimeError, match="已经 shutdown"):
        engine.has_unfinished_requests()
    with pytest.raises(RuntimeError, match="已经 shutdown"):
        core.step()
    with pytest.raises(RuntimeError, match="已经 shutdown"):
        core.add_request(object())
    # 关闭后 Worker 不再持有 Runner，调用明确报错。
    assert worker.model_runner is None
    with pytest.raises(RuntimeError, match="已经 shutdown"):
        worker.sample_tokens(None)


def test_shutdown_one_engine_does_not_touch_another_live_engine():
    first, first_runner, _ = make_fake_engine(tokens={"a": [7]})
    second, second_runner, _ = make_fake_engine(tokens={"b": [9]})
    first.shutdown()
    second.add_request("b", [1, 2, 3, 4],
                       SamplingParams(max_tokens=1, eos_token_id=999))
    out = second.step()
    assert out[0].token_ids == [9]
    assert second_runner.pending is None
    second.shutdown()


def test_runner_reference_is_released_after_shutdown():
    engine, runner, worker = make_fake_engine(tokens={"a": [7]})
    ref = weakref.ref(runner)
    engine.shutdown()
    del runner
    gc.collect()
    assert ref() is None, "runner 仍被引擎/Worker 的引用链持有"
    assert worker.model_runner is None


class _FailingRunner:
    def initialize_kv_cache(self, kv_cache_config):
        raise RuntimeError("injected kv init failure")


def test_partial_init_failure_closes_executor_and_releases_runner():
    cfg = _config()
    runner = _FailingRunner()
    ref = weakref.ref(runner)
    executor = UniProcExecutor(cfg, Worker(cfg, model_runner=runner))
    with pytest.raises(RuntimeError, match="injected kv init failure"):
        LLMEngine(cfg, executor)
    assert executor._shutdown
    assert executor.driver_worker is None
    del runner
    gc.collect()
    assert ref() is None


def test_graph_registry_does_not_keep_instances_alive():
    cfg = VllmConfig(
        model_config=__import__("minivllm").ModelConfig(
            model="tests/step83/fake-model", hf_config={"vocab_size": 32}),
        cache_config=__import__("minivllm").CacheConfig(),
        scheduler_config=__import__("minivllm").SchedulerConfig(),
        compilation_config=CompilationConfig(cudagraph_mode="full_decode_only"),
    )
    before = CUDAGraphWrapper.live_instance_count()
    first = CUDAGraphWrapper(lambda: None, cfg, CUDAGraphMode.FULL)
    ref = weakref.ref(first)
    assert CUDAGraphWrapper.live_instance_count() == before + 1
    first.close()
    assert CUDAGraphWrapper.live_instance_count() == before
    del first
    gc.collect()
    assert ref() is None, "关闭后的图包装器仍被全局注册表强引用"


def test_closing_one_graph_owner_does_not_clear_another():
    cfg = VllmConfig(
        model_config=__import__("minivllm").ModelConfig(
            model="tests/step83/fake-model", hf_config={"vocab_size": 32}),
        cache_config=__import__("minivllm").CacheConfig(),
        scheduler_config=__import__("minivllm").SchedulerConfig(),
        compilation_config=CompilationConfig(cudagraph_mode="full_decode_only"),
    )
    left = CUDAGraphWrapper(lambda: None, cfg, CUDAGraphMode.FULL)
    right = CUDAGraphWrapper(lambda: None, cfg, CUDAGraphMode.FULL)
    from minivllm.forward_context import BatchDescriptor
    desc = BatchDescriptor(num_tokens=1)
    left.concrete_cudagraph_entries[desc] = CUDAGraphEntry(batch_descriptor=desc)
    right.concrete_cudagraph_entries[desc] = CUDAGraphEntry(batch_descriptor=desc)
    left.close()
    assert left.concrete_cudagraph_entries == {}
    assert right.concrete_cudagraph_entries, "关闭一个引擎时清掉了另一个引擎的图"
    right.close()
