"""83-B：真实 tiny Runner 的关闭、图注销与连续建销毁（需要 CUDA）。"""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import pytest
import torch

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.compilation import CUDAGraphWrapper
from minivllm.config import CompilationConfig
from minivllm.testing.tiny_models import (tiny_mtp_dir, tiny_qwen3_config,
                                          tiny_qwen3_dir)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Runner 生命周期集成用例需要 CUDA/Triton")


def _run(engine, *, max_tokens=2):
    engine.add_request("a", [1, 2, 3, 4],
                       SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                      eos_token_id=999))
    final = {}
    for _ in range(100):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            final[out.request_id] = list(out.token_ids)
    return final


def _v1_graph_engine():
    directory = tiny_qwen3_dir("tiny_gqa")
    hf = tiny_qwen3_config("tiny_gqa")
    target = ModelConfig(model=directory, dtype="float32", max_model_len=64,
                         hf_config=hf)
    draft = ModelConfig(model=directory, dtype="float32", max_model_len=64,
                        hf_config=hf)
    cfg = VllmConfig(
        model_config=target,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=32),
        scheduler_config=SchedulerConfig(max_num_seqs=2,
                                         max_num_batched_tokens=32),
        device_config=DeviceConfig(device="cuda"),
        speculative_config=SpeculativeConfig(
            method="draft_model", num_speculative_tokens=2,
            draft_model_config=draft),
        compilation_config=CompilationConfig(
            cudagraph_mode="full_decode_only",
            cudagraph_capture_sizes=[2, 4],
            max_cudagraph_capture_size=4),
    )
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    runner = (engine.engine_core.engine_core.model_executor
              .driver_worker.model_runner)
    return engine, runner


def test_v1_graph_wrapper_is_unregistered_by_shutdown():
    before = CUDAGraphWrapper.live_instance_count()
    engine, runner = _v1_graph_engine()
    wrapper = runner.model.cudagraph_wrapper
    assert CUDAGraphWrapper.live_instance_count() == before + 1
    ref = weakref.ref(wrapper)
    _run(engine)
    engine.shutdown()
    assert CUDAGraphWrapper.live_instance_count() == before
    assert wrapper.concrete_cudagraph_entries == {}
    del wrapper, runner, engine
    gc.collect()
    assert ref() is None


def _v2_config():
    directory = tiny_mtp_dir("tiny_gqa")
    hf = json.loads((Path(directory) / "config.json").read_text())
    target = ModelConfig(model=directory, dtype="float32", max_model_len=64,
                         hf_config=hf)
    draft = SpeculativeConfig(
        method="mtp", num_speculative_tokens=2,
        draft_model_config=None).derive_mtp_draft_config(target)
    spec = SpeculativeConfig(method="mtp", num_speculative_tokens=2,
                             draft_model_config=draft)
    return VllmConfig(
        model_config=target,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=32),
        scheduler_config=SchedulerConfig(max_num_seqs=2,
                                         max_num_batched_tokens=16),
        device_config=DeviceConfig(device="cuda"),
        speculative_config=spec,
        compilation_config=CompilationConfig(cudagraph_mode="none"),
    )


def _v2_engine():
    cfg = _v2_config()
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    runner = (engine.engine_core.engine_core.model_executor
              .driver_worker.model_runner)
    return engine, runner


def test_v2_three_create_destroy_cycles_release_runner_and_do_not_grow():
    allocations = []
    for _ in range(3):
        engine, runner = _v2_engine()
        ref = weakref.ref(runner)
        assert _run(engine)["a"]
        engine.shutdown()
        engine.shutdown()                 # 幂等：第二次不能抛
        del engine, runner
        gc.collect()
        assert ref() is None, "shutdown 后 V2 Runner 仍被引用链持有"
        allocations.append(torch.cuda.memory_allocated())
    assert max(allocations) - min(allocations) < 64 * 1024 * 1024


def test_v2_shutting_one_engine_keeps_another_alive():
    first, first_runner = _v2_engine()
    second, second_runner = _v2_engine()
    first.shutdown()
    assert first_runner._shutdown
    assert not second_runner._shutdown
    assert _run(second)["a"]
    second.shutdown()
