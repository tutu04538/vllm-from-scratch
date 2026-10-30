"""84-A：引擎 shutdown/del 后，RoPE 弱缓存不能保活旧模型。"""

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
from minivllm.config import CompilationConfig
from minivllm.testing.tiny_models import tiny_mtp_dir

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="V2 worker 生命周期需要 CUDA")


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


def _run(engine):
    engine.add_request("r", [1, 2, 3, 4],
                       SamplingParams(max_tokens=2, temperature=0.0,
                                      eos_token_id=999))
    for _ in range(100):
        if not engine.has_unfinished_requests():
            break
        engine.step()


def test_engine_shutdown_and_del_release_rope_modules():
    cfg = _v2_config()
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    runner = (engine.engine_core.engine_core.model_executor
              .driver_worker.model_runner)
    modules = [module for _, module in runner.model.named_modules()
               if hasattr(module, "cos_sin_cache")]
    assert modules, "tiny target 应该有 RoPE 层"
    refs = [weakref.ref(module) for module in modules]
    pointers = {module.cos_sin_cache.untyped_storage().data_ptr()
                for module in modules}
    assert len(pointers) == 1, "同一种 tiny sliding RoPE 应共享一份 storage"
    _run(engine)
    engine.shutdown()
    del runner, engine, modules
    gc.collect()
    assert all(ref() is None for ref in refs), (
        "shutdown/del 后 RoPE 弱缓存仍保活旧模型 module")
    # 弱缓存里不应再留下这些 storage 的条目。
    from minivllm.layers.rotary_embedding import _rope_cache_storage_ptrs
    assert pointers.isdisjoint(_rope_cache_storage_ptrs())
