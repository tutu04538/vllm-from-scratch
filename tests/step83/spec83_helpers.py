"""83 关共享夹具：不需要模型权重的 FakeRunner 引擎。"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                      SchedulerConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.config import CompilationConfig
from minivllm.testing.fake_runner import FakeRunner


def make_fake_engine(*, tokens=None, max_model_len=32, blocks=8):
    hf = {"vocab_size": 32}
    cfg = VllmConfig(
        model_config=ModelConfig(
            model="tests/step83/fake-model", dtype="float32",
            max_model_len=max_model_len, hf_config=hf),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
        device_config=DeviceConfig(device="cpu"),
        speculative_config=None,
        compilation_config=CompilationConfig(cudagraph_mode="none"),
    )
    runner = FakeRunner(tokens=tokens)
    subscriber = Worker(cfg, model_runner=runner)
    engine = LLMEngine(cfg, UniProcExecutor(cfg, subscriber), tokenizer=None)
    return engine, runner, subscriber
