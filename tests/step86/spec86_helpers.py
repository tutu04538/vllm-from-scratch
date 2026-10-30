"""086 测试辅助：CUDA 标记 + tiny Qwen3 V2 引擎。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (  # noqa: E402
    CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
    SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker,
)
from minivllm.config import CompilationConfig  # noqa: E402
from minivllm.testing.tiny_models import (  # noqa: E402
    tiny_eagle3_dir, tiny_qwen3_config, tiny_qwen3_dir,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="需要 CUDA")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")
DRAFT = tiny_eagle3_dir("tiny_gqa")
DRAFT_HF = json.loads((Path(DRAFT) / "config.json").read_text())
VOCAB = int(HF["vocab_size"])
EOS = VOCAB + 7


def make_engine(*, attention_backend="triton", mode="none", spec_k=3,
                capture_sizes=(1, 2, 3, 4, 8), max_num_seqs=2,
                block_size=4, budget=64, blocks=64):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    spec = None
    if spec_k is not None:
        spec = SpeculativeConfig(
            method="eagle3", num_speculative_tokens=spec_k,
            draft_sample_method="greedy",
            draft_model_config=ModelConfig(
                model=DRAFT, dtype="float32", max_model_len=64,
                hf_config=DRAFT_HF))
    config = VllmConfig(
        model_config=ModelConfig(
            model=TINY, dtype="float32", max_model_len=64, hf_config=HF),
        cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE),
        speculative_config=spec,
        compilation_config=CompilationConfig(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=list(capture_sizes) if capture_sizes else None),
        attention_backend=attention_backend,
    )
    engine = LLMEngine(
        config, UniProcExecutor(config, Worker(config)), tokenizer=None)
    core = engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.model_runner
    return engine, runner


def run_engine(engine, prompts, max_tokens=5):
    for req_id, prompt in prompts:
        engine.add_request(
            req_id, list(prompt),
            SamplingParams(max_tokens=max_tokens, temperature=0.0,
                           eos_token_id=EOS))
    out = {}
    for _ in range(50):
        if not engine.has_unfinished_requests():
            break
        for item in engine.step():
            out[item.request_id] = list(item.token_ids)
    return out
