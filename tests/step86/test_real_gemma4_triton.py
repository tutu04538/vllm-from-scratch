"""086：真实 Gemma4 target + assistant 的 eager/Q-only 验收。"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest
import torch

from spec86_helpers import requires_cuda
from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.attention.backends.torch_sdpa import TorchAttentionImpl
from minivllm.config import CompilationConfig

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "models" / "gemma-4-E2B-it"
ASSISTANT = ROOT / "models" / "gemma-4-E2B-it-assistant"
PROMPT = [2, 818, 5279, 529, 7001, 13]

requires_models = pytest.mark.skipif(
    not (TARGET / "model.safetensors").is_file()
    or not (ASSISTANT / "model.safetensors").is_file(),
    reason="需要本机真实 Gemma4 target/assistant 权重")


def _make_engine(attention_backend: str, *, k: int):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    target_hf = json.loads((TARGET / "config.json").read_text())
    assistant_hf = json.loads((ASSISTANT / "config.json").read_text())
    target = ModelConfig(
        model=str(TARGET), dtype="bfloat16", max_model_len=2048,
        hf_config=target_hf)
    draft = ModelConfig(
        model=str(ASSISTANT), dtype="bfloat16", max_model_len=2048,
        hf_config=assistant_hf)
    spec = None if k == 0 else SpeculativeConfig(
        method=None, num_speculative_tokens=k, draft_model_config=draft)
    cfg = VllmConfig(
        model_config=target,
        cache_config=CacheConfig(
            block_size=16, num_gpu_blocks=1024, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=128),
        device_config=DeviceConfig(device="cuda"),
        speculative_config=spec,
        compilation_config=CompilationConfig(cudagraph_mode="none"),
        attention_backend=attention_backend,
    )
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    runner = (engine.engine_core.engine_core.model_executor
              .driver_worker.model_runner)
    return engine, runner


def _run(engine, max_tokens: int, *, temperature: float = 0.0,
         seed: int | None = None):
    engine.add_request(
        "r", list(PROMPT),
        SamplingParams(max_tokens=max_tokens, temperature=temperature,
                       seed=seed, ignore_eos=True))
    toks = []
    for _ in range(50):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            toks = list(out.token_ids)
    return toks


def _close(engine):
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()


@requires_cuda
@requires_models
def test_real_target_k0_torch_vs_triton():
    torch_engine, _ = _make_engine("torch", k=0)
    want = _run(torch_engine, 8)
    _close(torch_engine)

    triton_engine, runner = _make_engine("triton", k=0)
    got = _run(triton_engine, 8)
    try:
        assert got == want
        assert runner.attention_backend.get_name() == "TRITON_ATTN"
        # target 的 20 个 Q-only 层也必须用 Triton。
        same = [m for _, m in runner.model.named_modules()
                if getattr(m, "q_only", False)]
        assert len(same) == 20
        assert all(type(m.impl).__name__ == "TritonAttentionImpl" for m in same)
    finally:
        _close(triton_engine)


@requires_cuda
@requires_models
def test_real_target_and_assistant_k2_no_fallback_and_rejection():
    torch_engine, _ = _make_engine("torch", k=2)
    want = _run(torch_engine, 8)
    _close(torch_engine)

    def _boom(*args, **kwargs):
        raise AssertionError("TRITON_ATTN 热路径调用了 Torch _forward_generic")

    original = TorchAttentionImpl._forward_generic
    TorchAttentionImpl._forward_generic = _boom
    triton_engine, runner = _make_engine("triton", k=2)
    try:
        got = _run(triton_engine, 8)
        assert got == want, "真实 K=2 target+assistant 的 token 应与 Torch 一致"

        assistant_layers = [
            m for _, m in runner.speculator.model.named_modules()
            if hasattr(m, "impl") and getattr(m, "q_only", False)
        ]
        assert len(assistant_layers) == 4
        assert all(type(m.impl).__name__ == "TritonAttentionImpl"
                   for m in assistant_layers)

        # 随机采样制造拒绝；统计由 Scheduler 的 SpecDecodingStats 提供。
        drafted = accepted = 0
        triton_engine.add_request(
            "rand", list(PROMPT),
            SamplingParams(max_tokens=16, temperature=1.0, seed=0,
                           ignore_eos=True))
        for _ in range(50):
            if not triton_engine.has_unfinished_requests():
                break
            for _out in triton_engine.step():
                pass
            stats = getattr(triton_engine, "last_scheduler_stats", None)
            sd = getattr(stats, "spec_decoding_stats", None)
            if sd is not None:
                drafted += sd.num_draft_tokens
                accepted += sd.num_accepted_tokens
        assert drafted > 0
        assert accepted < drafted, (
            f"没有触发拒绝（drafted={drafted}, accepted={accepted}），"
            "无法覆盖拒绝后的下一轮")

        # 请求结束后 slot 必须能复用：再跑一条请求不能因缓存未释放而失败。
        tail = _run(triton_engine, 4)
        assert len(tail) == 4
    finally:
        TorchAttentionImpl._forward_generic = original
        _close(triton_engine)
