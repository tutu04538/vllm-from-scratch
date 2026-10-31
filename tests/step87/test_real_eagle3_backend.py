"""087 收口：真实 Qwen3-1.7B + EAGLE3 草稿层跟随 target 的 Triton backend。

tiny EAGLE3 已经覆盖机制；这里补一份真实 checkpoint 的 eager/FULL 图差分，
避免只证明“tiny 的 11 个 token 能跑”。没有本机权重时整组 skip。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.config import CompilationConfig

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "models" / "Qwen3-1.7B"
DRAFT = ROOT / "models" / "Qwen3-1.7B-eagle3"
PROMPT = [1, 2, 3, 4, 5]
requires_real = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (TARGET / "model.safetensors.index.json").is_file()
    or not (DRAFT / "model.safetensors").is_file(),
    reason="需要本机真实 Qwen3-1.7B + EAGLE3 draft 权重")


def _make_engine(*, mode: str, capture_sizes=(1, 2, 3, 4, 8)):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    target_hf = json.loads((TARGET / "config.json").read_text())
    draft_hf = json.loads((DRAFT / "config.json").read_text())
    cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(TARGET), dtype="bfloat16", max_model_len=512,
            hf_config=target_hf),
        cache_config=CacheConfig(
            block_size=16, num_gpu_blocks=128, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=64),
        device_config=DeviceConfig(device="cuda"),
        speculative_config=SpeculativeConfig(
            method="eagle3", num_speculative_tokens=2,
            draft_model_config=ModelConfig(
                model=str(DRAFT), dtype="bfloat16", max_model_len=512,
                hf_config=draft_hf)),
        compilation_config=CompilationConfig(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=list(capture_sizes) if mode != "none" else None,
            max_cudagraph_capture_size=(
                max(capture_sizes) if mode != "none" else 0)),
        attention_backend="triton",
    )
    engine = LLMEngine(
        cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    runner = (engine.engine_core.engine_core.model_executor
              .driver_worker.model_runner)
    return engine, runner


def _run(engine, *, max_tokens=8):
    engine.add_request(
        "r", list(PROMPT),
        SamplingParams(max_tokens=max_tokens, temperature=0.0,
                       ignore_eos=True))
    out = []
    for _ in range(80):
        if not engine.has_unfinished_requests():
            break
        for item in engine.step():
            out = list(item.token_ids)
    return out


@requires_real
def test_real_eagle3_draft_follows_target_triton_eager_vs_graph():
    from minivllm.attention.backends.triton_attn import (
        TritonAttentionImpl, TritonAttentionMetadataBuilder,
    )

    eager, eager_runner = _make_engine(mode="none")
    want = _run(eager)
    eager.shutdown()

    graph, runner = _make_engine(mode="full")
    got = _run(graph)
    try:
        assert got == want
        assert runner.attention_backend.get_name() == "TRITON_ATTN"
        speculator = runner.speculator
        assert speculator.__class__.__name__ == "EagleSpeculator"
        assert speculator.draft_attention_backend.get_name() == "TRITON_ATTN"
        assert speculator.draft_layer_modules
        assert all(isinstance(m.impl, TritonAttentionImpl)
                   for m in speculator.draft_layer_modules.values())
        assert all(isinstance(g.get_metadata_builder(),
                             TritonAttentionMetadataBuilder)
                   for groups in speculator.attn_groups for g in groups)
        assert speculator.prefill_cudagraph_manager.num_replays > 0
        assert speculator.decode_cudagraph_manager.num_replays > 0
    finally:
        graph.shutdown()
