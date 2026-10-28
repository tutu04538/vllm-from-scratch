"""81 关共享夹具：tiny V1 投机 + tiny V2 MTP。"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.config import CompilationConfig
from minivllm.testing.tiny_models import (tiny_mtp_dir, tiny_qwen3_config,
                                          tiny_qwen3_dir)


def make_v1_engine(*, spec_k=3, blocks=32, budget=16, max_num_seqs=2,
                   max_model_len=64, prefix=False, device="cuda"):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    directory = tiny_qwen3_dir("tiny_gqa")
    hf = tiny_qwen3_config("tiny_gqa")
    target = ModelConfig(model=directory, dtype="float32",
                         max_model_len=max_model_len, hf_config=hf)
    if spec_k is None:
        spec = None
    else:
        draft = ModelConfig(model=directory, dtype="float32",
                            max_model_len=max_model_len, hf_config=hf)
        spec = SpeculativeConfig(method="draft_model",
                                 num_speculative_tokens=spec_k,
                                 draft_model_config=draft)
    cfg = VllmConfig(
        model_config=target,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks,
                                 enable_prefix_caching=prefix),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=device),
        speculative_config=spec,
        compilation_config=CompilationConfig(cudagraph_mode="none"))
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)))
    return engine, engine.engine_core.engine_core


def make_v2_mtp_engine(*, spec_k=2, blocks=64, budget=16, max_num_seqs=3,
                       max_model_len=64, device="cuda"):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
    directory = tiny_mtp_dir("tiny_gqa")
    hf = json.loads((Path(directory) / "config.json").read_text())
    target = ModelConfig(model=directory, dtype="float32",
                         max_model_len=max_model_len, hf_config=hf)
    draft = SpeculativeConfig(
        method="mtp", num_speculative_tokens=spec_k,
        draft_model_config=None).derive_mtp_draft_config(target)
    spec = SpeculativeConfig(method="mtp", num_speculative_tokens=spec_k,
                             draft_model_config=draft)
    cfg = VllmConfig(
        model_config=target,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=device),
        speculative_config=spec,
        compilation_config=CompilationConfig(cudagraph_mode="none"))
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)))
    return engine, engine.engine_core.engine_core


def run_engine(engine, requests, *, max_tokens=4, eos_token_id=999,
               max_steps=400, collect_stats=False):
    """requests: [(id, prompt)]；返回 ({id: tokens}, 每步 stats, 每步提交增量)。"""
    outputs: dict[str, list[int]] = {}
    stats = []
    committed = []
    for req_id, prompt in requests:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=eos_token_id))
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            break
        step_outputs = engine.step()
        increments = 0
        for out in step_outputs:
            previous = len(outputs.get(out.request_id, []))
            outputs[out.request_id] = list(out.token_ids)
            increments += len(out.token_ids) - previous
        committed.append(increments)
        stats.append(engine.last_scheduler_stats)
        if not collect_stats:
            continue
    return outputs, stats, committed
