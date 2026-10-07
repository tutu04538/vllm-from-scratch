"""shared fixtures for step69 tests.

69 关要验证的两块东西（对应需求 069 §3/§4）：

    padding     —— 补齐后的批：假行/假请求"物理存在但不留痕"（槽位哨兵、seq_len=0、
                    采样行由 prepare_inputs_padded 给出）
    cudagraph   —— 键 → 热身 → 捕获 → 分派 → 重放：形状补齐、地址稳定、模式回退按规则

这一份夹具把"造引擎 / 造图配置 / 读分派记录 / 比 KV"这几件事收在一起，让两个测试文件
（`test_drafter_padding.py` / `test_spec_cudagraph.py`）各自只关心自己的断言。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,  # noqa: E402
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.config import CompilationConfig, CUDAGraphMode  # noqa: E402
from minivllm.cudagraph_dispatcher import CudagraphDispatcher  # noqa: E402
from minivllm.forward_context import BatchDescriptor  # noqa: E402
from minivllm.testing.tiny_models import tiny_eagle3_dir, tiny_qwen3_config, tiny_qwen3_dir  # noqa: E402

# 59 关起投机验证走 Triton 内核（上游同样只有 GPU 路径）；CUDA Graph 本身也只有 CUDA 有。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
requires_cuda = __import__("pytest").mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA Graph 只能在本机 CUDA 上验证（CPU 上不支持捕获）：待验，不是通过")


def make_config(*, tiny_dir, hf_config, spec_k=None, draft_dir=None, draft_config=None,
                mode=None, budget=32, blocks=32, max_model_len=64, max_num_seqs=2,
                enforce_eager=False, block_size=4, device=DEVICE,
                method="draft_model"):
    """造一份可指定图模式的配置。

    `mode=None` = 用引擎的默认解析（CUDA 上 = `FULL_DECODE_ONLY`，见
    `VllmConfig._resolve_cudagraph_config`）；`"none"` = 纯 eager 参考路径。
    """
    model = ModelConfig(model=tiny_dir, dtype="float32", max_model_len=max_model_len,
                        hf_config=hf_config, enforce_eager=enforce_eager)
    spec = None
    if spec_k is not None:
        spec = SpeculativeConfig(
            method=method, num_speculative_tokens=spec_k,
            draft_model_config=ModelConfig(model=draft_dir or tiny_dir, dtype="float32",
                                           max_model_len=max_model_len,
                                           hf_config=draft_config or hf_config))
    return VllmConfig(
        model_config=model,
        cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=device), speculative_config=spec,
        compilation_config=CompilationConfig(cudagraph_mode=mode))


def make_engine(**kwargs):
    """返回 `(engine, core, runner)`；调用方负责 `engine.shutdown()`。"""
    config = make_config(**kwargs)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def tiny_model(name: str = "tiny_gqa"):
    return tiny_qwen3_dir(name), tiny_qwen3_config(name)


def greedy_outputs(*, prompts=(("r", [1, 2, 3, 4, 5, 6]),), max_tokens=6, **kwargs):
    engine, _core, _runner = make_engine(**kwargs)
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999))
    outputs = run_to_end(engine)
    engine.shutdown()
    return outputs


def run_to_end(engine, limit=400):
    final = {}
    for _ in range(limit):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            final[out.request_id] = list(out.token_ids)
    return final


def graph_snapshot(runner) -> dict:
    """读"图这一侧"的全部可观测状态（测试只读它，不猜内部字段名）。"""
    wrapper = getattr(runner.model, "cudagraph_wrapper", None)
    capture = runner.cudagraph_capture_stats or {}
    return {
        "mode": str(runner.compilation_config.cudagraph_mode),
        "sizes": list(runner.compilation_config.cudagraph_capture_sizes or []),
        "captured": capture.get("captured", 0),
        "keys": sorted(str(desc)
                       for _mode, descs in runner.cudagraph_dispatcher.get_capture_descs()
                       for desc in descs),
        "entries": ([] if wrapper is None
                    else sorted(str(desc) for desc in wrapper.concrete_cudagraph_entries)),
        "replays": (0 if wrapper is None else wrapper.num_replays),
        "captures": (0 if wrapper is None else wrapper.num_captures),
        "selections": list(runner.cudagraph_selections),
    }


def kv_digest(runner, skip_block_zero: bool = True) -> torch.Tensor:
    """所有 attention 层 KV 缓存的摘要（按层名排序拼接）。

    `skip_block_zero=True` 跳过 0 号块：它按约定是**垃圾桶**（padding 行的落点），
    内容本来就该是垃圾；真实块必须逐位可比。
    """
    parts = []
    for name in sorted(runner.kv_caches):
        cache = runner.kv_caches[name]
        flat = cache.reshape(2, cache.shape[1] * cache.shape[2], *cache.shape[3:])
        block_size = cache.shape[2]
        if skip_block_zero:
            flat = flat[:, block_size:]
        parts.append(flat.reshape(-1).float())
    return torch.cat(parts)


class DispatcherHarness:
    """只测分派器所需的**最小**上下文（不建引擎、不装模型）。

    上游 `CudagraphDispatcher` 只依赖 `VllmConfig` 里的 scheduler/compilation/spec 三项，
    所以可以脱离引擎单测——这也是把"键的规则"与"图的捕获"分开验收的前提。
    """

    def __init__(self, *, spec_k=0, budget=32, max_num_seqs=4, mode="full_decode_only",
                 capture_sizes=None, max_capture_size=None, device="cuda"):
        hf = tiny_qwen3_config("tiny_gqa")
        self.config = VllmConfig(
            model_config=ModelConfig(model="dummy", max_model_len=64, hf_config=hf),
            scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                             max_num_batched_tokens=budget),
            device_config=DeviceConfig(device=device),
            speculative_config=(None if spec_k == 0 else SpeculativeConfig(
                method="draft_model", num_speculative_tokens=spec_k,
                draft_model_config=ModelConfig(model="dummy", max_model_len=64, hf_config=hf))),
            compilation_config=CompilationConfig(
                cudagraph_mode=mode, cudagraph_capture_sizes=capture_sizes,
                max_cudagraph_capture_size=max_capture_size))
        self.dispatcher = CudagraphDispatcher(self.config)
        self.dispatcher.initialize_cudagraph_keys(
            self.config.compilation_config.cudagraph_mode,
            1 + self.config.num_speculative_tokens)

    def dispatch(self, num_tokens, uniform_decode=False):
        return self.dispatcher.dispatch(num_tokens, uniform_decode=uniform_decode)

    @property
    def keys(self):
        return self.dispatcher.cudagraph_keys


__all__ = ["DEVICE", "requires_cuda", "DispatcherHarness", "BatchDescriptor",
           "CUDAGraphMode", "CompilationConfig", "graph_snapshot", "kv_digest",
           "greedy_outputs", "make_config", "make_engine", "run_to_end", "tiny_eagle3_dir",
           "tiny_model"]
