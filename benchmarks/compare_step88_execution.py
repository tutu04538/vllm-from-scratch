"""需求 88：minivllm × vLLM 0.28 的同模式公平对照。

用法：

    python benchmarks/compare_step88_execution.py --dry-run
    python benchmarks/compare_step88_execution.py --all --repeats 3
    python benchmarks/compare_step88_execution.py --suite context --repeats 3
    python benchmarks/compare_step88_execution.py --labels mini_B1_K2_eager --repeats 1

设计约束：

- 父进程只编排，不 import torch/minivllm/vllm；每个 case 一个全新解释器；
- 两边都先预热，再测 `--repeats` 次；每次只数最终 committed output token；
- vLLM 用 `LLM` 建引擎，但用 `llm_engine.add_request()/step()` 手动步进，
  这样两边都能拿到逐 burst 输出时间戳，而不是只能拿 `generate()` 的 wall time；
- vLLM 的主对照使用 `compilation_config.mode=NONE` + `FULL_DECODE_ONLY`，
  不引入 torch.compile；如果 vLLM 在本机不支持该组合，记录失败，不静默退回 eager；
- vLLM 的 EngineCore 在独立进程，显存/KV/图状态通过 `collective_rpc()` 读取；
- 输出身份和性能表分开；跨引擎输出不要求逐 token 相等，但必须记录首分歧。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
for path in (ROOT, BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_runner import case_worker_main, run_matrix  # noqa: E402
from spec_decode_metrics import (AcceptanceAggregate, RequestTiming,  # noqa: E402
                                 median_range, memory_snapshot,
                                 reset_peak_memory, summarize_run)
from step88_common import (  # noqa: E402
    ASSISTANT, CONTEXT_PROMPT, CONTEXT_TARGET_TOKENS, MINI_KV_CACHE_BYTES,
    SHORT_PROMPT, TARGET, VLLM_GRAPH_CAPTURE_SIZES,
    VLLM_GRAPH_MAX_CAPTURE_SIZE, VLLM_KV_CACHE_BYTES,
    build_case_matrix, build_identity_report, build_perf_summary,
    check_identity, matrix_shape, model_manifest, package_hash, source_hashes,
    summarize_matrix_outcome, validate_case_matrix, vllm_boundary_gate,
)
from step88_runtime import reset_peak_worker, worker_snapshot  # noqa: E402

RESULT_DIR = ROOT / "benchmarks" / "results" / "step88"


def _dtype_name(value: Any) -> str:
    text = str(value)
    return text[len("torch."):] if text.startswith("torch.") else text


# ---------------------------------------------------------------------------
# 子进程：加载真实 tokenizer / prompt
# ---------------------------------------------------------------------------

def _load_prompt_ids(tokenizer, workload: str) -> list[int]:
    text = SHORT_PROMPT if workload == "short" else CONTEXT_PROMPT
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if workload == "context":
        if len(ids) < CONTEXT_TARGET_TOKENS:
            raise RuntimeError(
                f"上下文 prompt 只有 {len(ids)} token，少于目标 "
                f"{CONTEXT_TARGET_TOKENS}")
        ids = ids[:CONTEXT_TARGET_TOKENS]
    return list(ids)


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TARGET))


# ---------------------------------------------------------------------------
# mini：真实 V2 runner
# ---------------------------------------------------------------------------

def _mini_graph_snapshot(runner) -> dict[str, Any]:
    def manager_info(manager):
        if manager is None:
            return None
        graphs = getattr(manager, "graphs", None)
        mode = getattr(manager, "cudagraph_mode", None)
        return {
            "class": type(manager).__name__,
            "mode": (mode.name if hasattr(mode, "name") else str(mode)),
            "num_graphs": (len(graphs) if graphs is not None else None),
            "captured_token_counts": (
                [int(value) for value in manager.captured_token_counts()]
                if hasattr(manager, "captured_token_counts") else None),
            "num_replays": int(getattr(manager, "num_replays", 0)),
        }

    target = manager_info(getattr(runner, "target_cudagraph_manager", None))
    speculator = getattr(runner, "speculator", None)
    draft = None
    if speculator is not None:
        draft = {
            "class": type(speculator).__name__,
            "prefill": manager_info(getattr(
                speculator, "prefill_cudagraph_manager", None)),
            "decode": manager_info(getattr(
                speculator, "decode_cudagraph_manager", None)),
        }
    mode = getattr(runner, "cudagraph_mode", None)
    return {
        "actual": (mode.name if hasattr(mode, "name") else str(mode)),
        "target": target,
        "draft": draft,
        "capture_stats": getattr(runner, "cudagraph_capture_stats", None),
    }


def _mini_kv_snapshot(runner) -> dict[str, Any]:
    import torch

    total = 0
    layers = 0
    caches = getattr(runner, "kv_caches", None) or {}
    for value in caches.values():
        values = value if isinstance(value, (tuple, list)) else (value,)
        layers += 1
        for tensor in values:
            if isinstance(tensor, torch.Tensor):
                total += int(tensor.numel() * tensor.element_size())
    config = getattr(runner, "kv_cache_config", None)
    return {
        "bytes": total,
        "layers": layers,
        "num_gpu_blocks": (
            getattr(config, "num_gpu_blocks", None)
            if config is not None else None),
        "block_size": (
            getattr(config, "block_size", None)
            if config is not None else None),
        "spec": str(config) if config is not None else None,
    }


def _mini_identity(runner, case: dict[str, Any]) -> dict[str, Any]:
    config = getattr(runner, "vllm_config", None)
    scheduler = getattr(config, "scheduler_config", None)
    cache = getattr(config, "cache_config", None)
    spec = getattr(config, "speculative_config", None)
    compilation = getattr(config, "compilation_config", None)
    mode = getattr(compilation, "cudagraph_mode", None)
    backend = getattr(runner, "attention_backend", None)
    return {
        "dtype": _dtype_name(getattr(config.model_config, "dtype", ""))
        if config is not None else case["args"]["dtype"],
        "max_model_len": int(getattr(config.model_config, "max_model_len", 0))
        if config is not None else case["args"]["max_model_len"],
        "block_size": int(getattr(cache, "block_size", 0))
        if cache is not None else case["args"]["block_size"],
        "max_num_seqs": int(getattr(scheduler, "max_num_seqs", 0))
        if scheduler is not None else case["batch"],
        "max_num_batched_tokens": int(
            getattr(scheduler, "max_num_batched_tokens", 0))
        if scheduler is not None else case["args"]["budget"],
        "max_new_tokens": case["args"]["max_new_tokens"],
        "enable_prefix_caching": bool(
            getattr(cache, "enable_prefix_caching", False))
        if cache is not None else False,
        "num_speculative_tokens": int(
            getattr(spec, "num_speculative_tokens", 0) or 0)
        if spec is not None else 0,
        "ignore_eos": bool(case["args"]["ignore_eos"]),
        "temperature": 0.0,
        "attention_backend": (backend.get_name()
                              if callable(getattr(backend, "get_name", None))
                              else str(backend)),
        "cudagraph_mode": (mode.name if hasattr(mode, "name") else str(mode)),
        "num_gpu_blocks": (getattr(runner.kv_cache_config, "num_gpu_blocks", None)
                           if getattr(runner, "kv_cache_config", None) is not None
                           else None),
        "compilation_mode": "NONE",
    }


def _mini_run_once(engine, runner, case: dict[str, Any],
                   prompt_ids: list[int]) -> tuple[dict[str, Any], dict[str, list[int]]]:
    import torch

    from minivllm import SamplingParams

    batch = int(case["batch"])
    args = case["args"]
    timings = {
        f"r{index}": RequestTiming(f"r{index}", submitted_at=time.perf_counter())
        for index in range(batch)
    }
    for index in range(batch):
        engine.add_request(
            f"r{index}", list(prompt_ids),
            SamplingParams(max_tokens=args["max_new_tokens"],
                           temperature=0.0, ignore_eos=True))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    steps = 0
    tokens: dict[str, list[int]] = {}
    acceptance = AcceptanceAggregate()
    for _ in range(20000):
        if not engine.has_unfinished_requests():
            break
        outputs = engine.step()
        steps += 1
        now = time.perf_counter()
        for output in outputs:
            timings[output.request_id].observe(now, len(output.token_ids))
            tokens[output.request_id] = list(output.token_ids)
        stats = engine.last_scheduler_stats
        if stats is not None and stats.spec_decoding_stats is not None:
            acceptance.observe(stats.spec_decoding_stats)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - started
    summary = summarize_run(
        timings=timings, acceptance=acceptance, wall_time_s=wall,
        engine_steps=steps,
        preemptions=int(getattr(
            engine.engine_core.engine_core.scheduler,
            "num_preemptions", 0)))
    summary.update({
        "k": int(case["k"]),
        "batch": batch,
        "graph": bool(case["graph"]),
        "max_new_tokens": args["max_new_tokens"],
        "attention_backend": args["attention_backend"],
        "engine_steps": steps,
    })
    return summary, tokens


def _run_mini_case(case: dict[str, Any]) -> dict[str, Any]:
    import torch

    from check_step82_real_spec import _build_engine

    args = SimpleNamespace(**case["args"])
    args.graph = bool(case["graph"])
    args.prefix = False
    args.device = "cuda"
    args.batch = int(case["batch"])
    args.blocks = int(case["args"]["blocks"])
    tokenizer = _tokenizer()
    prompt_ids = _load_prompt_ids(tokenizer, case["workload"])
    engine, runner, build_s = _build_engine(
        args, k=int(case["k"]), batch=int(case["batch"]),
        graph=bool(case["graph"]))
    identity = _mini_identity(runner, case)
    before_warm = _mini_graph_snapshot(runner)
    load_memory = memory_snapshot()
    kv = _mini_kv_snapshot(runner)
    samples = []
    try:
        warm_start = time.perf_counter()
        _mini_run_once(engine, runner, case, prompt_ids)
        warmup_s = time.perf_counter() - warm_start
        after_warm = _mini_graph_snapshot(runner)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            reset_peak_memory()
        measure_start_memory = memory_snapshot()
        for _ in range(int(case["args"]["repeats"])):
            summary, tokens = _mini_run_once(engine, runner, case, prompt_ids)
            summary["tokens"] = tokens
            summary["graph_snapshot"] = _mini_graph_snapshot(runner)
            summary["memory"] = memory_snapshot()
            samples.append(summary)
        measure_end_memory = memory_snapshot()
        after_measure = _mini_graph_snapshot(runner)
    finally:
        engine.shutdown()

    if case["graph"]:
        if not after_warm.get("target") or not after_warm["target"].get("num_graphs"):
            raise RuntimeError(f"{case['label']} 请求了 Graph 但没有捕获 target 图")
        if int(case["k"]) > 0:
            draft = after_warm.get("draft") or {}
            for phase in ("prefill", "decode"):
                item = draft.get(phase) or {}
                if not item.get("num_graphs"):
                    raise RuntimeError(
                        f"{case['label']} 请求了 Graph，但 assistant {phase} "
                        f"没有捕获图：{draft}")
    steady = _graph_steady_capture(before_warm, after_warm, after_measure)
    target_replay_delta = _replay_delta(after_warm, after_measure)
    capture_s = None
    stats = after_measure.get("capture_stats")
    if isinstance(stats, dict):
        capture_s = stats.get("capture_seconds")
    return _case_result(
        case=case, prompt_ids=prompt_ids, identity=identity,
        build_s=build_s, warmup_s=warmup_s, samples=samples,
        load_memory=load_memory, measure_start_memory=measure_start_memory,
        measure_end_memory=measure_end_memory, kv=kv,
        graph={
            "requested": ("FULL_DECODE_ONLY" if case["graph"] else "NONE"),
            "actual": after_measure.get("actual"),
            "steady_capture": steady,
            "replay_delta": target_replay_delta,
            "capture_s": capture_s,
            "after_warmup": after_warm,
            "after_measure": after_measure,
        },
        graph_capture_s=capture_s,
    )


def _graph_steady_capture(before: dict[str, Any], after_warm: dict[str, Any],
                          after_measure: dict[str, Any]) -> bool:
    """测量期间是否又捕获了新图。"""
    def count(snapshot):
        total = 0
        target = snapshot.get("target") or {}
        total += int(target.get("num_graphs") or 0)
        draft = snapshot.get("draft") or {}
        for phase in ("prefill", "decode"):
            item = draft.get(phase) or {}
            total += int(item.get("num_graphs") or 0)
        return total

    return count(after_measure) > count(after_warm)


def _replay_delta(after_warm: dict[str, Any],
                  after_measure: dict[str, Any]) -> dict[str, int]:
    def replay(snapshot):
        target = snapshot.get("target") or {}
        total = int(target.get("num_replays") or 0)
        draft = snapshot.get("draft") or {}
        for phase in ("prefill", "decode"):
            total += int((draft.get(phase) or {}).get("num_replays") or 0)
        return total

    return {"before": replay(after_warm), "after": replay(after_measure),
            "delta": replay(after_measure) - replay(after_warm)}


# ---------------------------------------------------------------------------
# vLLM：LLM 建引擎，手动 add/step；EngineCore 状态走 RPC
# ---------------------------------------------------------------------------

def _vllm_build(case: dict[str, Any], *,
                profiler_dir: str | None = None):
    # `collective_rpc()` 要把只读快照函数送进 EngineCore；本机 benchmark 的
    # 父子进程都是同一份代码，允许 pickle 回退即可。生产路径不依赖这个开关。
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM
    from vllm.config import CompilationConfig

    args = case["args"]
    graph = bool(case["graph"])
    kwargs: dict[str, Any] = {
        "model": str(TARGET),
        "tokenizer": str(TARGET),
        "dtype": args["dtype"],
        "max_model_len": int(args["max_model_len"]),
        "gpu_memory_utilization": float(args["gpu_memory_utilization"]),
        "enforce_eager": not graph,
        "disable_log_stats": False,
        "block_size": int(args["block_size"]),
        "kv_cache_memory_bytes": int(args["vllm_kv_cache_bytes"]),
        "enable_prefix_caching": False,
        "max_num_seqs": int(case["batch"]),
        "max_num_batched_tokens": int(args["budget"]),
        "compilation_config": CompilationConfig(
            mode="NONE",
            cudagraph_mode=("FULL_DECODE_ONLY" if graph else "NONE"),
            cudagraph_capture_sizes=(list(args["vllm_graph_capture_sizes"])
                                     if graph else []),
            max_cudagraph_capture_size=(
                int(args["vllm_graph_max_capture_size"]) if graph else 0),
        ),
        # 文本负载；避免每个 case 花 ~29s 做多模态 processor warmup。
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "skip_mm_profiling": True,
        "seed": 0,
    }
    if int(case["k"]) > 0:
        kwargs.update({
            "spec_method": "mtp",
            "spec_model": str(ASSISTANT),
            "spec_tokens": int(case["k"]),
        })
    if profiler_dir is not None:
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(profiler_dir),
            "torch_profiler_use_gzip": False,
            "torch_profiler_with_stack": False,
            "torch_profiler_dump_cuda_time_total": False,
        }
    started = time.perf_counter()
    llm = LLM(**kwargs)
    build_s = time.perf_counter() - started
    return llm, build_s


def _vllm_rpc(llm, fn) -> dict[str, Any]:
    try:
        values = llm.collective_rpc(fn)
        return values[0] if values else {}
    except Exception as exc:  # noqa: BLE001 —— 测量辅助失败不能伪造数据
        return {"rpc_error": f"{type(exc).__name__}: {exc}"}


def _vllm_identity(llm, case: dict[str, Any],
                   snapshot: dict[str, Any]) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config
    scheduler = config.scheduler_config
    cache = config.cache_config
    compilation = config.compilation_config
    spec = config.speculative_config
    mode = compilation.cudagraph_mode
    return {
        "dtype": _dtype_name(config.model_config.dtype),
        "max_model_len": int(config.model_config.max_model_len),
        "block_size": int(cache.block_size),
        "max_num_seqs": int(scheduler.max_num_seqs),
        "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
        "max_new_tokens": case["args"]["max_new_tokens"],
        "enable_prefix_caching": bool(cache.enable_prefix_caching),
        "num_speculative_tokens": int(
            getattr(spec, "num_speculative_tokens", 0) or 0)
        if spec is not None else 0,
        "ignore_eos": bool(case["args"]["ignore_eos"]),
        "temperature": 0.0,
        "attention_backend": _first_target_backend(snapshot),
        "cudagraph_mode": (mode.name if hasattr(mode, "name") else str(mode)),
        "compilation_mode": (
            compilation.mode.name if hasattr(compilation.mode, "name")
            else str(compilation.mode)),
        "enforce_eager": bool(getattr(config, "enforce_eager", False)),
        "num_gpu_blocks": (snapshot.get("kv") or {}).get("num_gpu_blocks"),
        "kv_cache_memory_bytes": int(cache.kv_cache_memory_bytes or 0),
    }


def _first_target_backend(snapshot: dict[str, Any]) -> str:
    backends = (snapshot.get("attention") or {}).get("target_backends") or []
    return str(backends[0]) if backends else "<unknown>"


def _vllm_run_once(llm, case: dict[str, Any],
                   prompt_ids: list[int]) -> tuple[dict[str, Any], dict[str, list[int]]]:
    from vllm import SamplingParams
    from step88_common import (acceptance_from_spec_metrics,
                                spec_metrics_delta, spec_metrics_snapshot)

    engine = llm.llm_engine
    batch = int(case["batch"])
    args = case["args"]
    timings = {
        f"r{index}": RequestTiming(f"r{index}", submitted_at=time.perf_counter())
        for index in range(batch)
    }
    before_metrics = spec_metrics_snapshot(llm.get_metrics())
    for index in range(batch):
        engine.add_request(
            f"r{index}", list(prompt_ids),
            SamplingParams(max_tokens=args["max_new_tokens"],
                           temperature=0.0, ignore_eos=True))
    started = time.perf_counter()
    steps = 0
    tokens: dict[str, list[int]] = {}
    for _ in range(20000):
        if not engine.has_unfinished_requests():
            break
        outputs = engine.step()
        steps += 1
        now = time.perf_counter()
        for output in outputs:
            if not output.outputs:
                continue
            timings[output.request_id].observe(
                now, len(output.outputs[0].token_ids))
            tokens[output.request_id] = list(output.outputs[0].token_ids)
    wall = time.perf_counter() - started
    after_metrics = spec_metrics_snapshot(llm.get_metrics())
    delta = spec_metrics_delta(after_metrics, before_metrics)
    acceptance_data = acceptance_from_spec_metrics(delta)
    acceptance = AcceptanceAggregate()
    acceptance.num_drafts = acceptance_data["num_drafts"]
    acceptance.num_draft_tokens = acceptance_data["num_draft_tokens"]
    acceptance.num_accepted_tokens = acceptance_data["num_accepted_tokens"]
    acceptance.accepted_per_pos = list(acceptance_data["accepted_per_pos"])
    acceptance.draft_per_pos = list(acceptance_data["draft_per_pos"])
    summary = summarize_run(
        timings=timings, acceptance=acceptance, wall_time_s=wall,
        engine_steps=steps, preemptions=0)
    summary["acceptance"].update({
        "source": acceptance_data["source"],
        "draft_per_pos_available": False,
        "raw_metric_delta": delta,
    })
    summary.update({
        "k": int(case["k"]),
        "batch": batch,
        "graph": bool(case["graph"]),
        "max_new_tokens": args["max_new_tokens"],
        "attention_backend": "TRITON_ATTN",
        "engine_steps": steps,
    })
    return summary, tokens


def _run_vllm_case(case: dict[str, Any]) -> dict[str, Any]:
    import torch

    tokenizer = _tokenizer()
    prompt_ids = _load_prompt_ids(tokenizer, case["workload"])
    llm, build_s = _vllm_build(case)
    snapshot_before = _vllm_rpc(llm, worker_snapshot)
    identity = _vllm_identity(llm, case, snapshot_before)
    load_memory = _snapshot_memory(snapshot_before)
    kv = _snapshot_kv(snapshot_before)
    samples = []
    try:
        warm_start = time.perf_counter()
        _vllm_run_once(llm, case, prompt_ids)
        warmup_s = time.perf_counter() - warm_start
        snapshot_warm = _vllm_rpc(llm, worker_snapshot)
        _vllm_rpc(llm, reset_peak_worker)
        measure_start = _snapshot_memory(_vllm_rpc(llm, worker_snapshot))
        for _ in range(int(case["args"]["repeats"])):
            summary, tokens = _vllm_run_once(llm, case, prompt_ids)
            summary["tokens"] = tokens
            summary["graph_snapshot"] = _vllm_rpc(llm, worker_snapshot)
            summary["memory"] = _snapshot_memory(
                summary["graph_snapshot"])
            samples.append(summary)
        snapshot_end = _vllm_rpc(llm, worker_snapshot)
        measure_end_memory = _snapshot_memory(snapshot_end)
    finally:
        del llm
        gc.collect()

    if case["graph"]:
        actual = _vllm_graph_mode(snapshot_warm)
        if actual not in {"FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "FULL"}:
            raise RuntimeError(
                f"{case['label']} 请求 FULL decode Graph，但实际模式是 {actual!r}；"
                f"snapshot={snapshot_warm}")
        target = (snapshot_warm.get("graph") or {}).get("target") or {}
        if not target.get("num_graphs"):
            raise RuntimeError(
                f"{case['label']} 请求 Graph，但 worker 没有捕获 target 图："
                f"{snapshot_warm.get('graph')}")
    steady = _vllm_steady_capture(snapshot_warm, snapshot_end)
    return _case_result(
        case=case, prompt_ids=prompt_ids, identity=identity,
        build_s=build_s, warmup_s=warmup_s, samples=samples,
        load_memory=load_memory, measure_start_memory=measure_start,
        measure_end_memory=measure_end_memory, kv=kv,
        graph={
            "requested": ("FULL_DECODE_ONLY" if case["graph"] else "NONE"),
            "actual": _vllm_graph_mode(snapshot_warm),
            "steady_capture": steady,
            "replay_delta": None,
            "capture_s": None,
            "before": snapshot_before,
            "after_warmup": snapshot_warm,
            "after_measure": snapshot_end,
            "note": "vLLM 不暴露 replay 计数；Graph 实际重放由 trace_step88 证明",
        },
        graph_capture_s=None,
    )


def _vllm_graph_mode(snapshot: dict[str, Any]) -> str:
    target = (snapshot.get("graph") or {}).get("target") or {}
    return str(target.get("mode") or "<none>")


def _vllm_steady_capture(before: dict[str, Any],
                         after: dict[str, Any]) -> bool:
    def total(snapshot):
        graph = snapshot.get("graph") or {}
        value = 0
        for key in ("target",):
            value += int((graph.get(key) or {}).get("num_graphs") or 0)
        draft = graph.get("draft") or {}
        for key in ("prefill", "decode"):
            value += int((draft.get(key) or {}).get("num_graphs") or 0)
        return value

    return total(after) > total(before)


def _snapshot_memory(snapshot: dict[str, Any]) -> dict[str, Any]:
    cuda = snapshot.get("cuda") or {}
    if not cuda:
        return {"available": False, "rpc_error": snapshot.get("rpc_error")}
    return {
        "available": True,
        "allocated_bytes": cuda.get("allocated_bytes"),
        "reserved_bytes": cuda.get("reserved_bytes"),
        "device_total_bytes": cuda.get("device_total_bytes"),
        "device_free_bytes": cuda.get("device_free_bytes"),
        "peak_allocated_bytes": cuda.get("peak_allocated_bytes"),
        "peak_reserved_bytes": cuda.get("peak_reserved_bytes"),
        "source": "vllm worker RPC",
    }


def _snapshot_kv(snapshot: dict[str, Any]) -> dict[str, Any]:
    kv = dict(snapshot.get("kv") or {})
    kv["source"] = "vllm worker RPC"
    return kv


# ---------------------------------------------------------------------------
# 两边共用结果封装
# ---------------------------------------------------------------------------

def _case_result(*, case: dict[str, Any], prompt_ids: list[int],
                 identity: dict[str, Any], build_s: float, warmup_s: float,
                 samples: list[dict[str, Any]], load_memory: dict[str, Any],
                 measure_start_memory: dict[str, Any],
                 measure_end_memory: dict[str, Any], kv: dict[str, Any],
                 graph: dict[str, Any],
                 graph_capture_s: float | None) -> dict[str, Any]:
    identity_check = check_identity(case, identity)
    tokens = dict(samples[-1].get("tokens") or {})
    tps = median_range([sample["output_tokens_per_s"] for sample in samples])
    wall = median_range([sample["wall_time_s"] for sample in samples])
    acceptance_rates = [sample["acceptance"]["acceptance_rate"]
                        for sample in samples
                        if sample["acceptance"]["acceptance_rate"] is not None]
    ttft = median_range([
        sample["ttft_s"]["mean"] for sample in samples
        if sample["ttft_s"]["mean"] is not None])
    tpot = median_range([
        sample["tpot_s"]["mean"] for sample in samples
        if sample["tpot_s"]["mean"] is not None])
    return {
        "kind": "step88_perf",
        "label": case["label"],
        "pair": case["pair"],
        "engine": case["engine"],
        "batch": int(case["batch"]),
        "k": int(case["k"]),
        "graph": bool(case["graph"]),
        "graph_requested": bool(case["graph"]),
        "workload": case["workload"],
        "prompt_token_ids": prompt_ids,
        "prompt_len": len(prompt_ids),
        "prompt_sha256": hashlib.sha256(
            ",".join(map(str, prompt_ids)).encode()).hexdigest(),
        "build_s": build_s,
        # mini 的图在首次执行/预热时捕获，vLLM 在 build 内捕获；两边都单列
        # graph_capture_s。build_s 是“建引擎/载权重”的墙钟，不重复扣减。
        "load_s_estimate": build_s,
        "graph_capture_s": graph_capture_s,
        "warmup_s": warmup_s,
        "samples": samples,
        "tokens": tokens,
        "output_tokens_per_s": tps,
        "wall_time_s": wall,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "acceptance_rate": median_range(acceptance_rates),
        "acceptance": samples[-1].get("acceptance"),
        "memory": {
            "load_capture": load_memory,
            "measure_start": measure_start_memory,
            "measure_end": measure_end_memory,
        },
        "kv": kv,
        "graph_evidence": graph,
        "identity": identity_check,
        "metric_schema": "step88",
    }


def _dispatch_case(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("kind") != "step88_perf":
        raise ValueError(f"未知 case kind={case.get('kind')!r}")
    engine = case.get("engine")
    if engine == "minivllm":
        return _run_mini_case(case)
    if engine == "vllm":
        return _run_vllm_case(case)
    raise ValueError(f"未知 engine={engine!r}")


# ---------------------------------------------------------------------------
# 父进程：manifest、矩阵、汇总
# ---------------------------------------------------------------------------

def _git() -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"head": head, "dirty": dirty}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _gpu_info() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True)
        devices = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            devices.append({
                "name": parts[0],
                "total_bytes": int(parts[1]) * 1024 * 1024,
                "driver_version": parts[2],
                "temperature_c": parts[3] if len(parts) > 3 else None,
                "power_w": parts[4] if len(parts) > 4 else None,
            })
        return {"source": "nvidia-smi", "devices": devices}
    except Exception as exc:  # noqa: BLE001
        return {"source": "nvidia-smi", "error": str(exc)}


def _versions() -> dict[str, str | None]:
    out = {"python": platform.python_version()}
    for name in ("torch", "triton", "vllm", "transformers"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _command_line() -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                    default=str))
    os.replace(temporary, path)


def _preflight() -> dict[str, Any]:
    missing = [
        str(path.relative_to(ROOT))
        for path in (TARGET / "model.safetensors",
                     ASSISTANT / "model.safetensors",
                     TARGET / "tokenizer.json")
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"缺少模型文件：{missing}")
    return {
        "models": model_manifest(),
        "gpu": _gpu_info(),
        "versions": _versions(),
        "git": _git(),
        "package": package_hash(),
        "source_hashes": source_hashes(),
        "command": _command_line(),
        "cwd": str(ROOT),
    }


def _select_cases(args) -> tuple[list[dict[str, Any]], bool]:
    full = build_case_matrix(include_context=True)
    if args.suite == "main":
        cases = [case for case in full if case["workload"] == "short"]
    elif args.suite == "context":
        cases = [case for case in full if case["workload"] == "context"]
    else:
        cases = list(full)
    if args.labels:
        wanted = set(args.labels)
        cases = [case for case in cases if case["label"] in wanted]
        unknown = wanted - {case["label"] for case in cases}
        if unknown:
            raise SystemExit(f"未知 label：{sorted(unknown)}")
    return cases, args.suite == "all" and not args.labels


def _print_table(summary: dict[str, Any]) -> None:
    print("\ncase                                        tok/s       wall(s)    TTFT(s)")
    for row in summary.get("rows", []):
        print(f"{row['label']:<42} "
              f"{(row.get('output_tokens_per_s') or {}).get('median', 0):>9.2f} "
              f"{(row.get('wall_time_s') or {}).get('median', 0):>10.3f} "
              f"{(row.get('ttft_s') or {}).get('median') or 0:>9.3f}")
    print("\npair                         mini tok/s  vLLM tok/s  mini/vLLM")
    for pair in (summary.get("performance") or {}).get("table", []):
        mini = pair["mini"].get("tok_s") or 0
        vllm = pair["vllm"].get("tok_s") or 0
        ratio = pair.get("mini_over_vllm")
        print(f"{pair['pair']:<28} {mini:>10.2f} {vllm:>11.2f} "
              f"{'n/a' if ratio is None else f'{ratio:.3f}'}")
    identity = summary.get("identity") or {}
    print("\nwithin-engine eager==graph:",
          identity.get("within_engine_exact"),
          "| within acceptance:", identity.get("within_engine_acceptance_same"),
          "| cross-engine exact:", identity.get("cross_engine_exact"),
          "| cross acceptance:", identity.get("cross_engine_same_acceptance"))
    boundary = identity.get("vllm_numeric_boundary") or []
    if boundary:
        print("vLLM numeric boundary (not strict-equivalent):", boundary)
    gate = (summary.get("numeric_gate") or {}).get("vllm_gate") or {}
    if gate and not gate.get("ok", True):
        print("vLLM boundary gate failed:", gate, file=sys.stderr)
    if summary.get("failures"):
        print(f"\nfailures: {len(summary['failures'])}", file=sys.stderr)


def _finalize_summary(out_dir: Path, matrix: dict[str, Any],
                     cases: list[dict[str, Any]], manifest: dict[str, Any],
                     complete_requested: bool,
                     args) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = summarize_matrix_outcome(matrix)
    # 兼容早期 matrix：vLLM 的 measure_start/end 曾直接落 RPC 快照；
    # 正式字段统一成与 mini 同形的 memory snapshot。
    for row in summary["rows"]:
        memory = row.get("memory") or {}
        for key in ("measure_start", "measure_end"):
            value = memory.get(key)
            if isinstance(value, dict) and "cuda" in value:
                memory[key] = _snapshot_memory(value)
    # 配置身份逐 case 审核；缺失/错误都进 summary，不伪造为通过。
    identity_failures = [
        {"label": row["label"], "mismatches": row["identity"]["mismatches"]}
        for row in summary["rows"]
        if not (row.get("identity") or {}).get("ok")
    ]
    mini_exact = bool(summary["identity"].get("mini_within_engine_exact"))
    mini_acceptance = bool(
        summary["identity"].get("mini_within_acceptance_same"))
    vllm_gate = vllm_boundary_gate(summary["identity"])
    numeric_gate = mini_exact and mini_acceptance and vllm_gate["ok"]
    strict_equivalent = bool(
        summary["identity"].get("within_engine_exact")
        and summary["identity"].get("within_engine_acceptance_same"))
    all_expected_ok = len(summary["rows"]) == len(cases)
    summary.update({
        "ok": bool(matrix["ok"] and all_expected_ok and numeric_gate
                   and not identity_failures
                   and (not complete_requested
                        or not summary["identity"]["missing"])),
        "identity_failures": identity_failures,
        "numeric_gate": {
            "mini_within_engine_exact": mini_exact,
            "mini_acceptance_same": mini_acceptance,
            "vllm_gate": vllm_gate,
            "strict_equivalent": strict_equivalent,
        },
        "expected_cases": len(cases),
        "actual_cases": len(summary["rows"]),
        "complete_requested": complete_requested,
        "manifest": manifest,
    })
    if not complete_requested:
        # 调试/冒烟仍要求选中 case 全部成功；不把“没跑全矩阵”伪装成完成。
        # 未选中的配对会在 identity.missing 里，但不应阻断本次子集自检。
        summary["ok"] = bool(matrix["ok"] and all_expected_ok
                            and numeric_gate and not identity_failures)
        summary["partial"] = True
        summary["partial_allowed"] = bool(args.allow_partial or not args.labels)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "results.json", {
        "ok": summary["ok"],
        "metric_schema": "step88",
        "manifest": manifest,
        "summary": summary,
        "matrix": matrix,
    })
    return summary, identity_failures


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--case-json" in argv:
        return case_worker_main(_dispatch_case)

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="跑完整 16+4 矩阵（默认）")
    parser.add_argument("--suite", choices=("all", "main", "context"),
                        default="all")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=5400)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--reprocess", action="store_true",
                        help="复用已有 matrix.json 重算 summary，不重跑 case")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats 必须 >= 1")
    if args.suite == "all":
        args.all = True

    cases, complete_requested = _select_cases(args)
    validation = validate_case_matrix(build_case_matrix(include_context=True))
    if not validation["ok"]:
        print(json.dumps(validation, ensure_ascii=False, indent=1),
              file=sys.stderr)
        return 2
    for case in cases:
        case["args"]["repeats"] = args.repeats
        case["args"]["max_new_tokens"] = args.max_new_tokens
    if args.dry_run:
        print(json.dumps({
            "ok": True,
            "matrix": matrix_shape(cases),
            "validation": validation,
            "cases": cases,
        }, ensure_ascii=False, indent=1))
        return 0
    if args.reprocess:
        out_dir = Path(args.output_dir)
        matrix = json.loads((out_dir / "matrix.json").read_text())
        manifest = json.loads((out_dir / "manifest.json").read_text())
        reprocess_cases = [record["case"] for record in matrix["cases"]]
        summary, identity_failures = _finalize_summary(
            out_dir, matrix, reprocess_cases, manifest, True, args)
        _print_table(summary)
        if identity_failures:
            print(f"identity mismatches: {len(identity_failures)}",
                  file=sys.stderr)
        if not summary["ok"]:
            return 1
        return 0

    preflight = _preflight()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **preflight,
        "suite": args.suite,
        "labels": [case["label"] for case in cases],
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "isolation": "one-engine-case-per-subprocess",
        "matrix_validation": validation,
    }
    _write_json(out_dir / "manifest.json", manifest)
    print(f"manifest: {out_dir / 'manifest.json'}", flush=True)
    print(f"running {len(cases)} cases, repeats={args.repeats}", flush=True)

    child_env = {
        "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    matrix = run_matrix(
        [sys.executable, str(Path(__file__).resolve())],
        cases, timeout_s=args.timeout_s, env=child_env)
    _write_json(out_dir / "matrix.json", matrix)
    summary, identity_failures = _finalize_summary(
        out_dir, matrix, cases, manifest, complete_requested, args)
    _print_table(summary)
    if identity_failures:
        for item in identity_failures:
            print(f"identity mismatch {item['label']}: "
                  f"{item['mismatches']}", file=sys.stderr)
    if not summary["ok"]:
        print(f"step88 未通过：{out_dir / 'summary.json'}", file=sys.stderr)
        return 1
    print(f"step88 完成：{out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
