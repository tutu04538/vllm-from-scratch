"""vLLM EngineCore 侧只读快照。

88 的 vLLM 默认把 EngineCore 放在独立进程；父进程的 `torch.cuda.memory_allocated()`
看不到模型/KV/图。用 `LLM.collective_rpc()` 把这个模块里的**顶层函数**送进 worker，
读取真实的 runner 状态。函数只能返回 JSON 友好、体积小的数据。

不要在这里改模型或执行推理；它只用于测量与身份核对。
"""

from __future__ import annotations

import os
from typing import Any


def _iter_tensors(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _unique_storage_bytes(value: Any) -> tuple[int, int, int]:
    """按底层 storage 去重；YOCO 的 Q-only 层会别名到源层 KV。

    直接对 `kv_caches` 的每个 tensor 求和会把同一块物理 KV 重复计数，
    这正是 88 初次探针把 144 MiB 误读成 1.76 GiB 的原因。
    """
    seen: dict[tuple[int, int], int] = {}
    tensors = 0
    for tensor in _iter_tensors(value):
        tensors += 1
        storage = tensor.untyped_storage()
        key = (int(storage.data_ptr()), int(storage.nbytes()))
        seen[key] = key[1]
    return sum(seen.values()), len(seen), tensors


def _graph_manager_snapshot(manager: Any) -> dict[str, Any] | None:
    if manager is None:
        return None
    graphs = getattr(manager, "graphs", None)
    mode = getattr(manager, "cudagraph_mode", None)
    token_counts = None
    captured = None
    if hasattr(manager, "captured_token_counts"):
        try:
            token_counts = [int(value) for value in manager.captured_token_counts()]
        except Exception:  # noqa: BLE001
            token_counts = None
    if hasattr(manager, "captured_graph_count"):
        try:
            captured = int(manager.captured_graph_count())
        except Exception:  # noqa: BLE001
            captured = None
    return {
        "class": type(manager).__name__,
        "mode": (mode.name if hasattr(mode, "name") else str(mode)),
        "num_graphs": (len(graphs) if graphs is not None else None),
        "captured_graph_count": captured,
        "captured_token_counts": token_counts,
        "num_replays": (int(getattr(manager, "num_replays", 0))
                        if hasattr(manager, "num_replays") else None),
    }


def _attention_backends(groups: Any) -> list[str]:
    names: list[str] = []
    for group_list in groups or []:
        for group in group_list:
            backend = getattr(group, "backend", None)
            name = getattr(backend, "get_name", None)
            if callable(name):
                names.append(str(name()))
            elif backend is not None:
                names.append(str(backend))
    return names


def worker_snapshot(worker: Any) -> dict[str, Any]:
    """读取一个 vLLM worker 的显存/KV/Graph/Runner 身份。"""
    import torch

    runner = getattr(worker, "model_runner", None)
    info: dict[str, Any] = {
        "pid": os.getpid(),
        "worker_class": type(worker).__name__,
        "runner_class": (type(runner).__name__ if runner is not None else None),
        "cuda": {},
        "kv": {},
        "graph": {},
        "compilation_counter": {},
        "attention": {},
        "speculator": {},
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info["cuda"] = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "device_free_bytes": int(free),
            "device_total_bytes": int(total),
        }
    if runner is None:
        return info

    kv_caches = getattr(runner, "kv_caches", None)
    kv_bytes = 0
    unique_storages = 0
    kv_tensors = 0
    if isinstance(kv_caches, dict):
        kv_tensors = len(kv_caches)
        kv_bytes, unique_storages, _ = _unique_storage_bytes(
            list(kv_caches.values()))
    elif kv_caches is not None:
        kv_bytes, unique_storages, kv_tensors = _unique_storage_bytes(kv_caches)
    config = getattr(runner, "kv_cache_config", None)
    num_blocks = None
    block_size = None
    groups = None
    if config is not None:
        num_blocks = getattr(config, "num_gpu_blocks", None)
        if num_blocks is None:
            num_blocks = getattr(config, "num_blocks", None)
        block_size = getattr(config, "block_size", None)
        raw_groups = getattr(config, "kv_cache_groups", None) or []
        groups = []
        for group in raw_groups:
            spec = getattr(group, "kv_cache_spec", None)
            groups.append({
                "block_size": getattr(spec, "block_size", None),
                "num_kv_heads": getattr(spec, "num_kv_heads", None),
                "head_size": getattr(spec, "head_size", None),
                "sliding_window": getattr(spec, "sliding_window", None),
                "layer_count": len(getattr(group, "layer_names", []) or []),
            })
    info["kv"] = {
        "bytes": kv_bytes,
        "unique_storages": unique_storages,
        "kv_tensors": kv_tensors,
        "layers": kv_tensors,
        "num_gpu_blocks": num_blocks,
        "block_size": block_size,
        "groups": groups,
    }

    manager = getattr(runner, "cudagraph_manager", None)
    info["graph"]["target"] = _graph_manager_snapshot(manager)
    speculator = getattr(runner, "speculator", None)
    if speculator is not None:
        info["graph"]["draft"] = {
            "class": type(speculator).__name__,
            "prefill": _graph_manager_snapshot(
                getattr(speculator, "prefill_cudagraph_manager", None)),
            "decode": _graph_manager_snapshot(
                getattr(speculator, "decode_cudagraph_manager", None)),
        }
        info["speculator"] = {
            "class": type(speculator).__name__,
            "draft_attention_backend": str(getattr(
                getattr(speculator, "draft_attention_backend", None),
                "get_name", lambda: None)()),
        }
    info["attention"] = {
        "target_backends": _attention_backends(
            getattr(runner, "attn_groups", None)),
        "speculator_backends": _attention_backends(
            getattr(speculator, "attn_groups", None)
            if speculator is not None else None),
    }

    try:
        from vllm.compilation.counter import compilation_counter

        info["compilation_counter"] = {
            "num_cudagraph_captured": int(
                compilation_counter.num_cudagraph_captured),
            "num_gpu_runner_capture_triggers": int(
                compilation_counter.num_gpu_runner_capture_triggers),
        }
    except Exception:  # noqa: BLE001
        pass
    return info


def reset_peak_worker(worker: Any) -> bool:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return True
    return False


__all__ = ["reset_peak_worker", "worker_snapshot"]
