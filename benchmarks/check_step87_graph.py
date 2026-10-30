"""087：真实 Gemma4 K=2 FULL Graph 的 kernel trace 与容量对照。

用法：

    python benchmarks/check_step87_graph.py --trace
    python benchmarks/check_step87_graph.py --capacity
    python benchmarks/check_step87_graph.py --all

trace 只数 kernel/launch，不做吞吐结论；capacity 用同一 prompt/同一有效长度，
只改物理 KV 容量（512 vs 2048 blocks），观察输出/图键/耗时是否被容量牵动。
结果写到 ignored 的 `benchmarks/results/step87/`。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import torch  # noqa: E402

RESULTS = BENCH / "results" / "step87"
PROMPT = "Hello, my name is"


def _kernel_stats(prof) -> dict:
    from torch.profiler import DeviceType

    cpu_events = [e for e in prof.events() if e.device_type == DeviceType.CPU]
    gpu_events = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    launches = [e for e in cpu_events
                if e.name.startswith(("cudaLaunchKernel", "cudaLaunchKernelExC",
                                      "cudaGraphLaunch"))]
    by_name: dict[str, int] = {}
    for e in gpu_events:
        by_name[e.name] = by_name.get(e.name, 0) + 1
    top = sorted(by_name.items(), key=lambda item: item[1], reverse=True)[:25]
    return {
        "gpu_kernel_events": len(gpu_events),
        "cpu_kernel_launch_events": len(launches),
        "distinct_gpu_kernels": len(by_name),
        "unified_attention_events": sum(
            c for n, c in by_name.items() if "kernel_unified_attention" in n),
        "reduce_segments_events": sum(
            c for n, c in by_name.items() if "reduce_segments" in n),
        "reshape_cache_events": sum(
            c for n, c in by_name.items() if "reshape_and_cache_kernel" in n),
        "legacy_paged_events": sum(
            c for n, c in by_name.items()
            if "_paged_attention_kernel" in n or "fixed_paged_attention" in n),
        "top_gpu_kernels": [{"name": n, "count": c} for n, c in top],
    }


def _assistant_workspaces(runner) -> list[dict]:
    speculator = getattr(runner, "speculator", None)
    if speculator is None:
        return []
    out = []
    for groups in getattr(speculator, "attn_groups", []):
        for group in groups:
            builder = group.get_metadata_builder()
            pool = getattr(builder, "workspace_pool", None)
            if pool is None:
                continue
            out.append({
                "layers": list(group.layer_names),
                "backend": group.backend.get_name(),
                "builder": type(builder).__name__,
                "workspace_buckets": [
                    {"key": repr(key), "bytes": pool.bucket_bytes(key)}
                    for key in pool.keys()
                ],
            })
    return out


def _trace() -> dict:
    from check_step82_real_spec import (
        _build_engine, _load_tokenizer, _prompt_ids, _sampling_params,
    )
    from torch.profiler import ProfilerActivity, profile

    args = argparse.Namespace(
        device="cuda", dtype="bfloat16", attention_backend="triton",
        max_model_len=2048, block_size=16, blocks=512,
        budget=128, batch=1, max_new_tokens=33, repeats=1,
        graph=True, prefix=False, ignore_eos=True,
    )
    tokenizer = _load_tokenizer()
    prompt = _prompt_ids(tokenizer, PROMPT)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    engine, runner, build_s = _build_engine(args, k=2, batch=1, graph=True)
    tokens: list[int] = []
    profiled_steps = 0
    try:
        engine.add_request(
            "r", prompt,
            _sampling_params(argparse.Namespace(
                **{**vars(args), "ignore_eos": True})))
        # prefill + 首个 token 放在 profiler 外。
        first = engine.step()
        for out in first:
            tokens = list(out.token_ids)
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for _ in range(8):
                if not engine.has_unfinished_requests():
                    break
                for out in engine.step():
                    tokens = list(out.token_ids)
                profiled_steps += 1
        stats = _kernel_stats(prof)
        memory = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        capture = {
            "stats": runner.cudagraph_capture_stats,
            "target_graphs": [
                {"num_tokens": d.num_tokens, "num_reqs": d.num_reqs,
                 "uniform_token_count": d.uniform_token_count,
                 "max_query_len": d.max_query_len}
                for d in getattr(runner.target_cudagraph_manager, "graphs", {})
            ],
            "target_selections": list(
                getattr(runner.target_cudagraph_manager, "selections", [])),
            "draft_selections": list(
                getattr(runner.speculator, "cudagraph_selections", [])),
            "target_workspace_buckets": [
                {"key": repr(key), "bytes": (
                    runner.attn_metadata_builder.workspace_pool.bucket_bytes(key))}
                for key in runner.attn_metadata_builder.workspace_pool.keys()
            ],
            "assistant_workspaces": _assistant_workspaces(runner),
        }
    finally:
        engine.shutdown()
    del runner, engine
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "prompt_len": len(prompt),
        "build_s": build_s,
        "tokens": tokens,
        "profiled_decode_steps": profiled_steps,
        "kernel_stats": stats,
        "per_step": {
            "gpu_kernel_events": stats["gpu_kernel_events"] / max(profiled_steps, 1),
            "cpu_kernel_launch_events": (
                stats["cpu_kernel_launch_events"] / max(profiled_steps, 1)),
            "unified_attention_events": (
                stats["unified_attention_events"] / max(profiled_steps, 1)),
            "reduce_segments_events": (
                stats["reduce_segments_events"] / max(profiled_steps, 1)),
            "reshape_cache_events": (
                stats["reshape_cache_events"] / max(profiled_steps, 1)),
        },
        "memory": memory,
        "capture": capture,
        "note": ("真实 target+assistant、K=2、FULL decode graph；首轮 prefill 在 "
                 "profiler 外，后面 8 个 decode step；只数 kernel/launch，不作吞吐计时"),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "trace.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False))
    return result


def _run_graph_once(args, *, prompt, max_new_tokens=33) -> dict:
    from check_step82_real_spec import (
        _build_engine, _sampling_params,
    )

    engine, runner, build_s = _build_engine(args, k=2, batch=1, graph=True)
    tokens: list[int] = []
    try:
        engine.add_request(
            "r", prompt,
            _sampling_params(argparse.Namespace(
                **{**vars(args), "ignore_eos": True})))
        # 第一轮 prefill/首个 decode 在计时外。
        for out in engine.step():
            tokens = list(out.token_ids)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(8):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                tokens = list(out.token_ids)
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
        captures = runner.cudagraph_capture_stats
        keys = sorted(
            (d.num_tokens, d.num_reqs, d.uniform_token_count, d.max_query_len)
            for d in getattr(runner.target_cudagraph_manager, "graphs", {})
        )
        workspace_bytes = runner.attn_metadata_builder.workspace_pool.num_bytes()
        return {
            "build_s": build_s,
            "tokens": tokens,
            "elapsed_ms_8_steps": elapsed_ms,
            "captures": captures,
            "target_graph_keys": keys,
            "target_workspace_bytes": workspace_bytes,
        }
    finally:
        engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def _capacity() -> dict:
    from check_step82_real_spec import _load_tokenizer, _prompt_ids

    tokenizer = _load_tokenizer()
    prompt = _prompt_ids(tokenizer, PROMPT)
    base = dict(
        device="cuda", dtype="bfloat16", attention_backend="triton",
        max_model_len=2048, block_size=16, budget=128, batch=1,
        max_new_tokens=33, repeats=1, graph=True, prefix=False,
        ignore_eos=True,
    )
    results = {}
    for blocks in (512, 2048):
        args = argparse.Namespace(**base, blocks=blocks)
        results[str(blocks)] = _run_graph_once(args, prompt=prompt)
        print(json.dumps({"blocks": blocks, **results[str(blocks)]},
                         ensure_ascii=False))
    out = {
        "blocks_512": results["512"],
        "blocks_2048": results["2048"],
        "tokens_equal": results["512"]["tokens"] == results["2048"]["tokens"],
        "same_graph_keys": (
            results["512"]["target_graph_keys"]
            == results["2048"]["target_graph_keys"]),
        "elapsed_ratio_2048_over_512": (
            results["2048"]["elapsed_ms_8_steps"]
            / max(results["512"]["elapsed_ms_8_steps"], 1e-9)),
        "note": ("同一 prompt/有效长度，只改物理 block 容量；耗时包含 8 个 "
                 "decode step 的完整 engine.step，噪声大，不当作 kernel 微基准"),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "capacity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, ensure_ascii=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not (args.trace or args.capacity or args.all):
        parser.error("至少给 --trace / --capacity / --all 之一")
    if args.all or args.trace:
        _trace()
    if args.all or args.capacity:
        _capacity()


if __name__ == "__main__":
    main()
