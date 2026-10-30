"""086：unified attention 小测 + 真实 Gemma4 K=2 eager trace。

用法：

    python benchmarks/check_step86_attention.py --micro
    python benchmarks/check_step86_attention.py --trace
    python benchmarks/check_step86_attention.py --all

微测每个 case 在独立子进程里跑 3 次，每次用 CUDA event 取 5 轮中位数；
结果写到 ignored 的 `benchmarks/results/step86/`。trace 只用于数
kernel/launch，不作为吞吐计时。
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import torch  # noqa: E402

from minivllm.attention.backends.torch_sdpa import TorchAttentionImpl  # noqa: E402
from minivllm.attention.metadata import AttentionMetadata  # noqa: E402
from minivllm.attention.ops.triton_unified_attention import (  # noqa: E402
    make_3d_workspace,
    unified_attention,
)

RESULTS = BENCH / "results" / "step86"
PROMPT = "Hello, my name is"
TARGET = ROOT / "models" / "gemma-4-E2B-it"
ASSISTANT = ROOT / "models" / "gemma-4-E2B-it-assistant"

MICRO_CASES = [
    # name, heads, head_size, q_len, seq_len, window, block_size
    ("decode_swa_d256_s4096", 8, 256, 1, 4096, 512, 16),
    ("decode_full_d512_s2048", 8, 512, 1, 2048, None, 16),
    ("prefill_swa_d256_s2048", 8, 256, 128, 2048, 512, 16),
    ("prefill_full_d512_s2048", 8, 512, 128, 2048, None, 16),
]


def _make_tensors(case):
    name, heads, head_size, q_len, seq_len, window, block_size = case
    device = "cuda"
    num_blocks = (seq_len + block_size - 1) // block_size
    kv = torch.randn(
        2, num_blocks, block_size, 1, head_size,
        device=device, dtype=torch.bfloat16)
    block_table = torch.arange(
        num_blocks, device=device, dtype=torch.int32).view(1, num_blocks)
    q = torch.randn(q_len, heads, head_size, device=device,
                    dtype=torch.bfloat16)
    out = torch.empty_like(q)
    qsl = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    seq = torch.tensor([seq_len], device=device, dtype=torch.int32)
    meta = AttentionMetadata(
        query_start_loc=qsl, seq_lens=seq, block_table=block_table,
        slot_mapping=torch.zeros(q_len, device=device, dtype=torch.int64),
        block_size=block_size, num_reqs=1, max_query_len=q_len,
        sliding_window=window, causal=True)
    workspace = make_3d_workspace(device, heads, head_size, 128, 16)
    return q, kv, out, qsl, seq, block_table, meta, workspace


def _time(fn, *, warmup=3, loops=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(loops):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # us
    return sorted(times)[len(times) // 2]


def _micro_one(case):
    q, kv, out, qsl, seq, block_table, meta, workspace = _make_tensors(case)
    name, heads, head_size, q_len, seq_len, window, block_size = case
    window_size = (-1, -1) if window is None else (window - 1, 0)

    def run_unified():
        unified_attention(
            q, kv[0], kv[1], out, qsl, seq, block_table,
            softmax_scale=1.0, causal=True, window_size=window_size,
            max_seqlen_q=q_len)

    def run_unified_3d():
        unified_attention(
            q, kv[0], kv[1], out, qsl, seq, block_table,
            softmax_scale=1.0, causal=True, window_size=window_size,
            max_seqlen_q=1, seq_threshold_3D=128,
            num_par_softmax_segments=16,
            softmax_segm_output=workspace[0],
            softmax_segm_max=workspace[1],
            softmax_segm_expsum=workspace[2])

    torch_impl = TorchAttentionImpl(heads, head_size, 1.0, 1)

    def run_torch():
        torch_impl._forward_generic(
            q.reshape(q_len, heads * head_size), kv, meta)

    unified_us = _time(run_unified)
    torch_us = _time(run_torch)
    result = {
        "name": name,
        "q_len": q_len,
        "seq_len": seq_len,
        "window": window,
        "head_size": head_size,
        "unified_2d_us": unified_us,
        "torch_generic_us": torch_us,
        "ratio_2d_over_torch": unified_us / torch_us,
    }
    if q_len == 1:
        result["unified_3d_us"] = _time(run_unified_3d)
        result["ratio_3d_over_torch"] = result["unified_3d_us"] / torch_us
    return result


def _run_micro_case(index: int) -> dict:
    return _micro_one(MICRO_CASES[index])


def _micro():
    results = []
    for i, case in enumerate(MICRO_CASES):
        runs = []
        for _ in range(3):
            proc = subprocess.run(
                [sys.executable, __file__, "--micro-one", str(i)],
                cwd=str(ROOT), capture_output=True, text=True, check=True)
            runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        # 3 次独立运行，逐字段取中位数。
        merged = {}
        for key, value in runs[0].items():
            if all(isinstance(run.get(key), (int, float))
                   and not isinstance(run.get(key), bool) for run in runs):
                vals = sorted(run[key] for run in runs)
                merged[key] = vals[len(vals) // 2]
            else:
                merged[key] = value
        results.append(merged)
        print(json.dumps(merged, ensure_ascii=False))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "micro.json").write_text(json.dumps(results, indent=2))
    return results


def _kernel_stats(prof) -> dict:
    from torch.profiler import DeviceType

    cpu_events = [e for e in prof.events() if e.device_type == DeviceType.CPU]
    gpu_events = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    launches = [e for e in cpu_events
                if e.name.startswith(("cudaLaunchKernel", "cudaLaunchKernelExC"))]
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


def _trace():
    from check_step82_real_spec import (
        _build_engine, _load_tokenizer, _prompt_ids, _sampling_params,
    )
    from torch.profiler import ProfilerActivity, profile

    args = argparse.Namespace(
        device="cuda", dtype="bfloat16", attention_backend="triton",
        max_model_len=2048, block_size=16, blocks=512,
        budget=128, batch=1, max_new_tokens=33, repeats=1,
        graph=False, prefix=False, ignore_eos=True,
    )
    tokenizer = _load_tokenizer()
    prompt = _prompt_ids(tokenizer, PROMPT)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    engine, runner, build_s = _build_engine(args, k=2, batch=1, graph=False)
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
        "note": ("真实 target+assistant、K=2、eager；首轮 prefill 在 profiler 外，"
                 "后面 8 个 decode step；只数 kernel/launch，不作吞吐计时"),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "trace.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--micro-one", type=int, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.micro_one is not None:
        print(json.dumps(_run_micro_case(args.micro_one), ensure_ascii=False))
        return
    if not (args.micro or args.trace or args.all):
        parser.error("至少给 --micro / --trace / --all 之一")
    if args.all or args.micro:
        _micro()
    if args.all or args.trace:
        _trace()


if __name__ == "__main__":
    main()
