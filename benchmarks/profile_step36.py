"""对第三十六关的代表性负载跑 profiler，回答「时间花在哪」。

    python benchmarks/profile_step36.py --engine mine --case prefill_c1
    python benchmarks/profile_step36.py --engine vllm --case prefill_c1

vLLM 的实际 GPU 工作在 EngineCore 子进程里，所以走它自己的 start/stop_profile，
trace 写到 VLLM_TORCH_PROFILER_DIR。我们的引擎在同一个进程里，直接用 torch.profiler。
"""

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from collections import Counter

PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from benchmarks.bench_step36_vllm_compare import (CASES, MODEL_DIR, build_mine, build_vllm,
                                                  load_inputs, run_mine, run_vllm)


def classify(name):
    """把 kernel 归到三类，用于和 vLLM 的 trace 比较同一套口径。"""
    if "attention" in name or "flash_fwd" in name:
        return "attention"
    low = name.lower()
    if "cutlass" in low or "gemm" in low or "gemv" in low:
        return "GEMM/GEMV"
    return "elementwise/norm/其他"


def summarize_mine(case, graph=True):
    import torch
    from torch.profiler import profile, ProfilerActivity
    spec = CASES[case]
    prompts = load_inputs(case)
    engine = build_mine(graph=graph)
    run_mine(engine, prompts, spec["gen"], "warm")
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        dt, got, steps = run_mine(engine, prompts, spec["gen"], "prof")
    ev = [e for e in prof.events() if str(getattr(e, "device_type", "")) == "DeviceType.CUDA"]
    kernels = Counter(e.name for e in ev)
    by_name = {}
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            by_name[e.key] = (e.self_device_time_total, e.count)
    gpu_busy = sum(t for t, _ in by_name.values())
    cats = {}
    for n, (t, c) in by_name.items():
        k = classify(n)
        prev = cats.setdefault(k, [0.0, 0])
        prev[0] += t
        prev[1] += c
    return dict(case=case, wall_s=round(dt, 4), steps=steps, gpu_kernels=len(ev),
                distinct_kernels=len(kernels), gpu_busy_us=round(gpu_busy, 1),
                gpu_busy_over_wall=round(gpu_busy / 1e6 / dt, 3),
                categories={k: dict(us=round(v[0], 1), calls=v[1],
                                    share=round(v[0] / gpu_busy, 4))
                            for k, v in sorted(cats.items(), key=lambda kv: -kv[1][0])},
                top=[dict(name=n[:64], us=round(t, 1), calls=c)
                     for n, (t, c) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:12]],
                launch_heavy=[dict(name=n[:64], calls=c) for n, c in kernels.most_common(8)])


def summarize_vllm(case):
    """vLLM 0.28 要显式给 ProfilerConfig 才允许 start_profile；
    只设 VLLM_TORCH_PROFILER_DIR 会被拒绝（"Profiling is not enabled"）。
    ignore_frontend=True：trace 只取真正跑 GPU 的 worker，不含前端进程。"""
    spec = CASES[case]
    prompts = load_inputs(case)
    outdir = PROJECT / "benchmarks" / "results" / "step36_traces"
    outdir.mkdir(parents=True, exist_ok=True)
    llm = build_vllm(profiler_dir=outdir)
    run_vllm(llm, prompts, spec["gen"])          # 预热
    llm.start_profile()
    dt, got = run_vllm(llm, prompts, spec["gen"])
    llm.stop_profile()
    # 默认输出 .pt.trace.json.gz（压缩），glob 要带上 .gz
    traces = sorted(outdir.glob("**/*.trace.json*"), key=lambda p: p.stat().st_mtime)
    return dict(case=case, wall_s=round(dt, 4),
                trace=str(traces[-1]) if traces else None,
                n_traces=len(traces))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("mine", "vllm"), required=True)
    ap.add_argument("--case", choices=tuple(CASES), required=True)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.engine == "mine":
        r = summarize_mine(args.case, graph=not args.no_graph)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        print(f"\n  GPU 忙 {r['gpu_busy_us']/1e3:.2f} ms / 墙钟 {r['wall_s']*1e3:.2f} ms "
              f"= {r['gpu_busy_over_wall']*100:.1f}%")
        print(f"  kernel 发射 {r['gpu_kernels']} 次，{r['distinct_kernels']} 种")
    else:
        print(json.dumps(summarize_vllm(args.case), ensure_ascii=False))


if __name__ == "__main__":
    main()
