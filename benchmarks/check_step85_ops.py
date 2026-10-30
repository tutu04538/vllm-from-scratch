"""85 关基准：Norm/激活/QKV 小测 + 一个真实 K=0 eager 8-step trace。

用法：

    python benchmarks/check_step85_ops.py --micro
    python benchmarks/check_step85_ops.py --trace
    python benchmarks/check_step85_ops.py --all

边界：
- `--micro` 只测固定 shape 的算子/GEMM，不代表端到端吞吐；
- `--trace` 只跑真实 target（K=0、eager、单 prompt、8 个 decode step）并数
  CUDA kernel/launch；不跑 B×K 性能矩阵；
- 计时包含 Python launch 间隙，不是 kernel-only；正式吞吐仍属 88。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_runner import case_worker_main, run_matrix  # noqa: E402
from check_step82_real_spec import (  # noqa: E402
    _build_engine, _git, _load_tokenizer, _package_hash, _parent_gpu_info,
    _prompt_ids, _sampling_params,
)

TARGET = ROOT / "models" / "gemma-4-E2B-it"
PROMPT = "请用一句话解释什么是 KV cache。"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")


def _worker_argv() -> list[str]:
    return [sys.executable, str(Path(__file__).resolve())]


def _median_time(fn, iters: int) -> float:
    import torch

    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    samples.sort()
    return samples[len(samples) // 2]


def _median_runs(fn, iters: int, runs: int = 3) -> tuple[float, list[float]]:
    """单次 event loop 仍受 GPU 时钟/调度噪声影响；做 3 次独立中位数再取中位数。"""
    values = [_median_time(fn, iters) for _ in range(runs)]
    return sorted(values)[len(values) // 2], values


def _case_micro(case: dict) -> dict:
    import torch
    import torch.nn.functional as F

    from minivllm.layers.ops.activation import (activation_and_mul,
                                                activation_and_mul_native)
    from minivllm.layers.ops.layernorm import rms_norm, rms_norm_native

    device = "cuda"
    dtype = torch.bfloat16
    gen = torch.Generator(device=device).manual_seed(8501)

    def rand(*shape, dt=dtype):
        return torch.randn(*shape, dtype=dt, device=device, generator=gen)

    norm_cases = [
        # name, shape, weight dtype, add_one, residual, iters
        ("decode_hidden_1x1536", (1, 1536), dtype, False, False, 500),
        ("prefill_hidden_128x1536", (128, 1536), dtype, False, False, 100),
        ("decode_qnorm_1x8x256", (1, 8, 256), dtype, False, False, 500),
        ("decode_qnorm_1x8x512", (1, 8, 512), dtype, False, False, 500),
        ("decode_vnorm_noweight_1x512", (1, 512), None, False, False, 500),
        ("decode_gemma_1x1536", (1, 1536), torch.float32, True, False, 500),
        ("decode_hidden_residual_1x1536", (1, 1536), dtype, False, True, 500),
    ]
    norm_rows = []
    for name, shape, weight_dtype, add_one, with_residual, iters in norm_cases:
        x = rand(*shape)
        weight = None if weight_dtype is None else rand(
            shape[-1], dt=weight_dtype)
        residual = rand(*shape) if with_residual else None
        got = rms_norm(x, weight, 1e-6, residual, add_one=add_one)
        want = rms_norm_native(x, weight, 1e-6, residual, add_one=add_one)
        if with_residual:
            err = max(
                (got[0].float() - want[0].float()).abs().max().item(),
                (got[1].float() - want[1].float()).abs().max().item())
        else:
            err = (got.float() - want.float()).abs().max().item()
        fused_us, fused_runs = _median_runs(
            lambda: rms_norm(x, weight, 1e-6, residual, add_one=add_one),
            iters)
        native_us, native_runs = _median_runs(
            lambda: rms_norm_native(x, weight, 1e-6, residual, add_one=add_one),
            iters)
        norm_rows.append({
            "name": name,
            "shape": list(shape),
            "weight_dtype": str(weight_dtype) if weight_dtype else None,
            "add_one": add_one,
            "residual": with_residual,
            "iters": iters,
            "fused_us_per_call": fused_us,
            "native_us_per_call": native_us,
            "fused_us_runs": fused_runs,
            "native_us_runs": native_runs,
            "fused_over_native_including_launch": (
                fused_us / native_us if native_us else None),
            "max_abs_error_vs_native": err,
        })

    act_cases = [
        # name, shape, gelu, approximate, iters
        ("decode_gelu_1x12288", (1, 12288), True, "tanh", 500),
        ("decode_gelu_double_1x24576", (1, 24576), True, "tanh", 500),
        ("prefill_gelu_128x24576", (128, 24576), True, "tanh", 50),
        ("decode_silu_1x17920", (1, 17920), False, "none", 500),
    ]
    act_rows = []
    for name, shape, gelu, approximate, iters in act_cases:
        x = rand(*shape)
        got = activation_and_mul(x, gelu=gelu, approximate=approximate)
        want = activation_and_mul_native(x, gelu=gelu, approximate=approximate)
        # backend 的 BF16 写回应与 FP32 参考的 BF16 舍入一致。
        ref = activation_and_mul_native(
            x.float(), gelu=gelu, approximate=approximate).to(x.dtype)
        err = (got.float() - ref.float()).abs().max().item()
        fused_us, fused_runs = _median_runs(
            lambda: activation_and_mul(x, gelu=gelu, approximate=approximate),
            iters)
        native_us, native_runs = _median_runs(
            lambda: activation_and_mul_native(
                x, gelu=gelu, approximate=approximate), iters)
        act_rows.append({
            "name": name,
            "shape": list(shape),
            "gelu": gelu,
            "approximate": approximate,
            "iters": iters,
            "fused_us_per_call": fused_us,
            "native_us_per_call": native_us,
            "fused_us_runs": fused_runs,
            "native_us_runs": native_runs,
            "fused_over_native_including_launch": (
                fused_us / native_us if native_us else None),
            "max_abs_error_vs_fp32_reference_bf16": err,
            "native_bf16_vs_fp32_reference_max_abs": (
                want.float() - ref.float()).abs().max().item(),
        })

    qkv_rows = []
    hidden = 1536
    for name, q_size, kv_size, tokens, iters in (
        ("decode_sliding_1x", 2048, 256, 1, 500),
        ("decode_full_1x", 4096, 512, 1, 500),
        ("prefill_sliding_128x", 2048, 256, 128, 50),
    ):
        x = rand(tokens, hidden)
        packed_w = rand(q_size + 2 * kv_size, hidden)
        qw = rand(q_size, hidden)
        kw = rand(kv_size, hidden)
        vw = rand(kv_size, hidden)
        packed_us, packed_runs = _median_runs(
            lambda: F.linear(x, packed_w), iters)
        split_us, split_runs = _median_runs(
            lambda: (F.linear(x, qw), F.linear(x, kw), F.linear(x, vw)),
            iters)
        qkv_rows.append({
            "name": name,
            "tokens": tokens,
            "q_size": q_size,
            "kv_size": kv_size,
            "iters": iters,
            "packed_gemm_us": packed_us,
            "three_gemm_us": split_us,
            "packed_gemm_us_runs": packed_runs,
            "three_gemm_us_runs": split_runs,
            "packed_over_three_including_launch": (
                packed_us / split_us if split_us else None),
            "packed_rows": q_size + 2 * kv_size,
        })

    return {
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "norm": norm_rows,
        "activation": act_rows,
        "qkv_gemm": qkv_rows,
        "timing": "median of 5 CUDA-event loops; includes Python launch gaps",
    }


def _kernel_stats(prof) -> dict:
    from torch.profiler import DeviceType

    cpu_events = [event for event in prof.events()
                  if event.device_type == DeviceType.CPU]
    gpu_events = [event for event in prof.events()
                  if event.device_type == DeviceType.CUDA]
    launches = [event for event in cpu_events
                if event.name.startswith(("cudaLaunchKernel",
                                          "cudaLaunchKernelExC"))]
    memcpy = [event for event in cpu_events
              if event.name.startswith("cudaMemcpy")]
    by_name: dict[str, int] = {}
    for event in gpu_events:
        by_name[event.name] = by_name.get(event.name, 0) + 1
    top = sorted(by_name.items(), key=lambda item: item[1], reverse=True)[:20]
    return {
        "gpu_kernel_events": len(gpu_events),
        "cpu_kernel_launch_events": len(launches),
        "cpu_memcpy_events": len(memcpy),
        "distinct_gpu_kernels": len(by_name),
        "triton_rms_norm_kernel_events": sum(
            count for name, count in by_name.items()
            if "_rms_norm_kernel" in name),
        "triton_act_and_mul_kernel_events": sum(
            count for name, count in by_name.items()
            if "_act_and_mul_kernel" in name),
        "top_gpu_kernels": [
            {"name": name, "count": count} for name, count in top
        ],
    }


def _case_trace(case: dict) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile

    args = argparse.Namespace(
        device="cuda", dtype="bfloat16", attention_backend="torch",
        max_model_len=2048, block_size=16, blocks=512,
        budget=128, batch=1, max_new_tokens=9, repeats=1,
        graph=False, prefix=False, ignore_eos=True,
    )
    tokenizer = _load_tokenizer()
    prompt = _prompt_ids(tokenizer, PROMPT)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    engine, runner, build_s = _build_engine(
        args, k=0, batch=1, graph=False)
    tokens: list[int] = []
    profiled_steps = 0
    try:
        engine.add_request(
            "r", prompt,
            _sampling_params(argparse.Namespace(
                **{**vars(args), "max_new_tokens": 9, "ignore_eos": True})))
        # 预填 + 首 token 放在 profiler 外；后面恰好 8 个 decode step。
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
        after = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
    finally:
        engine.shutdown()
    del runner, engine
    gc.collect()
    torch.cuda.empty_cache()

    # 每个 decode step 的理论 RMSNorm 调用次数：35 层 × 4 block norm
    # + 35 q_norm + 15×(k_norm+v_norm) + final norm；PLE 另加
    # 1 个 per_layer_projection_norm + 35 个 post_per_layer_input_norm。
    theoretical_norm_calls_per_step = 35 * 4 + 35 + 15 * 2 + 1 + 1 + 35
    theoretical_act_calls_per_step = 35
    return {
        "prompt_len": len(prompt),
        "build_s": build_s,
        "tokens": tokens,
        "num_tokens": len(tokens),
        "profiled_decode_steps": profiled_steps,
        "kernel_stats": stats,
        "per_step": {
            "gpu_kernel_events": stats["gpu_kernel_events"] / max(profiled_steps, 1),
            "cpu_kernel_launch_events": (
                stats["cpu_kernel_launch_events"] / max(profiled_steps, 1)),
            "theoretical_rms_norm_calls": theoretical_norm_calls_per_step,
            "theoretical_act_and_mul_calls": theoretical_act_calls_per_step,
        },
        "memory": after,
        "note": ("单 prompt、真实 target、K=0/eager、8 个 decode step；"
                 "profiler 只用于数 kernel/launch，不作为吞吐计时"),
    }


def _dispatch(case: dict) -> dict:
    kind = case["kind"]
    if kind == "micro":
        return _case_micro(case)
    if kind == "trace":
        return _case_trace(case)
    raise ValueError(f"未知 case kind={kind!r}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--case-json" in argv:
        return case_worker_main(_dispatch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir",
                        default=str(ROOT / "benchmarks/results/step85"))
    parser.add_argument("--timeout-s", type=float, default=1800)
    args = parser.parse_args(argv)
    if args.all:
        args.micro = args.trace = True
    if not (args.micro or args.trace):
        args.micro = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git": _git(),
        "package": _package_hash(),
        "gpu": _parent_gpu_info(),
        "target": {"path": str(TARGET)},
        "args": vars(args),
        "isolation": "one-mode-per-subprocess",
    }
    _write(out_dir / "manifest.json", manifest)
    cases = []
    if args.micro:
        cases.append({"kind": "micro", "label": "micro"})
    if args.trace:
        cases.append({"kind": "trace", "label": "trace"})
    result = run_matrix(
        _worker_argv(), cases, timeout_s=args.timeout_s, cwd=ROOT)
    _write(out_dir / "results.json", result)
    for record in result["cases"]:
        payload = record.get("result") or {}
        print(f"{record['case']['label']}: {record['status']} "
              f"({payload.get('_case_duration_s', 0):.1f}s)")
    return 0 if all(record["status"] == "ok" for record in result["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
