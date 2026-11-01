"""88 后续：同一 2D 输入下 mini 与 vLLM 的 unified_attention kernel 对照。

背景：mini 在 B=1 K=2 / 1024 上下文 Graph 下比 vLLM 慢，但此前只比较了
mini 的 2D 与 mini 的 3D，没有回答“为什么同一 2D 路径 mini 更慢”。

这个脚本不加载 engine，直接用同一组 q/paged K/V/block table 调：
  - mini `minivllm.attention.ops.triton_unified_attention.unified_attention`
  - vLLM `vllm.v1.attention.ops.triton_unified_attention.unified_attention`
并读取两边 Triton 编译对象的 `n_regs/n_spills/shared`。结论必须以
“同一输出 + 同一数学”为前提，不能用不同输入/不同 tolerance 解释。
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
for path in (ROOT, BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_step88_attention_micro import (  # noqa: E402
    _grid_info, _prepare, _run, _time_burst, case_matrix,
)
from step88_common import package_hash  # noqa: E402

RESULT_DIR = ROOT / "benchmarks" / "results" / "step88" / "vllm_kernel_diff"
TOLERANCE = {"atol": 2e-2, "rtol": 2e-2}


def _vllm_run(case: dict[str, Any], prepared: dict[str, Any]):
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention as vllm_unified_attention,
    )

    q = prepared["q"]
    k = prepared["k"]
    v = prepared["v"]
    out = prepared["out_vllm"]
    seq_len = case["context"] + case["q_len"]
    vllm_unified_attention(
        q, k, v, out, prepared["cu"], case["q_len"],
        prepared["seq_lens"], seq_len, prepared["scale"], True,
        prepared["window"], prepared["block_table"], 0.0,
        None, None, None)
    return out


def _kernel_meta(module) -> list[dict[str, Any]]:
    """返回该 kernel 当前进程已编译的所有变体。

    hd256/hd512 会各有一个条目；只取 first 会把 hd512 误报成 hd256 的
    register/spill。调用方必须记录 list，不能伪造单一值。
    """
    cache = module.kernel_unified_attention.device_caches[0][0]
    out = []
    for compiled in cache.values():
        metadata = getattr(compiled, "metadata", None)
        out.append({
            "n_regs": int(getattr(compiled, "n_regs", 0)),
            "n_spills": int(getattr(compiled, "n_spills", 0)),
            "shared_bytes": int(getattr(metadata, "shared", 0) or 0),
            "num_warps": int(getattr(metadata, "num_warps", 0) or 0),
            "num_stages": int(getattr(metadata, "num_stages", 0) or 0),
        })
    return out


def _bench_pair(case: dict[str, Any], prepared: dict[str, Any], *,
                warmup: int, repeats: int, rounds: int) -> tuple[dict, dict]:
    import torch

    calls = max(1, repeats // rounds)

    def measure(fn) -> float:
        return _time_burst(fn, calls=calls)

    for _ in range(max(1, warmup // 10)):
        _run(case, "2d", prepared)
        _vllm_run(case, prepared)
    torch.cuda.synchronize()
    mini_rounds, vllm_rounds = [], []
    for index in range(rounds):
        if index % 2 == 0:
            mini_rounds.append(measure(lambda: _run(case, "2d", prepared)))
            vllm_rounds.append(measure(lambda: _vllm_run(case, prepared)))
        else:
            vllm_rounds.append(measure(lambda: _vllm_run(case, prepared)))
            mini_rounds.append(measure(lambda: _run(case, "2d", prepared)))
    return (
        {"median_ms": statistics.median(mini_rounds),
         "min_ms": min(mini_rounds), "max_ms": max(mini_rounds),
         "values_ms": mini_rounds, "calls_per_round": calls},
        {"median_ms": statistics.median(vllm_rounds),
         "min_ms": min(vllm_rounds), "max_ms": max(vllm_rounds),
         "values_ms": vllm_rounds, "calls_per_round": calls},
    )


def _run_all(cases, *, warmup: int, repeats: int,
             rounds: int) -> list[dict[str, Any]]:
    import torch
    from minivllm.attention.ops import triton_unified_attention as mini_module
    from vllm.v1.attention.ops import triton_unified_attention as vllm_module

    results = []
    for index, case in enumerate(cases):
        prepared = _prepare(case)
        # 独立 seed 的 correctness：用同一个 prepared，避免随机数不同。
        mini_out = _run(case, "2d", prepared).clone()
        vllm_error = None
        vllm_out = None
        try:
            vllm_out = _vllm_run(case, prepared).clone()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            vllm_error = f"{type(exc).__name__}: {exc}"
        mini_meta = _kernel_meta(mini_module)
        if vllm_out is None:
            result = {
                **case,
                "correct": None,
                "max_abs": None,
                "mini_grid": _grid_info(case, is_3d=False),
                "mini": {"kernel_meta": mini_meta},
                "vllm": {"unsupported": vllm_error},
                "vllm_speedup": None,
            }
            results.append(result)
            print(
                f"[{index + 1:02d}/{len(cases)}] {case['label']:<28} "
                f"mini=supported vllm=UNSUPPORTED ({vllm_error[:60]})",
                flush=True)
            continue
        delta = (mini_out.float() - vllm_out.float()).abs()
        ok = bool(torch.isfinite(delta).all()
                  and float(delta.max()) <= TOLERANCE["atol"])
        vllm_meta = _kernel_meta(vllm_module)
        mini_t, vllm_t = _bench_pair(
            case, prepared, warmup=warmup, repeats=repeats, rounds=rounds)
        speed = (mini_t["median_ms"] / vllm_t["median_ms"]
                 if vllm_t["median_ms"] else None)
        result = {
            **case,
            "correct": ok,
            "max_abs": float(delta.max()),
            "mini_grid": _grid_info(case, is_3d=False),
            "mini": {**mini_t, "kernel_meta": mini_meta},
            "vllm": {**vllm_t, "kernel_meta": vllm_meta},
            "vllm_speedup": speed,
        }
        results.append(result)
        print(
            f"[{index + 1:02d}/{len(cases)}] {case['label']:<28} "
            f"mini={mini_t['median_ms']:.4f}ms "
            f"vllm={vllm_t['median_ms']:.4f}ms "
            f"speedup={speed:.3f} "
            f"mini_regs={mini_meta[0]['n_regs'] if mini_meta else '?'} "
            f"vllm_variants={[(m['n_regs'], m['n_spills']) for m in vllm_meta]} "
            f"correct={ok}", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cases = case_matrix()
    if args.dry_run:
        print(json.dumps({"ok": True, "cases": cases}, indent=1))
        return 0
    import torch
    if not torch.cuda.is_available():
        print("需要 CUDA", file=sys.stderr)
        return 2
    results = _run_all(cases, warmup=args.warmup, repeats=args.repeats,
                       rounds=args.rounds)
    failures = [row for row in results if row.get("correct") is False]
    payload = {
        "ok": not failures,
        "metric_schema": "step88_vllm_kernel_diff",
        "recorded_at": datetime.datetime.now().astimezone().isoformat(
            timespec="minutes"),
        "device": torch.cuda.get_device_name(0),
        "package": package_hash(),
        "warmup": args.warmup, "repeats": args.repeats,
        "rounds": args.rounds,
        "cases": results,
        "failures": failures,
        "unsupported": [row for row in results if row.get("correct") is None],
        "note": ("同一 q/paged KV/block table、同一 2D max_seqlen_q；"
                 "mini 与 vLLM kernel 输出逐元素对照；"
                 "n_regs/n_spills 来自各自 Triton CompiledKernel"),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    print(f"summary: {out_dir / 'summary.json'}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
