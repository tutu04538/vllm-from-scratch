"""88 后续诊断：2D 长上下文分页 attention 的并行度 microbenchmark。

只测 `unified_attention` kernel 本身，不加载模型、不跑 engine：

    2D: grid=(query_blocks, num_kv_heads)，KV tile 在 program 内串行
    3D: grid=(query_blocks, num_kv_heads, segments)，KV 切段后 reduce

K=2 target 的 q_len=3 目前被 `can_use_3d()` 挡在 3D 外；这里**只为诊断**把
`max_seqlen_q` 传成 1，强制 3D，验证 q_len>1 的数值/性能是否可行。生产路径
不改变，除非后续单独接 3D + Graph。

用法：
  python benchmarks/check_step88_attention_micro.py --dry-run
  python benchmarks/check_step88_attention_micro.py --repeats 100
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
for path in (ROOT, BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from step88_common import package_hash  # noqa: E402

RESULT_DIR = ROOT / "benchmarks" / "results" / "step88" / "attention_micro"
SEGMENTS = 16
TOLERANCE = {"atol": 2e-2, "rtol": 2e-2}


def _next_pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def case_matrix() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(label, *, q_len, context, batch=1, heads=8, kv_heads=1,
            head_dim=256, block_size=16, window=-1, segments=SEGMENTS):
        cases.append({
            "label": label,
            "q_len": int(q_len),
            "context": int(context),
            "batch": int(batch),
            "heads": int(heads),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
            "block_size": int(block_size),
            "window": int(window),
            "segments": int(segments),
        })

    for q_len in (1, 2, 4, 8):
        add(f"qlen{q_len}_ctx1024", q_len=q_len, context=1024)
    for context in (128, 256, 512, 1024, 2048):
        add(f"qlen3_ctx{context}", q_len=3, context=context)
    for batch in (2, 4, 8):
        add(f"qlen3_b{batch}_ctx1024", q_len=3, context=1024, batch=batch)
    for head_dim in (256, 512):
        for window in (-1, 512):
            add(f"qlen3_hd{head_dim}_win{window}",
                q_len=3, context=1024, head_dim=head_dim, window=window)
    for segments in (4, 8, 16, 32):
        add(f"qlen3_seg{segments}", q_len=3, context=1024,
            segments=segments)
    # 贴近真实 Gemma4：sliding 256 + full 512 两种层各取一个代表点。
    add("qlen3_hd512_win512_b4", q_len=3, context=1024, batch=4,
        head_dim=512, window=512)
    add("qlen3_hd256_win512_b4", q_len=3, context=1024, batch=4,
        head_dim=256, window=512)
    return cases


def _grid_info(case: dict[str, Any], *, is_3d: bool) -> dict[str, int]:
    group = case["heads"] // case["kv_heads"]
    block_m = 16 if group <= 16 else _next_pow2(group)
    block_q = max(1, block_m // group)
    num_tokens = case["q_len"] * case["batch"]
    q_blocks = num_tokens // block_q + case["batch"]
    programs = q_blocks * case["kv_heads"]
    if is_3d:
        programs *= case["segments"]
    seq_len = case["context"] + case["q_len"]
    tile = 16 if is_3d else 32
    if is_3d:
        tiles_per_program = math.ceil(
            seq_len / max(1, case["segments"] * tile))
    else:
        tiles_per_program = math.ceil(seq_len / tile)
    return {
        "group": group,
        "block_m": block_m,
        "block_q": block_q,
        "q_blocks": q_blocks,
        "programs": programs,
        "tiles_per_program": tiles_per_program,
    }


def _tensors(case: dict[str, Any], seed: int = 1234):
    import torch

    torch.manual_seed(seed)
    q_len = case["q_len"]
    batch = case["batch"]
    seq_len = case["context"] + q_len
    block_size = case["block_size"]
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = blocks_per_seq * batch
    q = torch.randn(
        batch * q_len, case["heads"], case["head_dim"],
        device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        total_blocks, block_size, case["kv_heads"], case["head_dim"],
        device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block_table = torch.arange(
        total_blocks, dtype=torch.int32, device="cuda").reshape(
        batch, blocks_per_seq)
    seq_lens = torch.full(
        (batch,), seq_len, dtype=torch.int32, device="cuda")
    cu = torch.arange(
        0, (batch + 1) * q_len, q_len,
        dtype=torch.int32, device="cuda")
    scale = case["head_dim"] ** -0.5
    window = (-1, -1) if case["window"] < 0 else (case["window"] - 1, 0)
    return q, k, v, block_table, seq_lens, cu, scale, window


def _make_workspace(case: dict[str, Any]):
    from minivllm.attention.ops.triton_unified_attention import (
        make_3d_workspace,
    )

    return make_3d_workspace(
        "cuda", case["heads"], case["head_dim"], 128, case["segments"])


def _prepare(case: dict[str, Any], seed: int = 1234) -> dict[str, Any]:
    """把 q/K/V、块表、workspace 全部在计时外准备好。"""
    import torch

    q, k, v, block_table, seq_lens, cu, scale, window = _tensors(case, seed)
    out2 = torch.empty_like(q)
    out3 = torch.empty_like(q)
    workspace = _make_workspace(case)
    torch.cuda.synchronize()
    return {
        "q": q, "k": k, "v": v, "block_table": block_table,
        "seq_lens": seq_lens, "cu": cu, "scale": scale,
        "window": window, "out2": out2, "out3": out3,
        "workspace": workspace,
    }


def _run(case: dict[str, Any], path: str, prepared: dict[str, Any]):
    from minivllm.attention.ops.triton_unified_attention import unified_attention

    q = prepared["q"]
    k = prepared["k"]
    v = prepared["v"]
    if path == "2d":
        out = prepared["out2"]
        unified_attention(
            q, k, v, out, prepared["cu"], prepared["seq_lens"],
            prepared["block_table"], softmax_scale=prepared["scale"],
            causal=True, window_size=prepared["window"],
            max_seqlen_q=case["q_len"])
        return out
    if path == "3d":
        out = prepared["out3"]
        workspace = prepared["workspace"]
        unified_attention(
            q, k, v, out, prepared["cu"], prepared["seq_lens"],
            prepared["block_table"], softmax_scale=prepared["scale"],
            causal=True, window_size=prepared["window"],
            max_seqlen_q=1, seq_threshold_3D=128,
            num_par_softmax_segments=case["segments"],
            softmax_segm_output=workspace[0],
            softmax_segm_max=workspace[1],
            softmax_segm_expsum=workspace[2])
        return out
    raise ValueError(f"未知 path={path!r}")

def _correctness(case: dict[str, Any],
                 prepared: dict[str, Any]) -> dict[str, Any]:
    import torch

    runs = []
    for seed in (1234, 1235, 1236):
        state = prepared if seed == 1234 else _prepare(case, seed)
        out2 = _run(case, "2d", state)
        out3 = _run(case, "3d", state)
        torch.cuda.synchronize()
        a = out2.float()
        b = out3.float()
        runs.append({
            "seed": seed,
            "max_abs": float((a - b).abs().max()),
            "mean_abs": float((a - b).abs().mean()),
            "output_2d": out2,
            "output_3d": out3,
        })
    max_abs = max(item["max_abs"] for item in runs)
    mean_abs = max(item["mean_abs"] for item in runs)
    exact = True
    error = None
    for item in runs:
        try:
            torch.testing.assert_close(
                item["output_2d"], item["output_3d"], **TOLERANCE)
        except AssertionError as exc:  # noqa: PERF203
            exact = False
            error = str(exc)[:1000]
            break
    return {"ok": exact, "max_abs": max_abs, "mean_abs": mean_abs,
            "seeds": [item["seed"] for item in runs], "error": error}


def _time_burst(fn, *, calls: int) -> float:
    """一次 event 包住多 call，除出单 call 时间；避免逐 call event 噪声。"""
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(calls):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / max(calls, 1)


def _bench_pair(case: dict[str, Any], prepared: dict[str, Any], *,
                warmup: int, repeats: int, rounds: int,
                flush_l2: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    calls = max(1, repeats // rounds)
    flush = None
    if flush_l2:
        # 32 MiB > 单层 1024-context 的 K/V 片段；把重复调用的 L2 命中压低。
        flush = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def measure(path: str) -> float:
        if flush is not None:
            flush.zero_()
            torch.cuda.synchronize()
        return _time_burst(lambda: _run(case, path, prepared), calls=calls)

    for _ in range(max(1, warmup // 10)):
        _run(case, "2d", prepared)
        _run(case, "3d", prepared)
    torch.cuda.synchronize()
    two_rounds: list[float] = []
    three_rounds: list[float] = []
    for index in range(rounds):
        # 交替先后顺序，减少时钟/热漂移只偏向一侧。
        if index % 2 == 0:
            two_rounds.append(measure("2d"))
            three_rounds.append(measure("3d"))
        else:
            three_rounds.append(measure("3d"))
            two_rounds.append(measure("2d"))
    two = {
        "median_ms": statistics.median(two_rounds),
        "min_ms": min(two_rounds),
        "max_ms": max(two_rounds),
        "values_ms": two_rounds,
        "calls_per_round": calls,
        "grid": _grid_info(case, is_3d=False),
    }
    three = {
        "median_ms": statistics.median(three_rounds),
        "min_ms": min(three_rounds),
        "max_ms": max(three_rounds),
        "values_ms": three_rounds,
        "calls_per_round": calls,
        "grid": _grid_info(case, is_3d=True),
    }
    return two, three

def _run_all(cases: list[dict[str, Any]], *, warmup: int,
             repeats: int, rounds: int,
             flush_l2: bool) -> list[dict[str, Any]]:
    results = []
    for index, case in enumerate(cases):
        prepared = _prepare(case)
        correctness = _correctness(case, prepared)
        two, three = _bench_pair(
            case, prepared, warmup=warmup, repeats=repeats,
            rounds=rounds, flush_l2=flush_l2)
        speedup = (two["median_ms"] / three["median_ms"]
                   if three["median_ms"] > 0 else None)
        result = {
            **case,
            "correctness": correctness,
            "2d": two,
            "3d_forced": three,
            "speedup_3d_over_2d": speedup,
        }
        results.append(result)
        print(
            f"[{index + 1:02d}/{len(cases)}] {case['label']:<28} "
            f"2D={two['median_ms']:.4f}ms "
            f"3D={three['median_ms']:.4f}ms "
            f"speedup={speedup:.3f} "
            f"correct={correctness['ok']}",
            flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--flush-l2", action="store_true")
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
    results = _run_all(
        cases, warmup=args.warmup, repeats=args.repeats,
        rounds=args.rounds, flush_l2=args.flush_l2)
    errors = [row for row in results if not row["correctness"]["ok"]]
    payload = {
        "ok": not errors,
        "metric_schema": "step88_attention_micro",
        "recorded_at": datetime.datetime.now().astimezone().isoformat(
            timespec="minutes"),
        "device": torch.cuda.get_device_name(0),
        "package": package_hash(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rounds": args.rounds,
        "flush_l2": bool(args.flush_l2),
        "tolerance": TOLERANCE,
        "cases": results,
        "failures": errors,
        "note": ("3D 在 q_len>1 上通过强制 max_seqlen_q=1 的 prototype 路径；"
                 "生产 `can_use_3d()` 仍拒绝 q_len>1，未改生产行为"),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    print(f"summary: {out_dir / 'summary.json'}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
