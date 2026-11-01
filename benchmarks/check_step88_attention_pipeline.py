"""88 后续诊断：同一 2D attention 输入下，mini 与 vLLM 的 pipeline 差异。

重点不是“谁快多少”，而是回答：
  * vLLM 的 Triton kernel 是否把 block-table/K/V load 变成
    `ttg.async_copy_global_to_local`（cp.async）；
  * mini 的本地 kernel 是否只有 `tt.load` + 后面的 `ttg.local_alloc`；
  * 两边的 registers/spills/shared 和实际单 kernel 时间。

不加载 engine，不修改生产路径。结果只写到 gitignored 的 benchmarks/results/。
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
    _prepare, _run, case_matrix,
)
from step88_common import package_hash  # noqa: E402

RESULT_DIR = ROOT / "benchmarks" / "results" / "step88" / "attention_pipeline"


def _compiled_meta(module) -> dict[str, Any]:
    cache = module.kernel_unified_attention.device_caches[0][0]
    entry = next(iter(cache.values()))
    meta = entry.metadata
    ttgir = entry.asm.get("ttgir", "")
    ptx = entry.asm.get("ptx", "")
    return {
        "n_regs": int(entry.n_regs),
        "n_spills": int(entry.n_spills),
        "shared_bytes": int(meta.shared or 0),
        "num_warps": int(meta.num_warps or 0),
        "num_stages": int(meta.num_stages or 0),
        "ttgir_async_copy": ttgir.count("async_copy"),
        "ttgir_async_commit": ttgir.count("async_commit"),
        "ttgir_async_wait": ttgir.count("async_wait"),
        "ttgir_local_alloc": ttgir.count("local_alloc"),
        "ptx_cp_async": ptx.count("cp.async"),
        "ptx_ldmatrix": ptx.count("ldmatrix"),
        "ptx_st_shared": ptx.count("st.shared"),
    }


def _vllm_run(case: dict[str, Any], prepared: dict[str, Any]):
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention as vllm_unified_attention,
    )

    out = prepared["out_vllm"]
    seq_len = case["context"] + case["q_len"]
    vllm_unified_attention(
        prepared["q"], prepared["k"], prepared["v"], out,
        prepared["cu"], case["q_len"], prepared["seq_lens"], seq_len,
        prepared["scale"], True, prepared["window"],
        prepared["block_table"], 0.0, None, None, None)
    return out


def _bench(fn, *, warmup: int, repeats: int, rounds: int) -> float:
    for _ in range(max(1, warmup // 10)):
        fn()
    vals = []
    import torch
    torch.cuda.synchronize()
    calls = max(1, repeats // rounds)
    for _ in range(rounds):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        end.synchronize()
        vals.append(start.elapsed_time(end) / calls)
    return statistics.median(vals)


def run(case_label: str, *, warmup: int, repeats: int,
        rounds: int) -> dict[str, Any]:
    import torch
    from minivllm.attention.ops import triton_unified_attention as mini_module
    from vllm.v1.attention.ops import triton_unified_attention as vllm_module

    case = next(c for c in case_matrix() if c["label"] == case_label)
    prepared = _prepare(case)
    mini_out = _run(case, "2d", prepared).clone()
    vllm_out = _vllm_run(case, prepared).clone()
    torch.cuda.synchronize()
    diff = (mini_out.float() - vllm_out.float()).abs().max().item()
    mini_meta = _compiled_meta(mini_module)
    vllm_meta = _compiled_meta(vllm_module)
    mini_ms = _bench(
        lambda: _run(case, "2d", prepared),
        warmup=warmup, repeats=repeats, rounds=rounds)
    vllm_ms = _bench(
        lambda: _vllm_run(case, prepared),
        warmup=warmup, repeats=repeats, rounds=rounds)
    return {
        "case": case,
        "max_abs": float(diff),
        "mini_ms": mini_ms,
        "vllm_ms": vllm_ms,
        "mini_vs_vllm": mini_ms / vllm_ms,
        "mini": mini_meta,
        "vllm": vllm_meta,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="qlen3_ctx1024")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"ok": True, "case": args.case,
                          "cases": [c["label"] for c in case_matrix()]},
                         indent=1))
        return 0
    import torch
    if not torch.cuda.is_available():
        print("需要 CUDA", file=sys.stderr)
        return 2
    row = run(args.case, warmup=args.warmup, repeats=args.repeats,
              rounds=args.rounds)
    payload = {
        "ok": True,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(
            timespec="minutes"),
        "device": torch.cuda.get_device_name(0),
        "package": package_hash(),
        "warmup": args.warmup, "repeats": args.repeats,
        "rounds": args.rounds,
        "row": row,
        "note": ("TTGIR/PTX 计数来自各自 Triton CompiledKernel；"
                 "max_abs 是 mini/vLLM 同输入输出差。"),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(row, ensure_ascii=False, indent=1, default=str))
    print(f"summary: {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
