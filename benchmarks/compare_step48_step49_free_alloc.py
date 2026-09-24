"""第 49 关的定点对比：`step48` vs `step49`，负载形状与
`benchmarks/profile_step48_scheduler_kv.py` 保持一致。

只回答一个问题：**补块时找「真正空闲块」的开销是否还随池子大小线性增长。**
重点是 32 请求 / 8192 块 / prefix 关的 `schedule()` 累计时间，
以及 `_plan_block_growth(seq, 1)` 的耗时。

不做端到端吞吐断言：剖析已说明，在当前这个 Python/Torch 小模型里调度 CPU 开销
还不是端到端主瓶颈。CPU 负载不跑模型，只量调度与 KV *元数据*。
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(package: str):
    return (importlib.import_module(package),
            importlib.import_module(f"{package}.cache"),
            importlib.import_module(f"{package}.request"),
            importlib.import_module(f"{package}.scheduler"))


def median_call_us(fn, iterations: int, repeats: int = 7) -> float:
    samples = []
    for _ in range(repeats):
        gc.collect()
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        samples.append((time.perf_counter_ns() - begin) / iterations / 1_000)
    return statistics.median(samples)


def make_pool(mods, blocks: int, *, prefix: bool = False, block_size: int = 16):
    _, cache, _, _ = mods
    return cache.KVCachePool(block_size, blocks, 1, 1, torch.device("cpu"), prefix,
                             num_layers=1, over_subscribe=True)


def allocator_micro(mods, blocks_list=(256, 4096, 16384)) -> list[dict]:
    """只测 _plan_block_growth(seq, 1)：这是本关直接针对的热路径。"""
    _, _, request, _ = mods
    results = []
    for blocks in blocks_list:
        for occupancy in ("free", "idle_cached"):
            p = make_pool(mods, blocks, prefix=occupancy == "idle_cached")
            if occupancy == "idle_cached":
                for b in range(blocks):
                    key = b.to_bytes(8, "little")
                    p.hash_to_block[key] = b
                    p.block_to_hash[b] = key
                    p.block_last_used[b] = b
                if hasattr(p, "free_heap"):
                    # 全部带 hash -> 一个「真正空闲」块都没有。必须同步清空空闲堆，
                    # 否则 step49 会从脏堆里拿到块、绕过淘汰路径，测出来的不是真东西。
                    p.free_heap.clear()
            seq = request.SequenceConfig("probe", [1], 2, 16)
            iterations = 500 if blocks <= 4096 else 120
            measured = median_call_us(lambda: p._plan_block_growth(seq, 1), iterations)
            results.append({"pool_blocks": blocks, "occupancy": occupancy,
                            "plan_one_block_us": round(measured, 2)})
    return results


def synthetic_once(mods, num_reqs: int, blocks: int, prefix: bool) -> dict:
    """与 profile_step48_scheduler_kv.py 的 synthetic_once 同形状。"""
    _, _, _, scheduler_mod = mods
    Scheduler = scheduler_mod.Scheduler
    p = make_pool(mods, blocks, prefix=prefix)
    s = Scheduler(max_num_seqs=num_reqs, max_num_batched_tokens=num_reqs,
                  block_size=16, kv_cache_pool=p, vocab_size=128,
                  eos_token_ids={127}, enable_prefix_caching=prefix,
                  preemption_mode="recompute")
    seqs = []
    for i in range(num_reqs):
        prompt = [i % 120] + [((i * 7 + j) % 120) for j in range(1, 32)]
        s.add_request({"request_id": str(i), "prompt_ids": prompt, "max_new_tokens": 64})
        seqs.append(s.waiting[-1])

    schedule_ns, phase_ns = [], {"admit": 0, "plan": 0, "reserve": 0}
    for phase, name in (("admit", "_admit_waiting"), ("plan", "_plan_tokens"),
                        ("reserve", "_reserve_blocks")):
        original = getattr(s, name)

        def timed(*args, _fn=original, _phase=phase, **kwargs):
            start = time.perf_counter_ns()
            result = _fn(*args, **kwargs)
            phase_ns[_phase] += time.perf_counter_ns() - start
            return result

        setattr(s, name, timed)

    steps = 0
    while s.has_unfinished_requests():
        t0 = time.perf_counter_ns()
        s.schedule()
        schedule_ns.append(time.perf_counter_ns() - t0)
        for item in s.scheduled_items:          # 模拟「模型已经算完了安排的 token」
            seq = item["request"]
            seq.cache.length += item["num_scheduled_tokens"]
            if item["can_sample"]:
                seq.output_ids.append(7)
        s.post_step()
        steps += 1
        if steps > 2000:
            raise RuntimeError("synthetic workload did not drain")
    if any(p.block_usage) or p.promised_blocks:
        raise AssertionError("KV references or promises leaked")
    return {"requests": num_reqs, "pool_blocks": blocks, "prefix": prefix,
            "steps": steps, "preemptions": s.num_preemptions,
            "schedule_total_ms": round(sum(schedule_ns) / 1e6, 3),
            "schedule_p50_us": round(statistics.median(schedule_ns) / 1_000, 2),
            "phase_total_ms": {k: round(v / 1e6, 3) for k, v in phase_ns.items()},
            "reused_tokens": sum(seq.reused_tokens for seq in seqs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="benchmarks/results/step49_vs_step48_free_alloc.json")
    parser.add_argument("--packages", default="step48,step49")
    args = parser.parse_args()

    packages = args.packages.split(",")
    mods = {name: load(name) for name in packages}

    report = {"environment": {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "platform": platform.platform(), "cpu_count": None,
        "note": "CPU 合成负载，不跑模型；只量调度与 KV 元数据",
    }, "allocator": {}, "synthetic": {}}
    try:
        import os
        report["environment"]["cpu_count"] = os.cpu_count()
    except Exception:                                       # noqa: BLE001
        pass

    for name in packages:
        report["allocator"][name] = allocator_micro(mods[name])
        report["synthetic"][name] = {
            f"{r}_{b}_{p}": synthetic_once(mods[name], r, b, p)
            for r, b, p in ((32, 8192, False), (32, 1024, False), (32, 8192, True))
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(f"Python {report['environment']['python']} / torch {report['environment']['torch']}")
    print(f"原始数据: {out}\n")

    print("_plan_block_growth(seq, 1)  —— 每次补 1 块的中位耗时（μs）")
    print(f"{'池子块数':>10} {'占用形态':>12} " +
          " ".join(f"{n:>12}" for n in packages))
    for i, row in enumerate(report["allocator"][packages[0]]):
        cells = [f"{report['allocator'][n][i]['plan_one_block_us']:>12.2f}" for n in packages]
        print(f"{row['pool_blocks']:>10} {row['occupancy']:>12} " + " ".join(cells))

    print("\n合成完整负载：schedule() 累计耗时（ms）")
    print(f"{'负载':>22} " + " ".join(f"{n:>12}" for n in packages) + "   其中 _reserve_blocks")
    for key in report["synthetic"][packages[0]]:
        cells = [f"{report['synthetic'][n][key]['schedule_total_ms']:>12.3f}" for n in packages]
        res = [f"{report['synthetic'][n][key]['phase_total_ms']['reserve']:.2f}" for n in packages]
        print(f"{key:>22} " + " ".join(cells) + "   " + "/".join(res))

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
