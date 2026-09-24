"""Targeted Step 48 scheduler/KV profile; no model in the synthetic cases.

Run with the project environment's Python. Writes a machine-readable JSON record.
The synthetic loop advances cache.length but does not compute real K/V: it measures
CPU scheduling/bookkeeping, not generation correctness or model throughput.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import json
import platform
import pstats
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step48 import Engine
from step48.cache import KVCachePool
from step48.request import SequenceConfig
from step48.scheduler import Scheduler


def pool(blocks: int, *, prefix: bool = False, block_size: int = 16) -> KVCachePool:
    return KVCachePool(block_size, blocks, 1, 1, torch.device("cpu"), prefix,
                       num_layers=1, over_subscribe=True)


def median_call_us(fn, iterations: int, repeats: int = 7) -> float:
    samples = []
    for _ in range(repeats):
        gc.collect()
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        samples.append((time.perf_counter_ns() - begin) / iterations / 1_000)
    return statistics.median(samples)


def history_copy_micro() -> list[dict]:
    results = []
    for num_reqs in (1, 16):
        for output_len in (1, 128, 2048, 8192):
            p = pool(1)
            s = Scheduler(max_num_seqs=num_reqs, max_num_batched_tokens=num_reqs,
                          block_size=16, kv_cache_pool=p, vocab_size=128,
                          preemption_mode="recompute")
            for i in range(num_reqs):
                seq = SequenceConfig(i, list(range(32)), output_len + 10, 16)
                seq.output_ids = [7] * output_len
                seq.cache.length = 32 + output_len - 1  # exactly one pending token
                s.running.append(seq)
            iterations = 3000 if output_len <= 128 else 600 if output_len <= 2048 else 150
            actual = median_call_us(s._plan_tokens, iterations)
            # Isolate the avoidable full-list copy, without changing project code.
            last = s.running[-1]
            full_copy = median_call_us(lambda: last.all_token_ids[-1:], iterations * 2)
            direct_slice = median_call_us(lambda: last.output_ids[-1:], iterations * 2)
            results.append({"running": num_reqs, "output_len": output_len,
                            "plan_tokens_us": round(actual, 2),
                            "full_history_last_us": round(full_copy, 2),
                            "output_last_us": round(direct_slice, 2)})
    return results


def allocator_micro() -> list[dict]:
    results = []
    for blocks in (256, 4096, 16384):
        for occupancy in ("free", "idle_cached"):
            p = pool(blocks, prefix=occupancy == "idle_cached")
            if occupancy == "idle_cached":
                for b in range(blocks):
                    key = b.to_bytes(8, "little")
                    p.hash_to_block[key] = b
                    p.block_to_hash[b] = key
                    p.block_last_used[b] = b
            seq = SequenceConfig("probe", [1], 2, 16)
            iterations = 500 if blocks <= 4096 else 120
            measured = median_call_us(lambda: p._plan_block_growth(seq, 1), iterations)
            results.append({"pool_blocks": blocks, "occupancy": occupancy,
                            "plan_one_block_us": round(measured, 2)})
    return results


def synthetic_once(num_reqs: int, blocks: int, prefix: bool) -> dict:
    p = pool(blocks, prefix=prefix)
    s = Scheduler(max_num_seqs=num_reqs, max_num_batched_tokens=num_reqs,
                  block_size=16, kv_cache_pool=p, vocab_size=128,
                  eos_token_ids={127}, enable_prefix_caching=prefix,
                  preemption_mode="recompute")
    seqs = []
    for i in range(num_reqs):
        # Distinct first blocks: no accidental inter-request prefix hit.
        prompt = [i % 120] + [((i * 7 + j) % 120) for j in range(1, 32)]
        s.add_request({"request_id": str(i), "prompt_ids": prompt,
                       "max_new_tokens": 64})
        seqs.append(s.waiting[-1])

    schedule_ns, post_ns, phase_ns = [], [], {"admit": 0, "plan": 0, "reserve": 0}
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
        for item in s.scheduled_items:
            seq = item["request"]
            seq.cache.length += item["num_scheduled_tokens"]
            if item["can_sample"]:
                seq.output_ids.append(7)
        t0 = time.perf_counter_ns()
        s.post_step()
        post_ns.append(time.perf_counter_ns() - t0)
        steps += 1
        if steps > 2000:
            raise RuntimeError("synthetic workload did not drain")
    if any(p.block_usage) or p.promised_blocks:
        raise AssertionError("KV references or promises leaked")
    if any(len(seq.output_ids) != 64 for seq in seqs):
        raise AssertionError("unexpected output length")
    return {"requests": num_reqs, "pool_blocks": blocks, "prefix": prefix,
            "steps": steps, "preemptions": s.num_preemptions,
            "schedule_p50_us": round(statistics.median(schedule_ns) / 1_000, 2),
            "schedule_p95_us": round(sorted(schedule_ns)[int(.95 * (steps - 1))] / 1_000, 2),
            "schedule_total_ms": round(sum(schedule_ns) / 1e6, 2),
            "post_total_ms": round(sum(post_ns) / 1e6, 2),
            "phase_total_ms": {k: round(v / 1e6, 2) for k, v in phase_ns.items()},
            "recomputed_tokens": sum(seq.recomputed_tokens for seq in seqs),
            "reused_tokens": sum(seq.reused_tokens for seq in seqs)}


def synthetic_cases() -> list[dict]:
    cases = [(8, 1024, False), (32, 1024, False), (64, 1024, False),
             (32, 8192, False), (32, 8192, True), (16, 32, False),
             (16, 32, True)]
    results = []
    for reqs, blocks, prefix in cases:
        samples = [synthetic_once(reqs, blocks, prefix) for _ in range(3)]
        middle = sorted(samples, key=lambda x: x["schedule_total_ms"])[1]
        results.append(middle)
    return results


def cprofile_case() -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    synthetic_once(32, 8192, True)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(24)
    return stream.getvalue()


def gpu_engine_case() -> list[dict] | None:
    if not torch.cuda.is_available():
        return None
    results = []
    for blocks in (128, 8192):
        samples = []
        for rep in range(3):
            torch.manual_seed(100 + rep)
            e = Engine(device="cuda", vocab_size=128, d_model=32, max_seq_len=128,
                       num_q_heads=4, num_kv_heads=2, num_layers=2,
                       intermediate_size=64, head_dim=8, eos_token_ids=[127],
                       max_num_seqs=8, max_num_batched_tokens=16, block_size=8,
                       num_kv_blocks=blocks, enable_prefix_caching=True,
                       preemption_mode="recompute", scheduling_policy="fcfs")
            for i in range(8):
                e.add_request({"request_id": str(i),
                               "prompt_ids": [1 + i] + list(range(12, 27)),
                               "max_new_tokens": 32})
            schedule_ns = 0
            original = e.scheduler.schedule

            def timed_schedule():
                nonlocal schedule_ns
                begin = time.perf_counter_ns()
                result = original()
                schedule_ns += time.perf_counter_ns() - begin
                return result

            e.scheduler.schedule = timed_schedule
            steps, t0 = 0, time.perf_counter_ns()
            while e.has_unfinished_requests():
                e.step()
                torch.cuda.synchronize()
                steps += 1
                if steps > 200:
                    raise RuntimeError("GPU workload did not drain")
            total_ns = time.perf_counter_ns() - t0
            if any(e.kv_cache_pool.block_usage):
                raise AssertionError("GPU KV references leaked")
            samples.append({"pool_blocks": blocks, "steps": steps,
                            "total_ms": round(total_ns / 1e6, 2),
                            "schedule_total_ms": round(schedule_ns / 1e6, 2),
                            "schedule_fraction": round(schedule_ns / total_ns, 4)})
        results.append(sorted(samples, key=lambda x: x["total_ms"])[1])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/results/step48_scheduler_kv_profile.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    record = {
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cpu": platform.processor(),
                        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                            text=True).strip(),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "device_note": "synthetic: CPU bookkeeping only; GPU: synchronized Torch eager"},
        "history_copy_micro": history_copy_micro(),
        "allocator_micro": allocator_micro(),
        "synthetic_workloads": synthetic_cases(),
        "cprofile": cprofile_case(),
        "gpu_engine": gpu_engine_case(),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n")
    print("Wrote", path)
    for key in ("history_copy_micro", "allocator_micro", "synthetic_workloads", "gpu_engine"):
        print(key, json.dumps(record[key], ensure_ascii=False))


if __name__ == "__main__":
    main()
