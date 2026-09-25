"""Measure scheduler/prefix-publish CPU cost as generated history grows.

No model forward or real KV writes. All requests are decode-ready; publication
has no newly completed blocks, so its work should be nearly constant.
"""

from __future__ import annotations

import gc
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def median_us(fn, iterations: int) -> float:
    samples = []
    for _ in range(7):
        gc.collect()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        samples.append((time.perf_counter_ns() - start) / iterations / 1000)
    return round(statistics.median(samples), 2)


def measure(package: str, output_len: int, running: int = 16) -> dict:
    cache_mod = importlib.import_module(f"{package}.cache")
    req_mod = importlib.import_module(f"{package}.request")
    sched_mod = importlib.import_module(f"{package}.scheduler")
    pool = cache_mod.KVCachePool(16, 1024, 1, 1, torch.device("cpu"),
                                  enable_prefix_caching=True, num_layers=1,
                                  over_subscribe=True)
    scheduler = sched_mod.Scheduler(max_num_seqs=running,
                                    max_num_batched_tokens=running, block_size=16,
                                    kv_cache_pool=pool, vocab_size=128,
                                    enable_prefix_caching=True,
                                    preemption_mode="recompute")
    for i in range(running):
        seq = req_mod.SequenceConfig(i, list(range(32)), output_len + 100, 16)
        if hasattr(seq, "append_output_ids"):
            # Step 51+ keeps committed history and output in sync through one writer.
            seq.append_output_ids([7] * output_len)
        else:
            seq.output_ids = [7] * output_len
        seq.cache.length = 32 + output_len - 1  # only last token needs decode
        seq.block_hashes = [b"x"] * (seq.cache.length // 16)  # no new full block
        scheduler.running.append(seq)
    iterations = 3000 if output_len <= 128 else 500 if output_len <= 2048 else 150
    last = scheduler.running[-1]
    return {"package": package, "running": running, "output_len": output_len,
            "plan_us": median_us(scheduler._plan_tokens, iterations),
            "publish_us": median_us(scheduler._publish_computed_blocks, iterations),
            "one_full_history_slice_us": median_us(lambda: last.all_token_ids[-1:],
                                                    iterations * 2),
            "one_output_tail_slice_us": median_us(lambda: last.output_ids[-1:],
                                                  iterations * 2)}


def main() -> None:
    torch.set_num_threads(1)
    packages = sys.argv[1:] or ["step50"]
    rows = [measure(package, length) for package in packages
            for length in (1, 128, 2048, 8192)]
    path = Path("benchmarks/results/step50_long_history_profile.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row)
    print("Wrote", path)


if __name__ == "__main__":
    main()
