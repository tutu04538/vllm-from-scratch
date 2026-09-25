"""Profile eviction with a naturally warmed, fully idle prefix cache (step49 vs step50).

No model forward: KV metadata transitions use the real pool methods. The script
does not modify step48/step49 implementations. It compares identical workloads.
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


def case(package: str, blocks: int, new_requests: int = 300) -> dict:
    cache = importlib.import_module(f"{package}.cache")
    request = importlib.import_module(f"{package}.request")
    pool = cache.KVCachePool(4, blocks, 1, 1, torch.device("cpu"),
                             enable_prefix_caching=True, num_layers=1,
                             over_subscribe=True)

    def complete_one(i: int, *, timed: bool) -> int | None:
        seq = request.SequenceConfig(i, [i + 1, 11, 12, 13], 1, 4)
        assert pool.allocate_block(seq)
        start = time.perf_counter_ns()
        assert pool.ensure_blocks(seq, 4)
        elapsed = time.perf_counter_ns() - start
        seq.cache.length = 4  # simulate successful model forward
        pool.publish_computed_blocks(seq)
        pool.deallocate_block(seq)
        return elapsed if timed else None

    # Every block is reached by the normal admission → growth → publish → release
    # path. There are no manual edits to hash maps or free lists.
    for i in range(blocks):
        complete_one(i, timed=False)
    assert len(pool.hash_to_block) == blocks
    assert not any(pool.block_usage)
    # step49 的空闲块结构；step48 没有（那里是每次扫全池算出来的）
    if hasattr(pool, "free_queue"):
        assert not pool.free_queue

    gc.collect()
    measured = [complete_one(blocks + i, timed=True) for i in range(new_requests)]
    if hasattr(pool, "free_queue"):
        assert set(pool.free_queue) == set(pool._free_block_indices())
    assert len(pool.hash_to_block) == blocks and not any(pool.block_usage)
    return {"package": package, "blocks": blocks, "new_requests": new_requests,
            "ensure_p50_us": round(statistics.median(measured) / 1000, 2),
            "ensure_p95_us": round(sorted(measured)[int(.95 * (len(measured) - 1))] / 1000, 2)}


def main() -> None:
    torch.set_num_threads(1)
    results = [case(pkg, n) for _ in range(3) for n in (1024, 8192)
               for pkg in ("step49", "step50")]
    path = Path("benchmarks/results/step50_warm_prefix_profile.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    for n in (1024, 8192):
        for pkg in ("step49", "step50"):
            vals = [r["ensure_p50_us"] for r in results
                    if r["blocks"] == n and r["package"] == pkg]
            print(f"{pkg} {n} blocks: median p50={statistics.median(vals):.2f} us; "
                  f"run range={min(vals):.2f}-{max(vals):.2f} us")
    print("Wrote", path)


if __name__ == "__main__":
    main()
