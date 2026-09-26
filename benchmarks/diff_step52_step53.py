"""投机关关闭时，step53 与 step52 应当逐步一致。

第五十三关动的是投机路径（批量行映射、提交顺序、预算分配、缩草稿）和一条入口
校验，**没动**不投机的路径。所以 `speculative_mode=None` 下两者必须逐字节一致
——包括物理块编号、hash 链、完成顺序与每个请求的计数。这里不放宽任何字段。

（第五十二关那份对照是 step51 的 recompute 模式 vs step52：当时 step52 刚删掉
承诺账本，默认容量策略与 step51 的默认不同。本关起直接比相邻两关，因为容量策略
已经统一成「按需分配 + 无条件抢占 + 不锁未来额度」。）

CPU / FP32 / 确定性小模型。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step52 import Engine as Engine52
from step53 import Engine as Engine53

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


def build(cls, seed, **kw):
    torch.manual_seed(seed)
    cfg = dict(max_num_seqs=2, max_num_batched_tokens=8, block_size=4, num_kv_blocks=8,
               enable_prefix_caching=False, scheduling_policy="fcfs")
    cfg.update(kw)
    return cls(device="cpu", **cfg, **DIMS)


def snapshot(e):
    s = e.scheduler
    seqs = s.running + s.waiting
    return {
        "running": [q.request_id for q in s.running],
        "waiting": [q.request_id for q in s.waiting],
        "plan": [(it["request"].request_id, it["num_scheduled_tokens"], it["can_sample"],
                  list(it.get("draft_ids", [])))
                 for it in s.scheduled_items],
        "input_ids": [list(it["input_ids"]) for it in s.scheduled_items],
        "done": [(r["request_id"], list(r["output_ids"]), r.get("error"))
                 for r in s.step_done],
        "cache_len": {q.request_id: q.cache.length for q in seqs if q.cache is not None},
        "block_usage": list(e.kv_cache_pool.block_usage),
        "block_table": {q.request_id: (list(q.cache.block_table)
                                       if q.cache is not None and q.cache.block_table is not None
                                       else None)
                        for q in seqs},
        "block_hashes": {q.request_id: list(q.block_hashes) for q in seqs},
        "all_ids": {q.request_id: list(q.all_token_ids) for q in seqs},
        "output_ids": {q.request_id: list(q.output_ids) for q in seqs},
    }


def run_pair(seed, arrival_plan, limit=800, **kw):
    engines = [build(Engine52, seed, **kw), build(Engine53, seed, **kw)]
    for e in engines:
        e.on_token = lambda ev: None
    traces = [[], []]
    counters = [{}, {}]
    step = 0
    for i, e in enumerate(engines):
        for r in arrival_plan.get(0, []):
            e.add_request(dict(r))
    while any(e.has_unfinished_requests() for e in engines):
        step += 1
        if step > limit:
            raise RuntimeError("疑似活锁")
        for i, (e, tr) in enumerate(zip(engines, traces)):
            e.step()
            tr.append(snapshot(e))
            for q in e.scheduler.running + e.scheduler.waiting:
                prev = counters[i].get(q.request_id, {})
                counters[i][q.request_id] = {
                    "pre": max(prev.get("pre", 0), q.num_preemptions),
                    "rec": max(prev.get("rec", 0), q.recomputed_tokens),
                    "reu": max(prev.get("reu", 0), q.reused_tokens),
                    "hw": max(prev.get("hw", 0), q.high_water),
                }
            for r in arrival_plan.get(step, []):
                e.add_request(dict(r))
    return engines, traces, counters


def _alloc_chain(pool):
    """把可分配链表按顺序读出来。"""
    if not hasattr(pool, "block_next"):
        return []
    out, cur = [], pool.block_next[pool._SENTINEL_HEAD]
    while cur != pool._SENTINEL_TAIL:
        out.append(cur)
        cur = pool.block_next[cur]
    return out


def summarize(e):
    s = e.scheduler
    return {

        "usage_zero": all(u == 0 for u in s.kv_cache_pool.block_usage),
        "hash_consistent": all(s.kv_cache_pool.block_to_hash.get(b) == h
                               for h, b in s.kv_cache_pool.hash_to_block.items())
        and len(s.kv_cache_pool.block_to_hash) == len(s.kv_cache_pool.hash_to_block),
        "running_empty": not s.running and not s.waiting,
        "num_preemptions": s.num_preemptions,
        "num_priority_preemptions": s.num_priority_preemptions,
        "cached_blocks_kept": len(s.kv_cache_pool.hash_to_block),
    }


REQS = [
    {"request_id": "A", "prompt_ids": [1, 2, 3, 4, 5, 6], "max_new_tokens": 8, "priority": 0},
    {"request_id": "B", "prompt_ids": [7, 8, 9, 10, 11, 12], "max_new_tokens": 8, "priority": 0},
    {"request_id": "C", "prompt_ids": [13, 14, 15, 16, 17, 18], "max_new_tokens": 8, "priority": 5},
]
HIGH = {"request_id": "H", "prompt_ids": [21, 22, 23, 24], "max_new_tokens": 6, "priority": -1}

SCENARIOS = [
    ("fcfs / prefix 关", dict(scheduling_policy="fcfs", num_kv_blocks=8), {0: REQS}),
    ("priority / prefix 关", dict(scheduling_policy="priority", num_kv_blocks=8), {0: REQS}),
    ("priority / 高优先级后到", dict(scheduling_policy="priority", num_kv_blocks=8),
     {0: REQS, 3: [HIGH]}),
    ("fcfs / 容量压力（4 块）", dict(scheduling_policy="fcfs", num_kv_blocks=4), {0: REQS}),
    ("priority / 容量压力（4 块）", dict(scheduling_policy="priority", num_kv_blocks=4), {0: REQS}),
    ("priority / prefix 开 + 容量压力", dict(scheduling_policy="priority", num_kv_blocks=5,
                                            enable_prefix_caching=True), {0: REQS}),
    ("fcfs / prefix 开 + 动态到达", dict(scheduling_policy="fcfs", num_kv_blocks=6,
                                        enable_prefix_caching=True), {0: REQS[:2], 4: [REQS[2]]}),
    ("priority / prefix 开 + 共享前缀", dict(scheduling_policy="priority", num_kv_blocks=6,
                                            enable_prefix_caching=True),
     {0: [{"request_id": "P", "prompt_ids": [1, 2, 3, 4, 5, 6, 7, 8], "max_new_tokens": 4},
          {"request_id": "Q", "prompt_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9], "max_new_tokens": 4}]}),
    ("priority / budget=1", dict(scheduling_policy="priority", max_num_batched_tokens=1,
                                 num_kv_blocks=16), {0: REQS, 2: [HIGH]}),
    ("priority / max_num_seqs=1", dict(scheduling_policy="priority", max_num_seqs=1,
                                       num_kv_blocks=8), {0: REQS, 3: [HIGH]}),
    ("fcfs / prefix 开 + 抢占", dict(num_kv_blocks=5, enable_prefix_caching=True), {0: REQS}),
]

for label, kw, plan in SCENARIOS:
    for seed in (29, 7):
        engines, traces, counters = run_pair(seed, plan, **kw)
        t51, t52 = traces
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(t51, t52), 1) if a != b]
        check(f"{label}（seed={seed}）：逐步计划 / 队列 / 输出 / 块表完全一致",
              not diff and len(t51) == len(t52),
              "" if not diff else f"第 {diff[0][0]} 步不同："
                                  f"{ {k for k in diff[0][1] if diff[0][1][k] != diff[0][2][k]} }")
        check(f"{label}（seed={seed}）：每请求计数一致", counters[0] == counters[1],
              "" if counters[0] == counters[1] else f"\n  51={counters[0]}\n  52={counters[1]}")
        s51, s52 = summarize(engines[0]), summarize(engines[1])
        check(f"{label}（seed={seed}）：结束时状态一致", s51 == s52,
              "" if s51 == s52 else f"\n  51={s51}\n  52={s52}")
        check(f"{label}（seed={seed}）：引用归零、可分配链与真实空闲一致",
              s52["usage_zero"] and s52["hash_consistent"]
              and s52["running_empty"]
              and set(_alloc_chain(engines[1].kv_cache_pool))
              == set(engines[1].kv_cache_pool._allocatable_block_indices()), str(s52))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
