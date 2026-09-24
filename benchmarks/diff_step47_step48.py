"""逐步对比 step47 与 step48：证明第四十八关只是行为不变的重构。

同一模型 seed、同一请求、同一动态到达时刻，逐步比较：
  running / waiting 顺序、本轮计划（请求、token 数、can_sample）、
  完成顺序与输出、每请求计数、承诺总额、块引用与双向 hash。

CPU / FP32 / 确定性小模型。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step47 import Engine as Engine47
from step48 import Engine as Engine48

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
               enable_prefix_caching=False, preemption_mode="recompute",
               scheduling_policy="fcfs")
    cfg.update(kw)
    return cls(device="cpu", **cfg, **DIMS)


def snapshot(e):
    s = e.scheduler
    return {
        "running": [q.request_id for q in s.running],
        "waiting": [q.request_id for q in s.waiting],
        "plan": [(it["request"].request_id, it["num_scheduled_tokens"], it["can_sample"])
                 for it in s.scheduled_items],
        "input_ids": [list(it["input_ids"]) for it in s.scheduled_items],
        "done": [(r["request_id"], list(r["output_ids"]), r.get("error"))
                 for r in s.step_done],
        "cache_len": {q.request_id: q.cache.length
                      for q in s.running + s.waiting if q.cache is not None},
    }


def run_pair(seed, arrival_plan, limit=800, **kw):
    """arrival_plan: {step_index: [request, ...]}，step_index 从 0 开始，先提交再 step。"""
    engines = [build(Engine47, seed, **kw), build(Engine48, seed, **kw)]
    for e in engines:
        e.on_token = lambda ev: None
        e.scheduler.on_finished = lambda rec: None
    traces = [[], []]
    counters = [{}, {}]
    step = 0
    for e, tr in zip(engines, traces):
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
                    "reu": max(prev.get("reu", 0), getattr(q, "reused_tokens", 0)),
                    "hw": max(prev.get("hw", 0), q.high_water),
                }
            for r in arrival_plan.get(step, []):
                e.add_request(dict(r))
    return engines, traces, counters


def _hash_map(pool):
    """step48 把 block_hash 更名为 hash_to_block；两个版本都能取。"""
    return pool.hash_to_block if hasattr(pool, "hash_to_block") else pool.block_hash


def summarize(e):
    s = e.scheduler
    # 完成顺序从逐步快照里拿不到（step_done 每轮清空），改用 on_finished 的顺序
    return {
        "promised_pool": s.kv_cache_pool.promised_blocks,
        "usage_zero": all(u == 0 for u in s.kv_cache_pool.block_usage),
        # step48 把 block_hash 更名为 hash_to_block，两边名字不同，按存在与否取
        "hash_consistent": all(s.kv_cache_pool.block_to_hash.get(b) == h
                               for h, b in _hash_map(s.kv_cache_pool).items())
        and len(s.kv_cache_pool.block_to_hash) == len(_hash_map(s.kv_cache_pool)),
        "running_empty": not s.running and not s.waiting,
        "num_preemptions": s.num_preemptions,
        "num_priority_preemptions": s.num_priority_preemptions,
    }


REQS = [
    {"request_id": "A", "prompt_ids": [1, 2, 3, 4, 5, 6], "max_new_tokens": 8, "priority": 0},
    {"request_id": "B", "prompt_ids": [7, 8, 9, 10, 11, 12], "max_new_tokens": 8, "priority": 0},
    {"request_id": "C", "prompt_ids": [13, 14, 15, 16, 17, 18], "max_new_tokens": 8, "priority": 5},
]
HIGH = {"request_id": "H", "prompt_ids": [21, 22, 23, 24], "max_new_tokens": 6, "priority": -1}

SCENARIOS = [
    ("fcfs / 三条同批到达", dict(scheduling_policy="fcfs", num_kv_blocks=8), {0: REQS}),
    ("priority / 三条同批到达", dict(scheduling_policy="priority", num_kv_blocks=8), {0: REQS}),
    ("priority / 高优先级后到", dict(scheduling_policy="priority", num_kv_blocks=8),
     {0: REQS, 3: [HIGH]}),
    ("fcfs / 容量压力（4 块）", dict(scheduling_policy="fcfs", num_kv_blocks=4), {0: REQS}),
    ("priority / 容量压力（4 块）", dict(scheduling_policy="priority", num_kv_blocks=4), {0: REQS}),
    ("priority / prefix 开 + 容量压力", dict(scheduling_policy="priority", num_kv_blocks=5,
                                            enable_prefix_caching=True), {0: REQS}),
    ("fcfs / prefix 开 + 动态到达", dict(scheduling_policy="fcfs", num_kv_blocks=6,
                                        enable_prefix_caching=True), {0: REQS[:2], 4: [REQS[2]]}),
    ("priority / budget=1", dict(scheduling_policy="priority", max_num_batched_tokens=1,
                                 num_kv_blocks=16), {0: REQS, 2: [HIGH]}),
    ("priority / max_num_seqs=1", dict(scheduling_policy="priority", max_num_seqs=1,
                                       num_kv_blocks=8), {0: REQS, 3: [HIGH]}),
]

for label, kw, plan in SCENARIOS:
    for seed in (29, 7):
        e47, e48 = None, None
        engines, traces, counters = run_pair(seed, plan, **kw)
        t47, t48 = traces
        diff_steps = [i for i, (a, b) in enumerate(zip(t47, t48), 1) if a != b]
        check(f"{label}（seed={seed}）：逐步计划/队列完全一致",
              not diff_steps and len(t47) == len(t48),
              "" if not diff_steps else f"第 {diff_steps[:3]} 步不同")
        s47 = summarize(engines[0])
        s48 = summarize(engines[1])
        check(f"{label}（seed={seed}）：每请求计数一致", counters[0] == counters[1],
              "" if counters[0] == counters[1] else f"\n  47={counters[0]}\n  48={counters[1]}")
        check(f"{label}（seed={seed}）：结束时计数与资源一致", s47 == s48,
              "" if s47 == s48 else f"\n  step47={s47}\n  step48={s48}")
        check(f"{label}（seed={seed}）：承诺归零 / 引用归零 / hash 双向一致",
              s48["promised_pool"] == 0 and s48["usage_zero"] and s48["hash_consistent"]
              and s48["running_empty"], str(s48))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
