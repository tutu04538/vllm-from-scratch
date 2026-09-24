"""第 49 关：空闲队列的增量维护与「计划失败无副作用」（需求 §不变量 / §验收 1、2）。

两件事：
  1. 每一次**状态转换之后**都校验 set(free_queue) 等于独立扫描出来的真正空闲集合，
     且堆内无重复；
  2. 从 step48 移植「准入失败 / 补块失败 / 计划阶段不淘汰」的无副作用测试，
     并把 free_queue 加进快照。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step49 import Engine
from step49.cache import InfeasibleRequest, KVCachePool
from step49.request import SequenceConfig

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# ------------------------------- 每次状态转换后校验不变量

VIOLATIONS = []


def assert_invariant(pool, where):
    truth = set(pool._free_block_indices())          # 独立全池扫描，不用堆自己校验自己
    heap_set = set(pool.free_queue)
    if heap_set != truth:
        VIOLATIONS.append({"where": where, "heap": sorted(heap_set), "truth": sorted(truth)})
    if len(pool.free_queue) != len(heap_set):
        VIOLATIONS.append({"where": where, "why": "堆内重复", "heap": list(pool.free_queue)})


IN_COMMIT = {"depth": 0}
EVICT_OUTSIDE_COMMIT = []


def instrument(pool):
    """包住所有会改块状态的操作，**操作完成后**立刻校验不变量。

    `_evict_block` 是 `_commit_block_growth()` 的内部步骤，被排除在外：
    堆和 block_usage 是两个结构，任何中间时刻都不自洽；而且被淘汰的闲置块
    在同一次提交里立刻被复用，"淘汰后就是真正空闲" 从来不是一个可观测状态。
    为了不让这个豁免变成借口，下面会断言它**永远只在提交内部被调用**。
    """
    def wrap(name, check_after=True):
        orig = getattr(pool, name)

        def wrapper(*a, **kw):
            if name == "_commit_block_growth":
                IN_COMMIT["depth"] += 1
            try:
                if name == "_evict_block" and not IN_COMMIT["depth"]:
                    EVICT_OUTSIDE_COMMIT.append(True)
                out = orig(*a, **kw)
            finally:
                if name == "_commit_block_growth":
                    IN_COMMIT["depth"] -= 1
            if check_after:
                assert_invariant(pool, name)
            return out
        setattr(pool, name, wrapper)

    for name in ("allocate_block", "ensure_blocks", "deallocate_block",
                 "publish_computed_blocks", "_commit_admission", "_commit_block_growth"):
        wrap(name)
    wrap("_evict_block", check_after=False)


def make_pool(*, blocks=8, block_size=4, prefix=True, over_subscribe=True):
    pool = KVCachePool(block_size, blocks, 2, 4, torch.device("cpu"),
                       enable_prefix_caching=prefix, num_layers=1, dtype=torch.float32,
                       over_subscribe=over_subscribe)
    instrument(pool)
    return pool


DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


def make_engine(seed=29, **kw):
    torch.manual_seed(seed)
    cfg = dict(max_num_seqs=2, max_num_batched_tokens=8, block_size=4, num_kv_blocks=8,
               enable_prefix_caching=False, preemption_mode="recompute",
               scheduling_policy="fcfs")
    cfg.update(kw)
    e = Engine(device="cpu", **cfg, **DIMS)
    instrument(e.kv_cache_pool)
    return e


def req(rid, ids, maxnew, bs=4):
    return SequenceConfig(rid, list(ids), maxnew, bs)


# ------------------------------------------------ 1. 需求给的小例子

pool = make_pool(blocks=4, prefix=True, over_subscribe=True)
# 0 被使用、1 是闲置 prefix 缓存、2/3 真正空闲
a = req("a", [1, 2, 3, 4], 1)
pool.allocate_block(a)
pool.ensure_blocks(a, 4)
a.cache.length = 4
pool.publish_computed_blocks(a)          # 第 0 块带上 hash
pool.deallocate_block(a)                 # 引用归零，但带 hash -> 闲置缓存
b = req("b", [5, 6, 7, 8], 1)
pool.allocate_block(b)
pool.ensure_blocks(b, 4)                 # b 用掉第 1 块
cached = set(pool.hash_to_block.values())
check("小例子：带 hash 的块没有进空闲堆", not (cached & set(pool.free_queue)),
      f"cached={sorted(cached)} heap={sorted(pool.free_queue)}")

nxt = req("n", [9, 10, 11, 12], 1)
pool.allocate_block(nxt)
plan = pool._plan_block_growth(nxt, 1)
check("小例子：下一次应拿编号最小的真正空闲块 2",
      plan.new_block_ids == [2], str(plan.new_block_ids))
check("小例子：计划阶段没动堆", 2 in pool.free_queue, str(sorted(pool.free_queue)))
pool._commit_block_growth(nxt, plan)
check("小例子：提交后 2 离开了空闲堆", 2 not in pool.free_queue, str(sorted(pool.free_queue)))

# 0 释放后若无 hash，下一次应先拿 0
pool.deallocate_block(b)                 # b 用的第 1 块回到空闲堆
check("小例子：释放后第 1 块回到空闲堆", 1 in pool.free_queue, str(sorted(pool.free_queue)))

# ------------------------------------- 2. 从 step48 移植：计划失败无副作用


def snap(pool):
    return {
        "usage": list(pool.block_usage),
        "hash_to_block": dict(pool.hash_to_block),
        "block_to_hash": dict(pool.block_to_hash),
        "promised_pool": pool.promised_blocks,
        "free_queue": list(pool.free_queue),
    }


def seq_snap(seq):
    return {"table": list(seq.cache.block_table) if seq.cache.block_table is not None else None,
            "length": seq.cache.length, "hashes": list(seq.block_hashes),
            "promised": seq.promised_blocks, "reused": seq.reused_tokens}


pool = make_pool(blocks=4)
bad = req("bad", range(1, 6), 40)        # 最坏 11 块 > 池子 4 块
before, before_seq = snap(pool), seq_snap(bad)
raised = None
try:
    pool._plan_admission(bad)
except InfeasibleRequest as exc:
    raised = exc
check("不可行：计划阶段抛 InfeasibleRequest", raised is not None)
check("不可行：池状态与空闲堆都未变",
      snap(pool) == before and seq_snap(bad) == before_seq)

pool = make_pool(blocks=4, prefix=False, over_subscribe=False)
holder = req("holder", range(1, 5), 8)
pool.allocate_block(holder)
waiter = req("waiter", range(10, 18), 8)
before, before_seq = snap(pool), seq_snap(waiter)
plan = pool._plan_admission(waiter)
check("暂时不够：计划返回 None", plan is None)
check("暂时不够：池状态与空闲堆都未变", snap(pool) == before and seq_snap(waiter) == before_seq)
check("暂时不够：allocate_block 返回 False 且无副作用",
      pool.allocate_block(waiter) is False and snap(pool) == before)

pool = make_pool(blocks=2, prefix=False, over_subscribe=True)
s = req("s", [1, 2, 3, 4], 4)
pool.allocate_block(s)
pool.ensure_blocks(s, 4)
s.cache.length = 4
other = req("o", [10, 11, 12, 13], 4)
pool.allocate_block(other)
pool.ensure_blocks(other, 4)             # 池子满了
before, before_seq = snap(pool), seq_snap(s)
ok = pool.ensure_blocks(s, 4)
check("补块失败：返回 False", ok is False)
check("补块失败：池状态、请求状态、空闲堆都未变",
      snap(pool) == before and seq_snap(s) == before_seq)

# 计划阶段不淘汰、不弹堆
pool = make_pool(blocks=4, prefix=True, over_subscribe=True)
pub = req("pub", [1, 2, 3, 4], 1)
pool.allocate_block(pub)
pool.ensure_blocks(pub, 4)
pub.cache.length = 4
pool.publish_computed_blocks(pub)
pool.deallocate_block(pub)
grower = req("grow", [90, 91, 92, 93], 8)
pool.allocate_block(grower)
before = snap(pool)
plan = pool._plan_block_growth(grower, 4)
check("计划阶段：选了可淘汰块，但堆与 hash 都没动", snap(pool) == before,
      f"heap={pool.free_queue}")
pool._commit_block_growth(grower, plan)
check("提交阶段：淘汰的 hash 双向索引同步消失",
      all(b not in pool.block_to_hash for b in plan.evict_block_ids))

# ------------------------------------- 3. 跑完整负载，逐步校验不变量

LOADS = [
    ("fcfs / prefix 关", dict(scheduling_policy="fcfs", num_kv_blocks=6)),
    ("priority / prefix 开", dict(scheduling_policy="priority", num_kv_blocks=6,
                                  enable_prefix_caching=True)),
    ("priority / 容量压力", dict(scheduling_policy="priority", num_kv_blocks=4,
                                 enable_prefix_caching=True)),
    ("fcfs / 大池子", dict(scheduling_policy="fcfs", num_kv_blocks=64)),
]
for label, kw in LOADS:
    for seed in (29, 7):
        e = make_engine(seed=seed, **kw)
        for i in range(5):
            e.add_request({"request_id": f"r{i}", "prompt_ids": [(i * 7 + j) % 60 for j in range(6)],
                           "max_new_tokens": 8, "priority": (i % 3) - 1})
        n = 0
        while e.has_unfinished_requests():
            e.step()
            n += 1
            assert n < 600, "疑似活锁"
        assert_invariant(e.kv_cache_pool, f"{label} 结束")
        check(f"{label}（seed={seed}）：活动引用与承诺归零",
              all(u == 0 for u in e.kv_cache_pool.block_usage)
              and e.kv_cache_pool.promised_blocks == 0)

check("每一次状态转换操作完成后：集合等式与无重复都成立", not VIOLATIONS,
      str(VIOLATIONS[:2]))
check("被豁免的 _evict_block 确实只在提交内部被调用", not EVICT_OUTSIDE_COMMIT)
print(f"      共校验 {len(LOADS) * 2} 组负载的每一次块状态转换")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
