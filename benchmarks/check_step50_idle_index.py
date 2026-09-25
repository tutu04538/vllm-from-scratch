"""第 50 关：可分配块单链表的增量维护与「计划失败无副作用」。

两件事：
  1. 每一次**状态转换操作完成后**校验：链表集合 == {b | block_usage[b] == 0}、
     无重复、前后指针双向一致、num_allocatable 与链长相符；
  2. 从 step49 移植「准入失败 / 补块失败 / 计划阶段不动链表」的无副作用测试，
     并把链表与计数加进快照。

CPU / FP32，无需模型；直接构造池与请求状态。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step50.cache import InfeasibleRequest, KVCachePool
from step50.request import SequenceConfig

FAIL = []
VIOLATIONS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def chain(pool):
    """按 next 指针把可分配链表读出来。"""
    out, cur = [], pool.block_next[pool._SENTINEL_HEAD]
    while cur != pool._SENTINEL_TAIL:
        out.append(cur)
        cur = pool.block_next[cur]
    return out


def assert_invariant(pool, where):
    truth = set(pool._allocatable_block_indices())     # 独立全池扫描，不用链表自己校验自己
    seq = chain(pool)
    if set(seq) != truth:
        VIOLATIONS.append({"where": where, "chain": sorted(seq), "truth": sorted(truth)})
    if len(seq) != len(set(seq)):
        VIOLATIONS.append({"where": where, "why": "链内重复", "chain": seq})
    if pool.num_allocatable != len(seq):
        VIOLATIONS.append({"where": where, "why": "计数不符",
                           "num_allocatable": pool.num_allocatable, "len": len(seq)})
    back, cur = [], pool.block_prev[pool._SENTINEL_TAIL]
    while cur != pool._SENTINEL_HEAD:
        back.append(cur)
        cur = pool.block_prev[cur]
    if back != list(reversed(seq)):
        VIOLATIONS.append({"where": where, "why": "prev/next 不一致",
                           "forward": seq, "backward": list(reversed(back))})


IN_COMMIT = {"depth": 0}
EVICT_OUTSIDE_COMMIT = []


def instrument(pool):
    """包住所有会改块状态的操作，**操作完成后**校验不变量。

    `_evict_block` 是 `_commit_*` 的内部步骤（只删 hash 关联），
    链表与 block_usage 是两个结构，中间时刻本就不自洽；为免这个豁免变成借口，
    下面断言它 Never 在提交之外被调用。
    """
    def wrap(name, check_after=True):
        orig = getattr(pool, name)

        def wrapper(*a, **kw):
            if name in ("_commit_block_growth", "_commit_admission"):
                IN_COMMIT["depth"] += 1
            try:
                if name == "_evict_block" and not IN_COMMIT["depth"]:
                    EVICT_OUTSIDE_COMMIT.append(True)
                out = orig(*a, **kw)
            finally:
                if name in ("_commit_block_growth", "_commit_admission"):
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


def snap(pool):
    return {
        "usage": list(pool.block_usage),
        "hash_to_block": dict(pool.hash_to_block),
        "block_to_hash": dict(pool.block_to_hash),
        "promised_pool": pool.promised_blocks,
        "chain": chain(pool),
        "num_allocatable": pool.num_allocatable,
    }


def seq_snap(seq):
    return {"table": list(seq.cache.block_table) if seq.cache.block_table is not None else None,
            "length": seq.cache.length, "hashes": list(seq.block_hashes),
            "promised": seq.promised_blocks, "reused": seq.reused_tokens}


def req(rid, ids, maxnew, bs=4):
    return SequenceConfig(rid, list(ids), maxnew, bs)


# ------------------------------------------------ 1. 需求给的小例子

pool = make_pool(blocks=4, prefix=True, over_subscribe=True)
a = req("a", [1, 2, 3, 4], 1)
pool.allocate_block(a)
pool.ensure_blocks(a, 4)
a.cache.length = 4
pool.publish_computed_blocks(a)          # 第 0 块带上 hash
pool.deallocate_block(a)                 # 引用归零，但带 hash -> 闲置缓存
b = req("b", [5, 6, 7, 8], 1)
pool.allocate_block(b)
pool.ensure_blocks(b, 4)                 # b 用掉一个块
cached = set(pool.hash_to_block.values())
check("小例子：a 的缓存块在链上（可分配）", cached <= set(chain(pool)),
      f"chain={chain(pool)} cached={sorted(cached)}")
check("小例子：a 的缓存块排在队尾一侧（最后才被淘汰）",
      chain(pool).index(sorted(cached)[0]) > 0, f"chain={chain(pool)}")
order_before = list(chain(pool))

nxt = req("n", [9, 10, 11, 12], 1)
pool.allocate_block(nxt)
plan = pool._plan_block_growth(nxt, 1)
check("小例子：计划阶段不动链表", chain(pool) == order_before, str(chain(pool)))
pool._commit_block_growth(nxt, plan)
check("小例子：提交后取走的块离开链表",
      plan.new_block_ids[0] not in chain(pool), str(chain(pool)))

# ------------------------------------- 2. 命中借用的块先离开链表

pool = make_pool(blocks=8, prefix=True, over_subscribe=True)
src = req("src", [1, 2, 3, 4, 5, 6, 7, 8], 1)
pool.allocate_block(src)
pool.ensure_blocks(src, 8)
src.cache.length = 8
pool.publish_computed_blocks(src)
pool.deallocate_block(src)                       # 两个块成为闲置缓存
idle_blocks = set(pool.hash_to_block.values())
check("命中前：两个缓存块都在链表上", idle_blocks <= set(chain(pool)), str(chain(pool)))

hit = req("hit", [1, 2, 3, 4, 5], 4)      # 只够命中第 0 块
pool.allocate_block(hit)
check("命中后：借走的缓存块离开链表",
      not (set(hit.cache.block_table) & set(chain(pool))), f"chain={chain(pool)}")
check("命中后：未借到的缓存块还在链表上",
      bool((idle_blocks - set(hit.cache.block_table)) & set(chain(pool))))

pool.deallocate_block(hit)
check("借用的块释放后回到链表",
      set(hit.cache.block_table) <= set(chain(pool)), str(chain(pool)))

# ------------------------------------- 3. 计划失败无副作用

pool = make_pool(blocks=4)
bad = req("bad", range(1, 6), 40)        # 最坏 11 块 > 池子 4 块
before, before_seq = snap(pool), seq_snap(bad)
raised = None
try:
    pool._plan_admission(bad)
except InfeasibleRequest as exc:
    raised = exc
check("不可行：计划阶段抛 InfeasibleRequest", raised is not None)
check("不可行：池状态、链表都未变", snap(pool) == before and seq_snap(bad) == before_seq)

pool = make_pool(blocks=4, prefix=False, over_subscribe=False)
holder = req("holder", range(1, 5), 8)
pool.allocate_block(holder)
waiter = req("waiter", range(10, 18), 8)
before, before_seq = snap(pool), seq_snap(waiter)
check("暂时不够：计划返回 None", pool._plan_admission(waiter) is None)
check("暂时不够：池状态、链表都未变", snap(pool) == before and seq_snap(waiter) == before_seq)
check("暂时不够：allocate_block 返回 False 且无副作用",
      pool.allocate_block(waiter) is False and snap(pool) == before)

pool = make_pool(blocks=2, prefix=False, over_subscribe=True)
s = req("s", [1, 2, 3, 4], 4)
pool.allocate_block(s)
pool.ensure_blocks(s, 4)
s.cache.length = 4
other = req("o", [10, 11, 12, 13], 4)
pool.allocate_block(other)
pool.ensure_blocks(other, 4)             # 链空了
before, before_seq = snap(pool), seq_snap(s)
check("补块失败：返回 False", pool.ensure_blocks(s, 4) is False)
check("补块失败：池状态、请求状态、链表都未变",
      snap(pool) == before and seq_snap(s) == before_seq)

# ------------------------------------- 4. 跑完整负载，每次状态转换后校验

DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


def make_engine(seed=29, **kw):
    from step50 import Engine
    torch.manual_seed(seed)
    cfg = dict(max_num_seqs=2, max_num_batched_tokens=8, block_size=4, num_kv_blocks=8,
               enable_prefix_caching=False, preemption_mode="recompute",
               scheduling_policy="fcfs")
    cfg.update(kw)
    e = Engine(device="cpu", **cfg, **DIMS)
    instrument(e.kv_cache_pool)
    return e


LOADS = [
    ("fcfs / prefix 关", dict(scheduling_policy="fcfs", num_kv_blocks=6)),
    ("priority / prefix 开", dict(scheduling_policy="priority", num_kv_blocks=6,
                                  enable_prefix_caching=True)),
    ("priority / 容量压力", dict(scheduling_policy="priority", num_kv_blocks=4,
                                 enable_prefix_caching=True)),
    ("fcfs / 承诺式", dict(preemption_mode=None, num_kv_blocks=8,
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
        pool = e.kv_cache_pool
        check(f"{label}（seed={seed}）：活动引用与承诺归零、链表覆盖全池空闲块",
              all(u == 0 for u in pool.block_usage) and pool.promised_blocks == 0
              and set(chain(pool)) == set(pool._allocatable_block_indices()))

check("每一次状态转换操作完成后：集合、无重复、双向指针、计数都成立",
      not VIOLATIONS, str(VIOLATIONS[:2]))
check("被豁免的 _evict_block 确实只在提交内部被调用", not EVICT_OUTSIDE_COMMIT)

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
