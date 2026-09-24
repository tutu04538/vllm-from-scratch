"""第 48 关：KV 准入 / 补块的「先计划、后提交」边界（需求 §4.4 / §9.3）。

重点验证失败路径**一个字节都不改**：块引用、双向 hash、块表、cache.length、
两级承诺计数、reused_tokens 全部保持原样。

CPU / FP32，无需模型；直接构造池与请求状态。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step48.cache import InfeasibleRequest, KVCachePool
from step48.request import CacheConfig, SequenceConfig

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def make_pool(*, blocks=8, block_size=4, prefix=True, over_subscribe=True):
    return KVCachePool(block_size, blocks, 2, 4, torch.device("cpu"),
                       enable_prefix_caching=prefix, num_layers=1, dtype=torch.float32,
                       over_subscribe=over_subscribe)


def snap(pool):
    """池与请求状态的可比快照。"""
    return {
        "usage": list(pool.block_usage),
        "hash_to_block": dict(pool.hash_to_block),
        "block_to_hash": dict(pool.block_to_hash),
        "promised_pool": pool.promised_blocks,
        "lru": list(pool.block_last_used),
    }


def seq_snap(seq):
    return {
        "table": list(seq.cache.block_table) if seq.cache.block_table is not None else None,
        "length": seq.cache.length,
        "hashes": list(seq.block_hashes),
        "promised": seq.promised_blocks,
        "reused": seq.reused_tokens,
    }


def req(rid, ids, maxnew, bs=4):
    return SequenceConfig(rid, list(ids), maxnew, bs)


# ---------------------------------------------------------------- 1. 不可行

pool = make_pool(blocks=4)
bad = req("bad", range(1, 6), 40)          # 最坏 ceil((5+40-1)/4)=11 块 > 池子 4 块
before, before_seq = snap(pool), seq_snap(bad)
raised = None
try:
    pool._plan_admission(bad)
except InfeasibleRequest as exc:
    raised = exc
check("不可行：计划阶段抛 InfeasibleRequest", raised is not None)
check("不可行：计划失败后池状态未变", snap(pool) == before, str(snap(pool)))
check("不可行：计划失败后请求状态未变", seq_snap(bad) == before_seq, str(seq_snap(bad)))

# ------------------------------------------------------- 2. 暂时不够（承诺式）

pool = make_pool(blocks=4, prefix=False, over_subscribe=False)
holder = req("holder", range(1, 5), 8)     # 最坏 ceil((4+8-1)/4)=3 块
pool.allocate_block(holder)                # 承诺 3 块
waiter = req("waiter", range(10, 18), 8)   # 最坏 3 块
before, before_seq = snap(pool), seq_snap(waiter)
plan = pool._plan_admission(waiter)
check("暂时不够：计划阶段返回 None（不是抛异常）", plan is None, str(plan))
check("暂时不够：计划失败后池状态未变", snap(pool) == before)
check("暂时不够：计划失败后请求状态未变", seq_snap(waiter) == before_seq)
check("暂时不够：allocate_block 返回 False 且无副作用",
      pool.allocate_block(waiter) is False and snap(pool) == before
      and seq_snap(waiter) == before_seq)

# ------------------------------------------------- 3. 本轮补块失败（超卖模式）

pool = make_pool(blocks=2, prefix=False, over_subscribe=True)
# 每条单独跑最坏 ceil((4+4-1)/4)=2 块 <= 池子 2 块，都可行
s = req("s", [1, 2, 3, 4], 4)
pool.allocate_block(s)
pool.ensure_blocks(s, 4)                   # 占住第 0 块
s.cache.length = 4                         # 假装 prompt 已经算完
other = req("o", [10, 11, 12, 13], 4)
pool.allocate_block(other)
pool.ensure_blocks(other, 4)               # 占住第 1 块 —— 池子满了
before, before_seq = snap(pool), seq_snap(s)
ok = pool.ensure_blocks(s, 4)              # 想再补一块：分不到
check("补块失败：返回 False 而不是抛「记账损坏」", ok is False, str(ok))
check("补块失败：池状态未变（没有部分淘汰）", snap(pool) == before, str(snap(pool)))
check("补块失败：请求状态未变", seq_snap(s) == before_seq, str(seq_snap(s)))

# --------------------------------------------- 4. 计划阶段不淘汰、提交才淘汰

pool = make_pool(blocks=4, prefix=True, over_subscribe=True)
pub = req("pub", [1, 2, 3, 4], 1)
pool.allocate_block(pub)
pool.ensure_blocks(pub, 4)                 # 正常流程：先补块，模型再往里写
pub.cache.length = 4
pool.publish_computed_blocks(pub)          # 第 0 块进缓存
pool.deallocate_block(pub)                 # 变成闲置缓存
cached_before = dict(pool.hash_to_block)
check("发布/释放：闲置缓存还在（不是泄漏，是可复用块）", bool(cached_before))

grower = req("grow", [90, 91, 92, 93], 8)
pool.allocate_block(grower)
plan = pool._plan_block_growth(grower, 4)
before = snap(pool)
check("计划阶段：挑出的可淘汰块与 hash 一致",
      all(b in cached_before.values() for b in plan.evict_block_ids),
      f"plan={plan.evict_block_ids} cached={list(cached_before.values())}")
check("计划阶段：一个块都还没被淘汰（LRU/hash 未动）", snap(pool) == before,
      str(snap(pool)))
pool._commit_block_growth(grower, plan)
check("提交阶段：淘汰的 hash 双向索引同步消失",
      all(b not in pool.block_to_hash for b in plan.evict_block_ids)
      and all(h not in pool.hash_to_block for h in cached_before),
      str(pool.hash_to_block))
check("提交阶段：新块引用为 1 且进了块表",
      all(pool.block_usage[b] == 1 for b in plan.new_block_ids)
      and grower.cache.block_table[-len(plan.new_block_ids):] == plan.new_block_ids)

# -------------------------------------------- 5. 正命中：引用与 reused 在提交后才变

pool = make_pool(blocks=8, prefix=True, over_subscribe=True)
src = req("src", [1, 2, 3, 4, 5, 6, 7, 8], 1)
pool.allocate_block(src)
pool.ensure_blocks(src, 8)
src.cache.length = 8
pool.publish_computed_blocks(src)
pool.deallocate_block(src)

hit = req("hit", [1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
before = snap(pool)
plan = pool._plan_admission(hit)
check("正命中：计划阶段查到了完整块", plan is not None and len(plan.matched_block_ids) > 0,
      str(plan))
check("正命中：**查询不改引用、不改 LRU**", snap(pool) == before, str(snap(pool)))
check("正命中：计划阶段 reused_tokens 仍是 0", hit.reused_tokens == 0, str(hit.reused_tokens))
pool._commit_admission(hit, plan)
check("正命中：提交后才加引用与 reused_tokens",
      hit.reused_tokens == len(plan.matched_block_ids) * 4
      and hit.cache.length == len(plan.matched_block_ids) * 4
      and all(pool.block_usage[b] >= 1 for b in plan.matched_block_ids),
      f"reused={hit.reused_tokens} len={hit.cache.length}")

# ---------------------------------------- 6. 承诺式记账错误仍在计划阶段报出

pool = make_pool(blocks=2, prefix=False, over_subscribe=False)
a = req("a", [1, 2, 3, 4], 1)
pool.allocate_block(a)
pool.ensure_blocks(a, 4)                    # a 占 1 块，承诺还剩 0
b = req("b", [5, 6, 7, 8], 1)
pool.allocate_block(b)
pool.ensure_blocks(b, 4)                    # b 也占 1 块
fake = req("fake", [9, 10, 11, 12], 1)
fake.promised_blocks = 0                    # 假装它没有承诺额度
before = snap(pool)
err = None
try:
    pool._plan_block_growth(fake, 1)
except RuntimeError as exc:
    err = exc
check("承诺式：容量不足仍是明确的记账错误（保留原提示）",
      err is not None and "记账出错了" in str(err), str(err))
check("承诺式：报错前没有部分淘汰", snap(pool) == before, str(snap(pool)))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
