"""第 52 关：投机解码的纯函数与 KV 回滚（需求 §4.1）。

分四块，都不跑模型：
  1. propose_ngram：找不到 / 找到 1 个 / 找到 2 个 / 多个匹配取最近 / 只读视图；
  2. verify_drafts：首枚拒绝 / 部分接受 / 全部接受 / 无草稿 / EOS / 输出上限前置条件；
  3. KVCachePool.truncate 与 can_grow：跨块回滚、块表与引用、链表不变量；
  4. 配置组合校验：不支持的组合与非法采样参数都要**明确报错**，不静默退化。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step53 import Engine, KVCachePool, SequenceConfig
from step53.speculative import propose_ngram, verify_drafts

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        return str(exc)
    return None


# ------------------------------------------------ 1. propose_ngram

check("找不到：历史里没有相同的 n-gram", propose_ngram([1, 2, 3], 2, 2) == [])
check("历史比 n 还短", propose_ngram([1, 2], 2, 2) == [] and propose_ngram([], 2, 2) == [])
check("找到 2 个：末尾 (1,2) 之前出现过", propose_ngram([1, 2, 1, 2], 2, 2) == [1, 2])
check("找到 1 个：匹配后面只剩 1 个已知 token",
      propose_ngram([7, 5, 5, 5], 2, 2) == [5])
check("多个匹配取**最近**的那次，不是最早的那次",
      propose_ngram([1, 2, 3, 1, 2, 4, 1, 2], 2, 2) == [4, 1])
check("k 小于可用续写时按 k 截断",
      propose_ngram([1, 2, 1, 2], 2, 1) == [1])
check("n=1 也能用", propose_ngram([4, 5, 4], 1, 1) == [5])
check("末尾的 n 个 token 不会自己匹配自己",
      propose_ngram([1, 2, 3, 1, 2], 2, 2) == [3, 1])

# 只读视图：整段历史不复制成 list，只用下标访问
view_seq = SequenceConfig("view", [1, 2, 3, 1, 2], 4, 4)
check("对只读视图 all_token_ids 直接工作（不要求是 list）",
      propose_ngram(view_seq.all_token_ids, 2, 2) == [3, 1])

big = SequenceConfig("big", list(range(1, 33)), 16, 4)
big.append_output_ids(list(range(1000, 9192)))   # 严格递增：没有任何重复的 2-gram
import time
t0 = time.perf_counter_ns()
for _ in range(200):
    propose_ngram(big.all_token_ids, 2, 2)
dt = (time.perf_counter_ns() - t0) / 200 / 1000
check("8192 长历史 + 一路无匹配：线性扫描走满，不复制整段历史",
      dt < 20000, f"{dt:.1f} μs")

# ------------------------------------------------ 2. verify_drafts

EOS = {63}
d = verify_drafts([5, 6], [9, 7, 8], EOS, remaining_outputs=8)
check("首枚被拒：提交 [t0]，本轮输入只留 [x]",
      (d.num_accepted, d.committed_ids, d.kept_inputs) == (0, [9], 1))
d = verify_drafts([5, 6], [5, 9, 1], EOS, remaining_outputs=8)
check("部分接受：提交 [d0, t1]，本轮输入留 [x, d0]",
      (d.num_accepted, d.committed_ids, d.kept_inputs) == (1, [5, 9], 2))
d = verify_drafts([5, 6], [5, 6, 7], EOS, remaining_outputs=8)
check("全部接受：提交 [d0, d1, bonus]，本轮输入留 [x, d0, d1]",
      (d.num_accepted, d.committed_ids, d.kept_inputs) == (2, [5, 6, 7], 3))
d = verify_drafts([], [9], EOS, remaining_outputs=8)
check("无草稿：就是普通的 1-token 路径",
      (d.num_accepted, d.committed_ids, d.kept_inputs) == (0, [9], 1))
d = verify_drafts([5, 6], [5, 6, 63], EOS, remaining_outputs=8)
check("bonus 是 EOS：全部提交后停下，草稿的 KV 仍然要留",
      (d.num_accepted, d.committed_ids, d.kept_inputs) == (2, [5, 6, 63], 3))
d = verify_drafts([63, 6], [63, 6, 7], EOS, remaining_outputs=8)
check("**草稿本身**是 EOS：只提交它，它之前的草稿才留 KV（这里没有）",
      (d.num_accepted, d.committed_ids, d.kept_inputs) == (2, [63], 1))
d = verify_drafts([5, 63, 8], [5, 63, 9, 1], EOS, remaining_outputs=8)
check("第二枚草稿是 EOS：留第一枚的 KV，第三枚根本不提交",
      (d.num_accepted, d.committed_ids, d.kept_inputs) == (2, [5, 63], 2))
# 「因 EOS 停下」不再是单独一个字段：它就是 committed_ids 的最后一个元素。
# 下面三个 EOS 用例里，提交列表都比接受数少的那个瞬间短，就是这个信号。
check("因 EOS 停下 = 提交的最后一枚是终止 token（不再单开字段）",
      verify_drafts([5, 6], [5, 6, 63], EOS, 8).committed_ids[-1] in EOS
      and verify_drafts([5, 6], [5, 6, 7], EOS, 8).committed_ids[-1] not in EOS)

# 输出上限是**前置条件**（K <= R-1），不是截断阈值：_plan_drafts() 保证它成立，
# 传进来不满足就直接报错——截断会让 kept_inputs 超过实际提交的 token 数。
check("输出上限不够（K+1 > R）：报错，不默默截断",
      raises(verify_drafts, [5, 6], [5, 6, 7], EOS, 1) is not None
      and raises(verify_drafts, [5, 6], [5, 6, 7], EOS, 0) is not None)
check("刚好卡在边界（K+1 == R）时正常接受",
      verify_drafts([5, 6], [5, 6, 7], EOS, remaining_outputs=3).committed_ids == [5, 6, 7]
      and verify_drafts([5, 6], [5, 9, 7], EOS, remaining_outputs=3).committed_ids == [5, 9])
check("行数与草稿数不匹配要报错",
      raises(verify_drafts, [5, 6], [5, 6], EOS, 8) is not None)

# ------------------------------------------------ 3. truncate / can_grow


def make_pool(num_kv_blocks=8, enable_prefix_caching=False):
    return KVCachePool(4, num_kv_blocks, 1, 8, torch.device("cpu"),
                       enable_prefix_caching=enable_prefix_caching, num_layers=1)


def chain_of(pool):
    out, cur = [], pool.block_next[pool._SENTINEL_HEAD]
    while cur != pool._SENTINEL_TAIL:
        out.append(cur)
        cur = pool.block_next[cur]
    return out


def pool_ok(pool):
    """链表不变量 + 引用计数与块表的双向一致。"""
    chain = chain_of(pool)
    no_dup = len(chain) == len(set(chain))
    same = set(chain) == set(pool._allocatable_block_indices())
    count = pool.num_allocatable == len(chain)
    return no_dup and same and count


pool = make_pool()
seq = SequenceConfig("T", [1, 2, 3, 4, 5, 6], 8, 4)
assert pool.allocate_block(seq) and pool.ensure_blocks(seq, 6)
seq.cache.length = 6
check("回滚前：块表 2 块、长度 6、链表不变量成立",
      len(seq.cache.block_table) == 2 and seq.cache.length == 6 and pool_ok(pool))

pool.truncate(seq, 4)
check("回滚到块边界（4）：块表退到 1 块、多占的块已归还",
      seq.cache.length == 4 and len(seq.cache.block_table) == 1
      and pool.block_usage[0] == 1 and pool.block_usage[1] == 0 and pool_ok(pool))

assert pool.ensure_blocks(seq, 4)
seq.cache.length = 8
pool.truncate(seq, 3)
check("回滚到块内（3）：块表退到 1 块，尾块里的旧内容留着以后覆盖",
      seq.cache.length == 3 and len(seq.cache.block_table) == 1 and pool_ok(pool))

assert pool.ensure_blocks(seq, 5)
seq.cache.length = 8
pool.truncate(seq, 0)
check("一路回滚到 0：所有块归还、引用归零、链表完整",
      seq.cache.length == 0 and seq.cache.block_table == []
      and all(u == 0 for u in pool.block_usage) and pool_ok(pool))

assert pool.ensure_blocks(seq, 6)
seq.cache.length = 6
check("非法回滚目标（负数 / 超过当前长度）要报错",
      raises(pool.truncate, seq, -1) is not None
      and raises(pool.truncate, seq, 7) is not None)

# 已经发布过的完整块不能被回滚动到：投机每一步的 new_length 都不小于本步起点，
# 而发布过的是 [0, floor(start/block_size)) 那些块，必然仍在保留范围内。
p2 = make_pool(enable_prefix_caching=True)
s2 = SequenceConfig("H", [1, 2, 3, 4, 5, 6, 7, 8], 4, 4)
assert p2.allocate_block(s2) and p2.ensure_blocks(s2, 8)
s2.cache.length = 8
p2.publish_computed_blocks(s2)
assert len(s2.block_hashes) == 2
published = {b: p2.block_to_hash[b] for b in s2.cache.block_table}   # 块 -> hash
assert p2.ensure_blocks(s2, 4)
s2.cache.length = 12                      # 模拟本轮 forward 多写了一整块
released = s2.cache.block_table[2]
p2.truncate(s2, 8)                        # 把这一整块回滚掉
check("回滚不会动已发布的完整块，hash 双向索引保持",
      p2.block_to_hash == published
      and all(p2.hash_to_block[h] == b for b, h in published.items())
      and s2.cache.block_table == list(published.keys())
      and s2.cache.length == 8 and pool_ok(p2))
check("被回滚的整块（从未发布过，因此没有 hash）以「真正空闲」回队首，优先复用",
      p2.block_next[p2._SENTINEL_HEAD] == released
      and p2.block_usage[released] == 0 and released not in p2.block_to_hash)

# can_grow：只读
p3 = make_pool(num_kv_blocks=4)
s3 = SequenceConfig("G", [1, 2, 3, 4], 8, 4)
assert p3.allocate_block(s3) and p3.ensure_blocks(s3, 4)
s3.cache.length = 4
before = (list(p3.block_usage), chain_of(p3), len(s3.cache.block_table), s3.cache.length)
grow_1 = p3.can_grow(s3, 1)
grow_16 = p3.can_grow(s3, 16)
after = (list(p3.block_usage), chain_of(p3), len(s3.cache.block_table), s3.cache.length)
check("can_grow 只读：问完之后池子与请求状态一个字节都没变", before == after)
check("can_grow 的判断与可分配链一致：够 1 个不够 16 个",
      grow_1 is True and grow_16 is False,
      f"grow_1={grow_1} grow_16={grow_16}")
check("can_grow 不摘链、不清 hash（链表不变量仍成立）", pool_ok(p3))

# ------------------------------------------------ 4. 配置组合校验

DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])
BASE = dict(device="cpu", max_num_seqs=3, max_num_batched_tokens=8, block_size=4,
            num_kv_blocks=16, enable_prefix_caching=False, attention_backend="torch",
            scheduling_policy="fcfs", speculative_mode="ngram",
            **DIMS)

ok_engine = Engine(**BASE)
check("最小合法组合能构造出来（本关起 max_num_seqs 可以大于 1）",
      ok_engine.speculative_mode == "ngram" and ok_engine.scheduler.max_num_seqs == 3)
check("max_num_seqs 不再受限：2 / 4 / 8 都能构造",
      all(raises(Engine, **dict(BASE, max_num_seqs=n)) is None for n in (2, 4, 8)))

for label, override, expect in [
    ("priority 调度", dict(scheduling_policy="priority"), "scheduling_policy"),
    ("开前缀缓存", dict(enable_prefix_caching=True), "enable_prefix_caching"),
    ("未知模式", dict(speculative_mode="eagle"), "speculative_mode"),
    ("num_speculative_tokens=0", dict(num_speculative_tokens=0), "num_speculative_tokens"),
    ("prompt_lookup_n=0", dict(prompt_lookup_n=0), "prompt_lookup_n"),
]:
    cfg = dict(BASE)
    cfg.update(override)
    message = raises(Engine, **cfg)
    check(f"{label}：明确报错（提到 {expect}）",
          message is not None and expect in message, message or "没有报错")

# 关闭投机时，这些组合仍然是合法的（不影响旧行为）
for label, override in [("priority 调度", dict(scheduling_policy="priority")),
                        ("开前缀缓存", dict(enable_prefix_caching=True))]:
    cfg = dict(BASE)
    cfg.update(override, speculative_mode=None)
    check(f"speculative_mode=None 时 {label} 仍然合法", raises(Engine, **cfg) is None)

# 采样参数：投机只支持贪心且无惩罚项
for label, request in [
    ("temperature=0.8", dict(temperature=0.8)),
    ("repetition_penalty=1.2", dict(repetition_penalty=1.2)),
    ("presence_penalty=0.5", dict(presence_penalty=0.5)),
]:
    engine = Engine(**BASE)
    message = raises(engine.add_request,
                     dict(request_id="x", prompt_ids=[1, 2, 3], max_new_tokens=4, **request))
    check(f"投机模式下 {label}：明确报错", message is not None, message or "没有报错")

engine = Engine(**BASE)
check("投机模式下普通的贪心请求可以入队",
      raises(engine.add_request,
             dict(request_id="x", prompt_ids=[1, 2, 3], max_new_tokens=4)) is None)

# ------------------------------------------------ 5. K 的四个上限（_plan_drafts）

# 历史 = [1,2,3,1,2,3,1]，已算到 6、还差最后 1 个；末尾 2-gram (3,1) 的续写是 [2,3]
from step53.scheduler import Scheduler


def plan_drafts(spare=8, max_new_tokens=8, **overrides):
    cfg = dict(speculative_mode="ngram", num_speculative_tokens=2, prompt_lookup_n=2,
               max_seq_len=64)
    cfg.update(overrides)
    seq = SequenceConfig("D", [1, 2, 3, 1, 2], max_new_tokens, 4)
    seq.append_output_ids([3, 1])
    seq.cache.length = 6
    return Scheduler(**cfg)._plan_drafts(seq, spare)


check("基准：四个上限都够时，K 取配置上限",
      plan_drafts(spare=8) == [2, 3])
check("剩余 token 预算不足：K 跟着缩短", plan_drafts(spare=1) == [2])
check("剩余输出上限：K <= R-1（全部接受还要一个 bonus）",
      plan_drafts(max_new_tokens=4) == [2]        # R = 4-2 = 2 -> K <= 1
      and plan_drafts(max_new_tokens=3) == [])    # R = 1 -> K <= 0
check("剩余上下文长度：算到 max_seq_len 为止",
      plan_drafts(max_seq_len=7) == []            # 7 - 6 - 1 = 0
      and plan_drafts(max_seq_len=8) == [2])      # 8 - 6 - 1 = 1
check("配置上限本身也生效", plan_drafts(num_speculative_tokens=1) == [2])
check("n 太长导致找不到匹配时返回空（n=5 时历史里没有同样的 5-gram）",
      plan_drafts(prompt_lookup_n=5) == [])
check("关掉投机时不提议", plan_drafts(speculative_mode=None) == [])

no_output = SequenceConfig("N", [1, 2, 3, 1, 2], 8, 4)
no_output.cache.length = 5                      # 还差 0 个 token 没算
check("还没有生成过 token 时不投机（没有可比较的续写行为）",
      Scheduler(speculative_mode="ngram", max_seq_len=64)._plan_drafts(no_output, 8) == [])
mid_prefill = SequenceConfig("M", [1, 2, 3, 1, 2], 8, 4)
mid_prefill.append_output_ids(3)
mid_prefill.cache.length = 3                    # 还差 3 个 token 没算
check("prefill / 重算途中（差的 token 不是 1 个）不投机",
      Scheduler(speculative_mode="ngram", max_seq_len=64)._plan_drafts(mid_prefill, 8) == [])

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
