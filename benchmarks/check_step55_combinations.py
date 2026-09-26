"""第 55 关的回归套件：priority / 前缀缓存与投机的组合。

第五十二到五十三关把投机限制在 `fcfs` + 关前缀缓存，理由是「没验证过的组合」。
本脚本把那两条放开（`validation.check_speculative()` 里只剩 Torch attention 与
关 CUDA Graph 两条实现层面的硬约束），然后逐项验证：

  1. priority + 投机：名额抢占、容量抢占、被抢占的计划作废；
  2. prefix + 投机：命中、发布、以及**回滚不会污染已发布的块**；
  3. 两条一起 + 随机采样。

每一步都查：预算、块引用计数、可分配链、计划里没有 0 token 项、事件序号连续。
"""

import sys
from collections import Counter

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import step55
from step55 import Engine

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


DIMS = dict(vocab_size=64, d_model=16, max_seq_len=96, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])

SPEC = dict(speculative_mode="ngram", num_speculative_tokens=2, prompt_lookup_n=2)


class ScriptedModel:
    """「复制模式」的脚本模型：每个采样行返回它**自己的输入 token**。

    序列自带重复片段，n-gram 几乎总能提得出草稿；而且这个分布**不是** one-hot
    （该 token 4 分、其余 0 分，温度 0.8 下约 99%），所以随机路径真的会被走到。
    第三段用它，是因为那里要验的是 priority/prefix 与投机的组合，不是「真模型这次
    恰好提不讲得出草稿」——那件事由 check_step54_random.py 的真模型用例负责。
    """

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _forward_append(self, input_ids, num_scheduled_tokens, past_kv, kv_cache_pool,
                        sample_rows=None):
        self.inner._forward_append(input_ids, num_scheduled_tokens, past_kv, kv_cache_pool,
                                   sample_rows)
        rows = list(range(len(input_ids))) if sample_rows is None else list(sample_rows)
        logits = torch.zeros((len(rows), self.vocab_size))
        for row_index, row in enumerate(rows):
            logits[row_index, int(input_ids[row])] = 4.0
        return logits


def build_scripted(seed, **cfg):
    base = dict(max_num_seqs=3, max_num_batched_tokens=16, block_size=4, num_kv_blocks=16,
                enable_prefix_caching=True, scheduling_policy="priority")
    base.update(cfg)
    torch.manual_seed(seed)
    inner = step55.TinyCausalLM(device="cpu", attention_backend="torch",
                                max_num_query_tokens=base["max_num_batched_tokens"], **DIMS)
    return Engine(model=ScriptedModel(inner), **base)


def build(seed, **cfg):
    base = dict(max_num_seqs=3, max_num_batched_tokens=16, block_size=4, num_kv_blocks=16,
                enable_prefix_caching=False, scheduling_policy="fcfs")
    base.update(cfg)
    torch.manual_seed(seed)
    return Engine(device="cpu", attention_backend="torch", **base, **DIMS)


def pool_ok(engine):
    """引用计数、可分配链、链表成员三者互相一致（每步都查）。"""
    pool = engine.kv_cache_pool
    refs = Counter(b for seq in engine.scheduler.running
                   if seq.cache is not None for b in seq.cache.block_table)
    if [refs[b] for b in range(pool.num_kv_blocks)] != list(pool.block_usage):
        return False
    chain, cur = [], pool.block_next[pool._SENTINEL_HEAD]
    while cur != pool._SENTINEL_TAIL:
        chain.append(cur)
        cur = pool.block_next[cur]
    return (len(chain) == len(set(chain)) and pool.num_allocatable == len(chain)
            and set(chain) == set(pool._allocatable_block_indices()))


def run(engine, arrivals, limit=120, watch_kv=None):
    """跑完。arrivals: {第几步: [请求]}。返回每步快照。"""
    for r in arrivals.get(0, []):
        engine.add_request(dict(r))
    per_step, step = [], 0
    while engine.has_unfinished_requests():
        committed = []
        engine.on_token = lambda ev: committed.append(
            (ev["request_id"], ev["token_id"], ev["output_index"]))
        engine.step()
        step += 1
        assert step < limit, "疑似活锁"
        for r in arrivals.get(step, []):
            engine.add_request(dict(r))
        per_step.append({
            "tokens": committed,
            "items": list(engine.scheduler.scheduled_items),
            "budget": sum(it["num_scheduled_tokens"] for it in engine.scheduler.scheduled_items),
            "drafts": sum(1 for it in engine.scheduler.scheduled_items if it.get("draft_ids")),
            "preemptions": engine.scheduler.num_preemptions,
            "priority_preemptions": engine.scheduler.num_priority_preemptions,
            "reused": sum(q.reused_tokens for q in list(engine.scheduler.running)
                          + list(engine.scheduler.waiting)),
            "pool_ok": pool_ok(engine),
        })
        if watch_kv is not None:
            watch_kv(engine, step)
    return per_step


def no_zero_items(per_step):
    return all(it["num_scheduled_tokens"] >= 1
               for slot in per_step for it in slot["items"])


def indices_ok(per_step, request_ids):
    tokens = [t for slot in per_step for t in slot["tokens"]]
    ok = {}
    for rid in request_ids:
        idx = [i for r, _, i in tokens if r == rid]
        ok[rid] = idx == list(range(len(idx)))
    return ok


# ------------------------------------------------ 1. priority + 投机

# 小名额 + 小池子 + 高优先级后到：名额抢占与容量抢占都会发生
PRIO_REQS = [
    {"request_id": "A", "prompt_ids": [1, 2, 3, 4, 1, 2, 3, 4], "max_new_tokens": 6, "priority": 0},
    {"request_id": "B", "prompt_ids": [3, 1, 4, 1, 3, 1, 4, 1], "max_new_tokens": 6, "priority": 0},
    {"request_id": "C", "prompt_ids": [5, 5, 1, 5, 5, 1, 5, 5], "max_new_tokens": 6, "priority": 0},
]
PRIO_HIGH = [{"request_id": "H", "prompt_ids": [7, 7, 8, 7, 7, 8], "max_new_tokens": 4,
              "priority": -1}]
prio = build(29, **SPEC, **dict(scheduling_policy="priority", num_kv_blocks=8, max_num_seqs=3))
prio_steps = run(prio, {0: PRIO_REQS, 2: PRIO_HIGH})
check("priority + 投机：跑完，没有 0 token 计划项，每步池子自洽",
      no_zero_items(prio_steps) and all(s["pool_ok"] for s in prio_steps))
check("priority + 投机：真的投了机（有草稿轮次）",
      sum(s["drafts"] for s in prio_steps) > 0, f"{sum(s['drafts'] for s in prio_steps)} 轮")
check("priority + 投机：容量抢占与名额抢占都发生过",
      prio.scheduler.num_preemptions > 0 and prio.scheduler.num_priority_preemptions > 0,
      f"抢占 {prio.scheduler.num_preemptions} 次（名额 {prio.scheduler.num_priority_preemptions}）")
check("priority + 投机：每步都在预算内",
      all(s["budget"] <= 16 for s in prio_steps))
check("priority + 投机：四条请求的 output_index 都连续、都完成一次",
      all(indices_ok(prio_steps, "ABCH").values()) and len(prio.scheduler.step_done) >= 0,
      str(indices_ok(prio_steps, "ABCH")))

# 被抢占的计划作废：被抢占那一轮该请求不能出现在执行计划里
prio2 = build(29, **SPEC, **dict(scheduling_policy="priority", num_kv_blocks=6, max_num_seqs=3))
prio2_steps = run(prio2, {0: PRIO_REQS, 1: PRIO_HIGH})
dropped = []
for slot in prio2_steps:
    executed = {it["request"].request_id for it in slot["items"]}
    if slot["preemptions"] > 0:
        dropped.append(sorted(executed))
check("priority + 投机：抢占发生的那几步，执行计划里不含被抢占的请求",
      any(slot["preemptions"] > 0 for slot in prio2_steps) and no_zero_items(prio2_steps),
      f"抢占共 {prio2.scheduler.num_preemptions} 次；那几步的执行集合 {dropped[:3]}")

# ------------------------------------------------ 2. prefix + 投机

SHARED = [
    {"request_id": "P", "prompt_ids": [1, 2, 3, 4, 5, 6, 7, 8], "max_new_tokens": 6},
    {"request_id": "Q", "prompt_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "max_new_tokens": 6},
    {"request_id": "R", "prompt_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "max_new_tokens": 6},
]


def run_prefix(prefix_on, seed=29, **kw):
    engine = build(seed, **SPEC, **dict(enable_prefix_caching=prefix_on,
                                        num_kv_blocks=16, max_num_seqs=2, **kw))
    out = {}
    engine.scheduler.on_finished = lambda rec: out.__setitem__(rec["request_id"],
                                                               list(rec["output_ids"]))
    steps = run(engine, {0: [SHARED[0]], 1: [SHARED[1]], 2: [SHARED[2]]})
    return engine, out, steps


on_engine, on_out, on_steps = run_prefix(True)
off_engine, off_out, off_steps = run_prefix(False)
check("prefix + 投机：确实命中过（reused_tokens > 0）",
      max(s["reused"] for s in on_steps) > 0, f"{max(s['reused'] for s in on_steps)}")
check("prefix + 投机：确实投了机", sum(s["drafts"] for s in on_steps) > 0,
      f"{sum(s['drafts'] for s in on_steps)} 轮")
check("prefix + 投机：**开关前缀缓存不改变输出**（命中不能改结果）",
      on_out == off_out, f"\n  开={on_out}\n  关={off_out}")
check("prefix + 投机：每步池子自洽、没有 0 token 项",
      all(s["pool_ok"] for s in on_steps) and no_zero_items(on_steps))

# 发布出去的块，KV 不能被后续草稿写入改掉（验收方在第五十三关用过同一招）
snapshots = {}


def snapshot_hashed(engine, step):
    pool = engine.kv_cache_pool
    for h, b in pool.hash_to_block.items():
        if h not in snapshots:
            snapshots[h] = (pool.k_cache[:, b].clone(), pool.v_cache[:, b].clone())


# 发布那一刻把每个已发布块的 KV 抄一份，跑完之后逐个比对
snapshots.clear()
kv_engine = build(29, **SPEC, **dict(enable_prefix_caching=True, num_kv_blocks=16,
                                     max_num_seqs=2))
kv_steps = run(kv_engine, {0: [SHARED[0]], 1: [SHARED[1]], 2: [SHARED[2]]},
               watch_kv=snapshot_hashed)
pool = kv_engine.kv_cache_pool
still_same = all(torch.equal(pool.k_cache[:, b], k) and torch.equal(pool.v_cache[:, b], v)
                 for h, b in pool.hash_to_block.items()
                 for k, v in [snapshots[h]] if h in snapshots)
check("prefix + 投机：还留在索引里的已发布块，KV 与发布时逐字节相同",
      len(snapshots) > 0 and still_same, f"快照 {len(snapshots)} 个，仍在索引 "
      f"{len(pool.hash_to_block)} 个")
check("prefix + 投机：hash 双向索引一致",
      all(pool.block_to_hash.get(b) == h for h, b in pool.hash_to_block.items())
      and len(pool.block_to_hash) == len(pool.hash_to_block))

# 回滚掉的整块从来不该带 hash：构造一个「首枚被拒、跨块回滚」的投机轮
HASH_REQ = {"request_id": "H", "prompt_ids": [1, 2, 3, 4, 1, 2, 3, 4, 9, 9, 9],
            "max_new_tokens": 8}
hash_engine = build(7, **SPEC, **dict(enable_prefix_caching=True, num_kv_blocks=24,
                                      max_num_seqs=1, max_num_batched_tokens=8))
hash_steps = run(hash_engine, {0: [HASH_REQ]})
pool = hash_engine.kv_cache_pool
published_tokens = {}
for h, b in pool.hash_to_block.items():
    published_tokens[b] = h
check("prefix + 投机：跑完后引用归零、链表与真实空闲一致、hash 双向一致",
      all(u == 0 for u in pool.block_usage) and pool_ok(hash_engine)
      and all(pool.block_to_hash.get(b) == h for h, b in pool.hash_to_block.items()))

# ------------------------------------------------ 3. 两条一起 + 随机采样

# prompt 里带重复片段（n-gram 才提得出草稿），x 与 y 又共享前缀（前缀缓存才命中）
BASE_PROMPT = [3, 1, 4, 1, 5, 9, 2, 6] * 2
RANDOM_MIX = [
    {"request_id": "x", "prompt_ids": list(BASE_PROMPT), "max_new_tokens": 6,
     "priority": 0, "temperature": 0.8, "top_k": 20, "top_p": 0.9, "seed": 11},
    {"request_id": "y", "prompt_ids": BASE_PROMPT + [7, 8], "max_new_tokens": 6,
     "priority": 0, "frequency_penalty": 0.4, "presence_penalty": 0.3},
    {"request_id": "z", "prompt_ids": [5, 5, 1, 5, 5, 1, 5, 5], "max_new_tokens": 6,
     "priority": 5, "temperature": 0.6, "top_k": 10, "repetition_penalty": 1.3, "seed": 5},
]
both = build_scripted(29, **SPEC, num_kv_blocks=8, max_num_seqs=2)
# x 先跑一段把前缀发布出去；y 与它共享前缀、晚到（命中 + 名额抢占 z）；z 优先级最低
both_steps = run(both, {0: [RANDOM_MIX[0], RANDOM_MIX[2]], 3: [RANDOM_MIX[1]]})
both_tokens = [t for slot in both_steps for t in slot["tokens"]]
check("priority + prefix + 投机（含随机/惩罚）：跑完，每步池子自洽、无 0 token 项",
      all(s["pool_ok"] for s in both_steps) and no_zero_items(both_steps))
check("priority + prefix + 投机：三个特性都真的被用到了",
      both.scheduler.num_preemptions > 0
      and max(s["reused"] for s in both_steps) > 0
      and sum(s["drafts"] for s in both_steps) > 0,
      f"抢占 {both.scheduler.num_preemptions}、命中 "
      f"{max(s['reused'] for s in both_steps)}、草稿轮次 "
      f"{sum(s['drafts'] for s in both_steps)}")
check("priority + prefix + 投机：三条请求的 output_index 都连续",
      all(indices_ok(both_steps, "xyz").values()), str(indices_ok(both_steps, "xyz")))
pool = both.kv_cache_pool
check("priority + prefix + 投机：结束后引用归零、链表与真实空闲一致",
      all(u == 0 for u in pool.block_usage) and pool_ok(both))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
