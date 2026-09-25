"""第 51 关：增量维护的已提交历史（需求 §验收 1）。

单请求层面验证：
  - 初始化 / 连续追加 1 个、多个 token / prompt-output 边界切片；
  - 四条不变量；
  - 只读视图不能被直接 append、赋值（改内容只有 append_output_ids 一条路）；
  - 采样事件、EOS、最终 output_ids 与 step50 保持原样。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step50 import Engine as Engine50
from step51 import Engine as Engine51
from step51.cache import KVCachePool, _stable_hash
from step51.request import SequenceConfig

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def invariants(seq):
    a, p, o = seq.all_token_ids, seq.prompt_ids, seq.output_ids
    return (len(a) == len(p) + len(o)
            and list(a[:len(p)]) == p
            and list(a[len(p):]) == list(o)
            and 0 <= seq.cache.length <= len(a))


# ------------------------------------------------ 1. 初始化与追加

source_prompt = [1, 2, 3, 4, 5]
s = SequenceConfig("A", source_prompt, 8, 4)
source_prompt.append(99)
check("初始化：all == prompt，output 为空",
      list(s.all_token_ids) == [1, 2, 3, 4, 5] and list(s.output_ids) == []
      and invariants(s))
check("初始化：prompt_ids 是独立副本（外部改原列表不影响请求）",
      s.prompt_ids == [1, 2, 3, 4, 5] and list(s.all_token_ids) == [1, 2, 3, 4, 5])

s.append_output_ids(7)
check("追加单个 int：两份列表同步", list(s.output_ids) == [7]
      and list(s.all_token_ids) == [1, 2, 3, 4, 5, 7] and invariants(s))
s.append_output_ids([8, 9, 10])
check("追加多个 int：两份列表同步", list(s.output_ids) == [7, 8, 9, 10]
      and list(s.all_token_ids) == [1, 2, 3, 4, 5, 7, 8, 9, 10] and invariants(s))
s.append_output_ids([])
check("追加空列表：什么都不变", list(s.output_ids) == [7, 8, 9, 10] and invariants(s))

# ------------------------------------------------ 2. prompt/output 边界切片

check("边界切片：prompt 段", list(s.all_token_ids[:5]) == [1, 2, 3, 4, 5])
check("边界切片：output 段", list(s.all_token_ids[5:]) == [7, 8, 9, 10])
check("边界切片：跨边界一段", list(s.all_token_ids[3:8]) == [4, 5, 7, 8, 9])
check("边界切片：单个位置", s.all_token_ids[5] == 7 and s.all_token_ids[-1] == 10)
check("切片返回的是普通 list（可继续当列表用）",
      type(s.all_token_ids[3:8]) is list)

# ------------------------------------------------ 3. 只读视图

for bad in ("append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse"):
    check(f"视图没有 {bad}()", not hasattr(s.output_ids, bad))
for name in ("output_ids", "all_token_ids"):
    try:
        setattr(s, name, [])
        check(f"{name} 不能被赋值", False)
    except AttributeError:
        check(f"{name} 不能被赋值", True)
check("视图支持 len / in / 迭代 / == list",
      len(s.output_ids) == 4 and 7 in s.output_ids and list(iter(s.output_ids)) == [7, 8, 9, 10]
      and s.output_ids == [7, 8, 9, 10])
check("视图支持与 list 相加（既有脚本会这么写）",
      s.prompt_ids + s.output_ids == [1, 2, 3, 4, 5, 7, 8, 9, 10]
      and s.output_ids + [99] == [7, 8, 9, 10, 99])

# 切片不复制整段历史：只取一小段时不该碰其余部分（用长度巨大的历史验证耗时）
big = SequenceConfig("big", list(range(32)), 100, 16)
big.append_output_ids([7] * 8192)
import time
t0 = time.perf_counter_ns()
for _ in range(2000):
    big.all_token_ids[-1:]
dt = (time.perf_counter_ns() - t0) / 2000 / 1000
check("8192 长度的历史切最后 1 个：耗时与历史长度无关（< 5 μs）", dt < 5, f"{dt:.2f} μs")
check("大历史下不变量仍成立", invariants(big))

# prompt 的最后 1 个 token 和刚生成的第 1 个 token 合成一个完整块。
pool = KVCachePool(4, 4, 1, 1, torch.device("cpu"),
                   enable_prefix_caching=True, num_layers=1, over_subscribe=True)
cross = SequenceConfig("cross", [1, 2, 3], 2, 4)
assert pool.allocate_block(cross) and pool.ensure_blocks(cross, 3)
cross.cache.length = 3
pool.publish_computed_blocks(cross)
check("跨 prompt/output 的块未写满时不发布", not cross.block_hashes)
cross.append_output_ids(9)
pool.publish_computed_blocks(cross)  # 9 尚未进入模型
check("刚采样的 token 尚未写 KV，不提前发布", not cross.block_hashes)
assert pool.ensure_blocks(cross, 1)
cross.cache.length = 4
pool.publish_computed_blocks(cross)
expected_hash = _stable_hash(b"", (1, 2, 3, 9))
check("跨 prompt/output 的完整块按真实历史发布",
      cross.block_hashes == [expected_hash]
      and pool.hash_to_block[expected_hash] == cross.cache.block_table[0])

# ------------------------------------------------ 4. 端到端行为与 step50 一致

DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])
LOADS = [
    ("普通 decode", dict(max_num_seqs=2, max_num_batched_tokens=8, block_size=4,
                        num_kv_blocks=16, enable_prefix_caching=False)),
    ("chunked prefill", dict(max_num_seqs=1, max_num_batched_tokens=8, block_size=4,
                             num_kv_blocks=64, enable_prefix_caching=False)),
    ("prefix 命中 + 跨 prompt/output 的完整块",
     dict(max_num_seqs=2, max_num_batched_tokens=8, block_size=4, num_kv_blocks=16,
          enable_prefix_caching=True)),
    ("容量压力 + 抢占后重算", dict(max_num_seqs=2, max_num_batched_tokens=8, block_size=4,
                                  num_kv_blocks=4, enable_prefix_caching=True)),
]
REQS = [("a", list(range(1, 7)), 8), ("b", list(range(7, 13)), 8),
        ("c", list(range(1, 7)), 6), ("d", [20], 5)]      # c 与 a 同前缀

for label, kw in LOADS:
    outs, events = [], []
    for cls in (Engine50, Engine51):
        torch.manual_seed(29)
        e = cls(device="cpu", preemption_mode="recompute", scheduling_policy="fcfs", **kw, **DIMS)
        got, ev = {}, []
        e.on_token = lambda x: ev.append((x["request_id"], x["token_id"], x["output_index"]))
        for rid, ids, mn in REQS:
            e.add_request({"request_id": rid, "prompt_ids": list(ids), "max_new_tokens": mn})
        n = 0
        while e.has_unfinished_requests():
            e.step()
            n += 1
            assert n < 400, "疑似活锁"
            for rec in e.scheduler.step_done:
                got[rec["request_id"]] = rec["output_ids"]
        outs.append(got)
        events.append(ev)
    check(f"{label}：输出与 step50 逐 token 相同", outs[0] == outs[1],
          "" if outs[0] == outs[1] else f"\n  50={outs[0]}\n  51={outs[1]}")
    check(f"{label}：on_token 事件序列相同（采样事件未改）", events[0] == events[1])
    check(f"{label}：输出记录里是普通 list（对外接口未变）",
          all(type(v) is list for v in outs[1].values()))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
