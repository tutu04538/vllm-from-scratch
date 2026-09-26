"""第 54 关：批量投机验证——行映射、提交顺序、预算与容量压力。

分四块：
  1. 行映射本身：拿需求 §2 那张表手工造 `scheduled_items`，直接验 `_sample_plan()`；
  2. 混批：同一步里有不同 K 的投机项、无草稿的普通 decode、中间 prefill，
     检查模型收到的一次 forward、sample_rows、筛选后偏移、提交顺序与事件序号；
  3. 预算压力：预算不够全部 ready 时「真实 token 优先」，不 assert、不排 0 token 项；
  4. 容量压力：先缩草稿、再抢占；被抢占的草稿计划整个作废，且**不会成为抢占别人的理由**。

脚本模型（`ScriptedModel`）包住真模型：KV、位置、块表照常由真实现推进，
只把 logits 换成脚本给定的 token，于是「目标模型会输出什么」完全可控。
"""

import sys
from collections import Counter

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import step54
from step54 import Engine as Engine53

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


DIMS = dict(vocab_size=64, d_model=16, max_seq_len=128, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


class ScriptedModel:
    """包住真模型：状态推进照旧，logits 换成脚本里的 token。

    另外记录每次 forward 的 `(input_ids, sample_rows, num_scheduled_tokens)`，
    行映射对不对就看这里——不是看引擎自己报的数字。
    """

    def __init__(self, inner, script=None):
        """script=None 表示「复制模式」：每个采样行返回它**自己的输入 token**。

        复制模式下序列自带重复片段，n-gram 几乎总能提出草稿、而且草稿必然被接受
        （t0 == d0、t1 == d1），适合压力/恢复这类只关心「投机一直在发生」的用例；
        需要精确控制接受/拒绝时再给一份脚本。
        """
        self.inner = inner
        self.script = None if script is None else list(script)
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _forward_append(self, input_ids, num_scheduled_tokens, past_kv, kv_cache_pool,
                        sample_rows=None):
        self.calls.append({"input_ids": list(input_ids),
                           "num_scheduled_tokens": list(num_scheduled_tokens),
                           "sample_rows": None if sample_rows is None else list(sample_rows)})
        self.inner._forward_append(input_ids, num_scheduled_tokens, past_kv, kv_cache_pool,
                                   sample_rows)
        rows = list(range(len(input_ids))) if sample_rows is None else list(sample_rows)
        if self.script is None:
            tokens = [int(input_ids[row]) for row in rows]
        else:
            tokens = [self.script.pop(0) for _ in rows]
        num_rows = len(rows)
        logits = torch.zeros((num_rows, self.vocab_size))
        for row, token_id in enumerate(tokens):
            logits[row, token_id] = 1.0
        return logits


def build(seed, script, **cfg):
    base = dict(max_num_seqs=4, max_num_batched_tokens=16, block_size=4, num_kv_blocks=64,
                enable_prefix_caching=False, scheduling_policy="fcfs",
                speculative_mode="ngram", num_speculative_tokens=2, prompt_lookup_n=2)
    base.update(cfg)
    torch.manual_seed(seed)
    inner = step54.TinyCausalLM(device="cpu", attention_backend="torch",
                                max_num_query_tokens=base["max_num_batched_tokens"], **DIMS)
    return Engine53(model=ScriptedModel(inner, script), **base)


def pool_ok(engine):
    """每步都能查的池子不变量：引用计数、可分配链、链表成员三者互相一致。"""
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


def run(engine, requests, arrivals=None, limit=80):
    """跑完。每一步记下提交事件、执行了的计划项、池子是否自洽。"""
    arrivals = arrivals or {}
    for r in arrivals.get(0, []):
        engine.add_request(dict(r))
    per_step = []
    step = 0
    while engine.has_unfinished_requests():
        committed = []
        engine.on_token = lambda ev: committed.append(
            (ev["request_id"], ev["token_id"], ev["output_index"]))
        engine.step()
        step += 1
        assert step < limit, "疑似活锁"
        for r in arrivals.get(step, []):
            engine.add_request(dict(r))
        per_step.append({"tokens": committed,
                         "items": list(engine.scheduler.scheduled_items),
                         "budget_used": sum(it["num_scheduled_tokens"]
                                            for it in engine.scheduler.scheduled_items),
                         "preemptions": engine.scheduler.num_preemptions,
                         "pool_ok": pool_ok(engine),
                         # 每个请求当时的 (KV 进度, 已提交输出数, 被抢占次数, 阻塞者)
                         "states": {q.request_id: (
                             q.cache.length if q.cache is not None else None,
                             len(q.output_ids), q.num_preemptions,
                             q.resume_blocker.request_id if q.resume_blocker else None)
                             for q in list(engine.scheduler.running)
                             + list(engine.scheduler.waiting)}})
    return per_step


# ------------------------------------------------ 1. 行映射（需求 §2 那张表）

A, B, C, D = (step54.SequenceConfig(rid, [1], 4, 4) for rid in "ABCD")
manual = [
    {"request": A, "num_scheduled_tokens": 3, "draft_ids": [11, 12], "can_sample": True},
    {"request": B, "num_scheduled_tokens": 4, "draft_ids": [], "can_sample": False},
    {"request": C, "num_scheduled_tokens": 1, "draft_ids": [], "can_sample": True},
    {"request": D, "num_scheduled_tokens": 2, "draft_ids": [13], "can_sample": True},
]
rows, picked = Engine53._sample_plan(manual)
check("需求 §2 那张表：sample_rows = [0,1,2,7,8,9]（中间 prefill 占输入位置但不取行）",
      rows == [0, 1, 2, 7, 8, 9], str(rows))
check("筛选后偏移：A=[0:3]、C=[3:4]、D=[4:6]",
      [it["sample_offset"] for it in picked] == [0, 3, 4]
      and [it["num_sample_rows"] for it in picked] == [3, 1, 2],
      str([(it["sample_offset"], it["num_sample_rows"]) for it in picked]))
check("中间 prefill 不进 picked（不采样）",
      [it["request"].request_id for it in picked] == ["A", "C", "D"])
check("原始行号与筛选偏移是两个坐标系：B 占了原始行 3..6，所以 C 的原始行是 7、偏移是 3",
      rows[3] == 7 and picked[1]["sample_offset"] == 3)


# ------------------------------------------------ 2. 混批端到端

# A：历史里 (4,1) 的续写是 [2,3] → K=2
# C：prompt 里没有重复片段 → 找不到草稿 → K=0 的普通 decode
# D：prompt 结尾是两个相同 token，采样一枚同值后末尾三连 → n-gram 只给 1 枚 → K=1
# B：长 prompt，同一轮里只算一段（中间 prefill）。FCFS 下它最后到达，所以排在批尾——
#    「prefill 夹在两条 decode 中间」那种排列靠到达顺序造不出来（没预算的总是排在
#    后面的那些），第 1 节的单元用例已经按需求 §2 那张表把中间位置覆盖了。
REQ_A = {"request_id": "A", "prompt_ids": [1, 2, 3, 4, 1, 2, 3, 4], "max_new_tokens": 6}
REQ_C = {"request_id": "C", "prompt_ids": [11, 21, 31, 41, 51, 61, 12, 22], "max_new_tokens": 4}
REQ_D = {"request_id": "D", "prompt_ids": [9, 9], "max_new_tokens": 4}
REQ_B = {"request_id": "B", "prompt_ids": [41 + (i % 20) for i in range(30)], "max_new_tokens": 2}

# 脚本 = 每步 sample_rows 顺序上的目标 token：
#   step1：A(1 行)=1、C(1 行)=10、D(1 行)=9     三个 prefill 跑完并各采样一枚
#   step2：A(3 行)=2,3,7（草稿全中）、C(1 行)=10、D(2 行)=9,5（草稿全中）
#   之后随便喂，用例只看 step2
engine = build(11, [1, 10, 9,
                    2, 3, 7, 10, 9, 5] + [0] * 40, max_num_batched_tokens=20)
per_step = run(engine, None, arrivals={0: [REQ_A, REQ_C, REQ_D], 1: [REQ_B]})
calls = engine.model.calls
step1, step2 = per_step[0], per_step[1]

check("混批：第 1 步三条 prefill 同批跑完，各取末行采样",
      [(it["request"].request_id, it["num_scheduled_tokens"]) for it in step1["items"]]
      == [("A", 8), ("C", 8), ("D", 2)]
      and calls[0]["sample_rows"] == [7, 15, 17],
      str(calls[0]["sample_rows"]))
check("混批：第 2 步的计划是 A=1+2、C=1+0、D=1+1、B=中间 prefill",
      [(it["request"].request_id, it["num_scheduled_tokens"], len(it["draft_ids"]))
       for it in step2["items"]] == [("A", 3, 2), ("C", 1, 0), ("D", 2, 1), ("B", 14, 0)],
      str([(it["request"].request_id, it["num_scheduled_tokens"], it["draft_ids"])
           for it in step2["items"]]))
check("混批：一轮只调用一次模型，输入是全部计划项拼起来的一维",
      len(calls) == len(per_step)
      and calls[1]["input_ids"] == [t for it in step2["items"] for t in it["input_ids"]]
      and calls[1]["num_scheduled_tokens"] == [3, 1, 2, 14],
      str(calls[1]["num_scheduled_tokens"]))
check("混批：sample_rows 跳过批尾 prefill 的 14 行，只剩 A/C/D 的行",
      calls[1]["sample_rows"] == [0, 1, 2, 3, 4, 5], str(calls[1]["sample_rows"]))
check("混批：筛选后偏移 A=[0:3]、C=[3:4]、D=[4:6]（B 不采样）",
      [(it["sample_offset"], it["num_sample_rows"]) for it in step2["items"]
       if it["can_sample"]] == [(0, 3), (3, 1), (4, 2)],
      str([(it["request"].request_id, it.get("sample_offset"), it.get("num_sample_rows"))
           for it in step2["items"]]))
check("混批：同一步里三种 K 都出现（2 / 0 / 1 / 无草稿）",
      [len(it["draft_ids"]) for it in step2["items"]] == [2, 0, 1, 0]
      and [it["can_sample"] for it in step2["items"]] == [True, True, True, False])
check("混批：同一步里既有全部接受（A 三枚、D 两枚）也没有被拒绝的草稿",
      [t for t in step2["tokens"] if t[0] == "A"] == [("A", 2, 1), ("A", 3, 2), ("A", 7, 3)]
      and [t for t in step2["tokens"] if t[0] == "D"] == [("D", 9, 1), ("D", 5, 2)],
      str(step2["tokens"]))
check("混批：提交顺序按 picked（A 的三枚、C、D 的两枚），不是「先 plain 后 drafts」",
      [t[0] for t in step2["tokens"]] == ["A", "A", "A", "C", "D", "D"],
      str(step2["tokens"]))
check("混批：B 是中间 prefill 的那一轮不发 token、也不重放 on_token",
      all(t[0] != "B" for t in step2["tokens"])
      and not any(it["can_sample"] for it in step2["items"]
                  if it["request"].request_id == "B"))

all_tokens = [t for slot in per_step for t in slot["tokens"]]
for rid in "ACDB":
    indices = [idx for r, _, idx in all_tokens if r == rid]
    check(f"混批：{rid} 的 output_index 连续",
          indices == list(range(len(indices))), str(indices))

pool = engine.kv_cache_pool
chain, cur = [], pool.block_next[pool._SENTINEL_HEAD]
while cur != pool._SENTINEL_TAIL:
    chain.append(cur)
    cur = pool.block_next[cur]
check("混批：结束后活动引用归零、可分配链与真实空闲一致",
      all(u == 0 for u in pool.block_usage)
      and set(chain) == set(pool._allocatable_block_indices())
      and not pool.hash_to_block)

# ------------------------------------------------ 3. 预算压力

# 预算比 ready 数量还少：**真实 token 优先**，按 running 顺序保留靠前的各 1 个，
# 其余的这轮不运行。以前这里会 assert「Decode token budget exceeds
# max_num_batched_tokens」——那是把可配置的组合当成不可能发生的事。
budget_engine = build(7, [0] * 200, max_num_seqs=4, max_num_batched_tokens=2)
budget_steps = run(budget_engine, None, arrivals={0: [
    {"request_id": rid, "prompt_ids": [1, 2, 3, 4], "max_new_tokens": 4}
    for rid in "ABCD"]})
check("预算 2 / 4 条 ready：每轮不超预算、没有 0 token 项、也不 assert",
      all(slot["budget_used"] <= 2 for slot in budget_steps)
      and all(len(slot["items"]) <= 2 for slot in budget_steps)
      and all(it["num_scheduled_tokens"] >= 1 for slot in budget_steps
              for it in slot["items"]),
      str([(slot["budget_used"], len(slot["items"])) for slot in budget_steps[:4]]))
check("预算压力：每一轮都不超预算，四条请求都正常跑完",
      all(slot["budget_used"] <= 2 for slot in budget_steps)
      and len(budget_steps[-1]["tokens"]) >= 0
      and not budget_engine.has_unfinished_requests())
check("预算压力：结束时池子干净", pool_ok(budget_engine))

# 预算只够一条 ready 的 1 个真实 token：草稿拿不到额度，K 自然为 0
tight_budget = build(11, [1, 2] + [0] * 40,
                     max_num_batched_tokens=8)
tight_steps = run(tight_budget, None, arrivals={0: [
    {"request_id": "A", "prompt_ids": [1, 2, 3, 4, 1, 2, 3, 4], "max_new_tokens": 4},
    {"request_id": "B", "prompt_ids": [11, 21, 31, 41, 51, 61, 12, 22], "max_new_tokens": 4}]})
check("预算够 2 条真实 token、剩 6 给草稿：A 先到先得，B 的草稿拿不到额度",
      [(it["request"].request_id, len(it["draft_ids"]))
       for it in tight_steps[1]["items"]] == [("A", 2), ("B", 0)]
      and tight_steps[1]["budget_used"] <= 8,
      str([(it["request"].request_id, it["num_scheduled_tokens"], it["draft_ids"])
           for it in tight_steps[1]["items"]]))

# ------------------------------------------------ 4. 容量压力：先缩草稿、再抢占

# 池子刚好被两条请求占满（各 2 块），链上一个可分配块都没有。
#   step2：A 的草稿要多占一块 -> 缩 K 到 0，**不抢占 B**（真实 token 还装得下）
#   step3：A 的真实 token 也要新块了 -> 才走 _make_room，抢占 B
# B 在第 3 步的**计划里已经有草稿**，被抢占后整个 item 必须作废。
PRESSURE_ARGS = dict(max_num_seqs=4, max_num_batched_tokens=16, num_kv_blocks=4)
A_REQ = {"request_id": "A", "prompt_ids": [1, 2, 3, 3, 5, 6, 3], "max_new_tokens": 6}
# 这一段用**复制模式**的脚本模型（每行返回它自己的输入 token）：序列自带重复片段，
# n-gram 一直在提草稿、而且草稿必然全中，正适合「缩草稿 / 抢占 / 恢复」这种只关心
# 「投机持续在发生」的场景。B 的 prompt 以两个相同 token 结尾，复制模式下历史末尾
# 必然是三连，第 3 步一定能提出草稿（这样才有「被抢占的 item 带着草稿计划」这一幕）。
B_REQ = {"request_id": "B", "prompt_ids": [11, 21, 31, 41, 51, 41, 41], "max_new_tokens": 6}

plans = []
_original_plan_tokens = step54.scheduler.Scheduler._plan_tokens


def _spy(self):
    planned = _original_plan_tokens(self)
    plans.append({it["request"].request_id: list(it["draft_ids"]) for it in planned})
    return planned


step54.scheduler.Scheduler._plan_tokens = _spy
try:
    pressure = build(3, None, **PRESSURE_ARGS)
    pressure_steps = run(pressure, None, arrivals={0: [A_REQ, B_REQ]})
finally:
    step54.scheduler.Scheduler._plan_tokens = _original_plan_tokens

check("容量压力：每一步的引用计数、可分配链、链表成员都自洽",
      all(slot["pool_ok"] for slot in pressure_steps))
check("容量压力：第 2 步 A 的草稿装不下 -> 缩到 K=0，**一次抢占都没有**",
      plans[1]["A"] == [5, 6]                            # 计划里本来有 2 枚草稿
      and [it["draft_ids"] for it in pressure_steps[1]["items"]] == [[], []]
      and pressure_steps[1]["preemptions"] == 0,
      f"计划={plans[1]} 执行={[it['draft_ids'] for it in pressure_steps[1]['items']]} "
      f"抢占={pressure_steps[1]['preemptions']}")
check("容量压力：第 3 步真实 token 也装不下 -> 才抢占，且被抢占的是排在后面的 B",
      pressure_steps[2]["preemptions"] == 1
      and [it["request"].request_id for it in pressure_steps[2]["items"]] == ["A"],
      str([it["request"].request_id for it in pressure_steps[2]["items"]]))
check("容量压力：被抢占的 B 那一轮**计划里已经有草稿**，但整个 item 没执行",
      plans[2]["B"] == [41]
      and [it["request"].request_id for it in pressure_steps[2]["items"]] == ["A"],
      f"计划={plans[2]} 执行={[it['request'].request_id for it in pressure_steps[2]['items']]}")
check("容量压力：B 那一轮没有进模型、没有发 token（整个 item 作废）",
      all(t[0] != "B" for t in pressure_steps[2]["tokens"])
      and len(pressure.model.calls[2]["input_ids"]) == pressure_steps[2]["budget_used"],
      str(pressure_steps[2]["tokens"]))

# 恢复：B 的历史与输出原样保留，KV 从重算进度追赶，追上后能继续投机
b_seq = next(it["request"] for slot in pressure_steps for it in slot["items"]
             if it["request"].request_id == "B")
check("恢复：被抢占那一刻 KV 进度归零、已提交输出原样保留、记下阻塞者",
      pressure_steps[2]["states"]["B"] == (0, 2, 1, "A"),
      str(pressure_steps[2]["states"]["B"]))
check("恢复：阻塞者没结束之前一直不重建 KV（队首被拦住，不白算一遍）",
      pressure_steps[2]["states"]["B"][0] == 0
      and pressure_steps[3]["states"]["B"][0] == 0
      and "B" not in [it["request"].request_id for it in pressure_steps[3]["items"]])
check("恢复：已提交历史与 prompt 一字不差，算到的位置也记着（high_water）",
      list(b_seq.prompt_ids) == B_REQ["prompt_ids"]
      and b_seq.high_water >= len(b_seq.prompt_ids))
check("重算：can_sample 为假的那一轮不采样、不重放 on_token（整段运行都成立）",
      all(all(t[0] != it["request"].request_id for t in slot["tokens"])
          for slot in pressure_steps for it in slot["items"] if not it["can_sample"]))
preempt_index = next(i for i, slot in enumerate(pressure_steps) if slot["preemptions"] >= 1)
replay_index = next(i for i in range(preempt_index + 1, len(pressure_steps))
                    if any(it["request"].request_id == "B" and it["num_scheduled_tokens"] > 1
                           for it in pressure_steps[i]["items"]))
check("恢复：重算那一轮把整段历史一次追平（KV 进度 = prompt 长度 + 之前提交的输出数）",
      pressure_steps[replay_index]["states"]["B"][0]
      == len(B_REQ["prompt_ids"]) + pressure_steps[replay_index - 1]["states"]["B"][1],
      f"重算那一步 cache.length="
      f"{pressure_steps[replay_index]['states']['B'][0]}，"
      f"上一步输出数={pressure_steps[replay_index - 1]['states']['B'][1]}")
check("恢复：追上之后马上又能投机（草稿又出现了）",
      any(len(it["draft_ids"]) > 0
          for slot in pressure_steps[replay_index + 1:]
          for it in slot["items"] if it["request"].request_id == "B"),
      str([(i, [it["draft_ids"] for it in slot["items"]
                if it["request"].request_id == "B"])
           for i, slot in enumerate(pressure_steps[replay_index + 1:], replay_index + 1)]))
check("容量压力：结束后活动引用归零、可分配链与真实空闲一致", pool_ok(pressure))
check("容量压力：重算统计记的是它真的重放过的那段历史",
      b_seq.recomputed_tokens > 0
      and b_seq.high_water >= len(b_seq.prompt_ids)
      and b_seq.recomputed_tokens <= b_seq.high_water,
      f"recomputed={b_seq.recomputed_tokens} high_water={b_seq.high_water}")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
