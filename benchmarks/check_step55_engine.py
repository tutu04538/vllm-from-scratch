"""第 55 关的回归套件：单请求与多请求等价性 + 目录加载入口。

分五块：
  1. **脚本模型**：包住真模型，KV / 位置推进照常由真实现算，只把 logits 换成
     脚本给定的 token。于是「目标模型会输出什么」完全可控，能造出草稿全中、
     首枚拒绝、部分接受、草稿本身是 EOS 这些例子，并逐轮断言提交了什么。
  2. **单请求真模型等价性**：开着 n-gram 投机跑出来的文本必须与普通贪心
     **逐 token 相同**——接受也好拒绝也好，提交的永远是目标模型自己的贪心 token。
  3. **回滚与不变量**：跨块边界的回滚；结束后活动引用归零；重算统计按真实起点算，
     不拿 `num_scheduled_tokens` 反推（回滚后的 cache.length 不是本轮的终点）。
  4. **请求内容校验**：空 prompt / 非整数 token / 越界 / 负 max_new_tokens 都在入口拒绝。
  5. **多请求等价性（本关新增）**：4 条并发时每个 request_id 的最终 token 序列
     与普通贪心相同；output_index 连续、on_finished 恰好一次；CUDA 同样成立。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import step51
import step55

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


class ScriptedModel:
    """把真模型包起来：状态推进照旧，logits 换成脚本里的 token。

    `script` 是「每次采样依次返回哪些 token」的队列。这样目标模型的输出完全可控，
    而 `cache.length`、块表、位置仍然是真实现算的——回滚要验的正是这些真实状态。
    """

    def __init__(self, inner, script):
        self.inner = inner
        self.script = list(script)
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _forward_append(self, input_ids, num_scheduled_tokens, past_kv, kv_cache_pool,
                        sample_rows=None):
        self.calls += 1
        self.inner._forward_append(input_ids, num_scheduled_tokens, past_kv, kv_cache_pool,
                                   sample_rows)
        num_rows = len(input_ids) if sample_rows is None else len(sample_rows)
        tokens = [self.script.pop(0) for _ in range(num_rows)]
        logits = torch.zeros((num_rows, self.vocab_size))
        for row, token_id in enumerate(tokens):
            logits[row, token_id] = 1.0
        return logits


def build(pkg, seed, script, **cfg):
    """同一个 seed 造出同一个内层模型，外面套脚本模型。"""
    base = dict(max_num_seqs=1, max_num_batched_tokens=8, block_size=4, num_kv_blocks=16,
                enable_prefix_caching=False, scheduling_policy="fcfs")
    base.update(cfg)
    torch.manual_seed(seed)
    inner = pkg.TinyCausalLM(device="cpu", attention_backend="torch",
                             max_num_query_tokens=base["max_num_batched_tokens"], **DIMS)
    return pkg.Engine(model=ScriptedModel(inner, script), **base)


def run(engine, requests, limit=200, on_token=None):
    """跑到没有未完成请求。

    返回 (每轮提交的 token 列表, 完成记录)。每个元素是
    `{"tokens": [...], "drafts": [[...], ...]}`——「本轮提交了什么」与
    「本轮提了什么草稿」放在一起，正好能看出被拒绝的草稿没被提交。
    """
    committed, finished, per_step = [], [], []
    seen_ids = []

    def record(ev):
        committed.append(ev["token_id"])
        seen_ids.append(ev["request_id"])
        if on_token is not None:
            on_token(ev)

    engine.on_token = record
    engine.scheduler.on_finished = lambda rec: finished.append(
        dict(rec, output_ids=list(rec["output_ids"])))
    for request in requests:
        engine.add_request(dict(request))
    steps = 0
    while engine.has_unfinished_requests():
        before = len(committed)
        engine.step()
        steps += 1
        assert steps < limit, "疑似活锁"
        per_step.append({"tokens": committed[before:],
                         "drafts": [it["draft_ids"] for it in engine.scheduler.scheduled_items
                                    if it.get("draft_ids")],
                         "items": list(engine.scheduler.scheduled_items),
                         "state": [(q.request_id, q.cache.length if q.cache else None,
                                    q.high_water, q.recomputed_tokens)
                                   for q in engine.scheduler.running]})
    return per_step, finished, seen_ids


def pool_ok(pool):
    out, cur = [], pool.block_next[pool._SENTINEL_HEAD]
    while cur != pool._SENTINEL_TAIL:
        out.append(cur)
        cur = pool.block_next[cur]
    return (len(out) == len(set(out)) and set(out) == set(pool._allocatable_block_indices())
            and pool.num_allocatable == len(out)
            and all(u == 0 for u in pool.block_usage)
            and not pool.hash_to_block)


SPEC = dict(speculative_mode="ngram", num_speculative_tokens=2, prompt_lookup_n=2)
PROMPT = [1, 2, 3, 4, 1, 2, 3, 4]
REQUEST = {"request_id": "S", "prompt_ids": PROMPT, "max_new_tokens": 6}

# ------------------------------------------------ 1. 脚本模型：四种验证结果

# 预填充后采样 1；下一轮 n-gram 从历史里找到 (4,1) 的续写 [2,3]，脚本让目标模型
# 给出 [2,3,7] → 全部接受，一次 forward 提交 3 枚。
# 期望轨迹：step1 提交 1；step2 提交 [2,3,7]；之后 (3,7) 找不到匹配，逐步各 1 枚。
ALL_ACCEPT_SCRIPT = [1, 2, 3, 7, 5, 6]
engine = build(step55, 11, ALL_ACCEPT_SCRIPT, **SPEC)
per_step, finished, _ = run(engine, [REQUEST])
check("全部接受：一次目标 forward 提交 3 枚 token",
      [slot["tokens"] for slot in per_step] == [[1], [2, 3, 7], [5], [6]],
      str([slot["tokens"] for slot in per_step]))
check("全部接受：那一轮提的就是 [2,3]，且都进了提交",
      per_step[1]["drafts"] == [[2, 3]])
check("全部接受：on_finished 只发一次，内容是完整输出",
      len(finished) == 1 and finished[0]["output_ids"] == [1, 2, 3, 7, 5, 6])
check("触及 max_new_tokens 的那一轮不多发（正好 6 枚，一个不多）",
      [t for slot in per_step for t in slot["tokens"]] == [1, 2, 3, 7, 5, 6])

plain = build(step51, 11, ALL_ACCEPT_SCRIPT)
plain_step, _, _ = run(plain, [REQUEST])
check("同样的目标模型、同样的文本：投机比普通贪心少调用目标模型",
      [t for slot in plain_step for t in slot["tokens"]] == [1, 2, 3, 7, 5, 6]
      and engine.model.calls < plain.model.calls,
      f"投机 {engine.model.calls} 次 / 普通 {plain.model.calls} 次目标 forward")

# 首枚拒绝：提交 [t0]，本轮输入只留 [x]
engine = build(step55, 11, [1, 9, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], **SPEC)
per_step, _, _ = run(engine, [REQUEST])
check("首枚拒绝：草稿提出来了，但那一轮只提交 t0",
      per_step[1]["drafts"] == [[2, 3]] and per_step[1]["tokens"] == [9],
      str(per_step[1]))
check("首枚拒绝：那一轮真正进模型的是 1+2=3 个 token，起点记在计划里",
      per_step[1]["items"][0]["num_scheduled_tokens"] == 3
      and per_step[1]["items"][0]["start_cache_length"] == 8
      and per_step[1]["items"][0]["can_sample"] is True)
check("首枚拒绝：结束后引用归零、链表完整、无残留 hash", pool_ok(engine.kv_cache_pool))

# 部分接受：提交 [d0, t1]
engine = build(step55, 11, [1, 2, 9, 5, 6, 7, 8, 9, 10, 11], **SPEC)
per_step, _, _ = run(engine, [REQUEST])
check("部分接受：提交 [d0, t1]，第三枚草稿与后面的都不提交",
      per_step[1]["drafts"] == [[2, 3]] and per_step[1]["tokens"] == [2, 9],
      str(per_step[1]))

# 草稿本身是 EOS：只提交它，它之前的草稿才留 KV
EOS_PROMPT = [7, 8, 63, 9, 10, 7]     # 历史里 (7,8) 后面紧跟 63
seen, keep_len = [], []


def watch(ev):
    seq = engine.scheduler.running[0]
    seen.append(ev["token_id"])
    keep_len.append(seq.cache.length)


engine = build(step55, 11, [8, 63, 9, 1], **SPEC)
per_step, finished, _ = run(engine, [{"request_id": "E", "prompt_ids": EOS_PROMPT,
                                      "max_new_tokens": 5}], on_token=watch)
check("草稿首枚就是 EOS：提的是 [63,9]，但只提交到 EOS 就停",
      per_step[1]["drafts"] == [[63, 9]] and per_step[1]["tokens"] == [63]
      and seen == [8, 63], f"drafts={per_step[1]['drafts']} tokens={per_step[1]['tokens']}")
# 第一枚是预填充那一轮提交的（cache.length 刚推到 6）；第二枚是投机那一轮提交的，
# 此时已经回滚过——只留 [x]=8 一个输入的 KV，所以停在 6+1=7。若按「草稿都接受」
# 保留，这里会是 9；若一个都不留，会是 6。
check("草稿是终止 token：只保留它之前那些草稿的 KV（投机那轮提交时停在 7）",
      keep_len == [6, 7], f"提交时的 cache.length={keep_len}")
check("草稿是终止 token：EOS 之后的草稿与 bonus 都不提交、不回调",
      len(finished) == 1 and finished[0]["output_ids"] == [8, 63])
check("草稿是终止 token：结束后池子干净", pool_ok(engine.kv_cache_pool))

# 找不到草稿：退回普通 1-token 路径
engine = build(step55, 11, [9, 8, 7, 6, 5, 4], **dict(SPEC, prompt_lookup_n=8))
per_step, _, _ = run(engine, [REQUEST])
check("找不到草稿：每一轮都按普通 1-token 路径走",
      all(not slot["drafts"] for slot in per_step)
      and [t for slot in per_step for t in slot["tokens"]] == [9, 8, 7, 6, 5, 4])

# 预算只够 1 个 token：K 被预算压成 0
engine = build(step55, 11, [1, 9, 8, 7, 6, 5], **dict(SPEC, max_num_batched_tokens=1))
per_step, _, _ = run(engine, [REQUEST])
check("token 预算只有 1：草稿长度为 0，退回普通路径",
      all(not slot["drafts"] for slot in per_step)
      and [t for slot in per_step for t in slot["tokens"]] == [1, 9, 8, 7, 6, 5]
      and all(it["num_scheduled_tokens"] == 1 for slot in per_step for it in slot["items"]))

# 输出上限只剩 1 个：K <= R-1 把它压成 0
engine = build(step55, 11, [1, 2, 3, 4, 5], **SPEC)
per_step, _, _ = run(engine, [dict(REQUEST, max_new_tokens=1)])
check("max_new_tokens=1：只剩一个输出额度，不投机（全部接受还要多一个 bonus）",
      all(not slot["drafts"] for slot in per_step))

# 上下文将满：没有任何一轮越过 max_seq_len
engine = build(step55, 11, [1, 2, 3, 7, 5, 6, 7, 8], **dict(SPEC, max_seq_len=13))
per_step, finished, _ = run(engine, [REQUEST])
check("上下文将满：没有任何一轮越过 max_seq_len",
      all(it["start_cache_length"] + it["num_scheduled_tokens"] <= 13
          for slot in per_step for it in slot["items"]))
check("上下文将满：输出仍然正确（被压短的只是草稿）",
      finished[0]["output_ids"] == [1, 2, 3, 7, 5, 6], str(finished[0]["output_ids"]))

# ------------------------------------------------ 2. 真模型：与普通贪心逐 token 相同

REAL_PROMPT = [3, 1, 4, 1, 5, 9, 2, 6] * 3     # 重复片段多，n-gram 更容易命中
REAL_REQUEST = {"request_id": "R", "prompt_ids": REAL_PROMPT, "max_new_tokens": 12}
REAL_CFG = dict(max_num_seqs=1, max_num_batched_tokens=8, block_size=4, num_kv_blocks=16,
                enable_prefix_caching=False, scheduling_policy="fcfs")


def real_run(pkg, seed, **cfg):
    base = dict(REAL_CFG)
    base.update(cfg)
    torch.manual_seed(seed)
    engine = pkg.Engine(device="cpu", attention_backend="torch", **base, **DIMS)
    per_step, finished, _ = run(engine, [REAL_REQUEST])
    return engine, finished[0]["output_ids"], per_step


plain_engine, plain_out, plain_steps = real_run(step51, 29)
spec_engine, spec_out, spec_steps = real_run(step55, 29, **SPEC)
# 每个「排出了 token」的 step 恰好调用一次 _forward_append
plain_forwards = sum(1 for slot in plain_steps if slot["items"])
spec_forwards = sum(1 for slot in spec_steps if slot["items"])
drafts = [d for slot in spec_steps for d in slot["drafts"]]
check("真模型：投机跑出来的文本与 step51 普通贪心逐 token 相同（接受与否都不改结果）",
      spec_out == plain_out, f"\n  普通={plain_out}\n  投机={spec_out}")
check("真模型：这次确实提出过草稿（不是「没投机所以相等」）",
      len(drafts) > 0, f"提出草稿的轮数={len(drafts)}，草稿={drafts[:4]}")
multi = [slot["tokens"] for slot in spec_steps if len(slot["tokens"]) > 1]
check("真模型：至少有一轮一次目标 forward 提交多枚 token",
      len(multi) >= 3 and all(len(tokens) == 3 for tokens in multi), str(multi))
check("真模型：同样的文本，投机少调用目标模型",
      spec_forwards < plain_forwards,
      f"投机 {spec_forwards} 次 / 普通 {plain_forwards} 次目标 forward")
check("真模型：结束后池子干净", pool_ok(spec_engine.kv_cache_pool))

# ------------------------------------------------ 3. 回滚与重算统计

engine = build(step55, 11, [1, 9, 5, 6, 7, 8, 9, 10], **SPEC)
per_step, _, _ = run(engine, [REQUEST])
item = per_step[1]["items"][0]
check("回滚后的那一刻：cache.length 与高水位都是「起点 + 只保留的 1 个输入」",
      per_step[1]["state"] == [("S", 9, 9, 0)],
      str(per_step[1]["state"]))
# 这一步进了 3 个 token、只保留 1 个。按 (end - num_scheduled_tokens) 反推起点会
# 得到 9-3=6，与旧高水位 8 的重叠是 2 —— 凭空记出 2 个「重算」。按计划里记下的
# start_cache_length=8 算才是 0。
check("重算统计按计划里的真实起点算，不拿 num_scheduled_tokens 反推",
      per_step[1]["state"][0][3] == 0
      and item["start_cache_length"] == 8 and item["num_scheduled_tokens"] == 3)
check("投机本身不产生重算量，也没有抢占（本关不支持抢占）",
      item["request"].num_preemptions == 0)

# 需求 §3.D 说「本关禁用抢占」。删掉 preemption_mode 之后这条**不再是开关**，
# 而是推论：max_num_seqs=1 + fcfs 让两个抢占入口都没有候选（名额没人可顶、
# 尾部没有更靠后的犠牲者）。所以直接把池子压到可行性下限，看它守不守得住。
TIGHT_PROMPT = [3, 1, 4, 1, 5, 9, 2, 6] * 2
TIGHT_NEW = 12
TIGHT_NEED = -(-(len(TIGHT_PROMPT) + TIGHT_NEW - 1) // 4)     # 池子的可行性下限
torch.manual_seed(29)
tight = step55.Engine(device="cpu", attention_backend="torch", max_num_seqs=1,
                      max_num_batched_tokens=8, block_size=4, num_kv_blocks=TIGHT_NEED,
                      enable_prefix_caching=False, scheduling_policy="fcfs", **SPEC, **DIMS)
tight.on_token = lambda ev: None
tight_steps, tight_finished, _ = run(tight, [{"request_id": "T", "prompt_ids": TIGHT_PROMPT,
                                              "max_new_tokens": TIGHT_NEW}])
check(f"投机 + 池子压到可行性下限（{TIGHT_NEED} 块）：一次抢占都没有，重算量恒为 0",
      tight.scheduler.num_preemptions == 0
      and all(it["request"].recomputed_tokens == 0
              for slot in tight_steps for it in slot["items"])
      and len(tight_finished) == 1 and len(tight_finished[0]["output_ids"]) == TIGHT_NEW,
      f"抢占 {tight.scheduler.num_preemptions} 次")
torch.manual_seed(29)
too_small = step55.Engine(device="cpu", attention_backend="torch", max_num_seqs=1,
                          max_num_batched_tokens=8, block_size=4,
                          num_kv_blocks=TIGHT_NEED - 1, enable_prefix_caching=False,
                          scheduling_policy="fcfs", **SPEC, **DIMS)
too_small.on_token = lambda ev: None
_, rejected, _ = run(too_small, [{"request_id": "T2", "prompt_ids": TIGHT_PROMPT,
                                  "max_new_tokens": TIGHT_NEW}])
check("池子比下限还小：请求被明确拒绝（InfeasibleRequest），不是卡死",
      len(rejected) == 1 and "永远无法完成" in (rejected[0].get("error") or ""),
      str(rejected))

# 跨块边界的回滚：反复申请又归还
engine = build(step55, 11, [1, 9, 8, 7, 6, 5, 4, 3], **dict(SPEC, num_kv_blocks=8))
per_step, _, _ = run(engine, [REQUEST])
check("跨块边界反复回滚：每一步的块表长度都与当时的 cache.length 相符",
      all(it["request"].cache is None
          or len(it["request"].cache.block_table) == -(-it["request"].cache.length // 4)
          for slot in per_step for it in slot["items"]))
check("跨块边界反复回滚：结束后池子干净", pool_ok(engine.kv_cache_pool))

# ------------------------------------------------ 4. 请求内容校验（入口就报错）


def reject(**overrides):
    """新引擎上试着入队一条请求，返回错误信息（没报错就返回 None）。"""
    engine = build(step55, 11, [1, 2, 3, 4, 5, 6], **SPEC)
    request = {"request_id": "X", "prompt_ids": [1, 2, 3], "max_new_tokens": 4}
    request.update(overrides)
    try:
        engine.add_request(request)
    except ValueError as exc:
        return str(exc)
    return None


check("空 prompt：入口直接拒绝（不再靠零进展守卫兜底）",
      "不能为空" in (reject(prompt_ids=[]) or ""), reject(prompt_ids=[]))
check("空 prompt + max_new_tokens=0 也拒绝：空历史本身就不合法",
      reject(prompt_ids=[], max_new_tokens=0) is not None)
check("max_new_tokens 为负：入口直接拒绝（以前会静默丢掉这条请求）",
      "不能为负" in (reject(max_new_tokens=-1) or ""), reject(max_new_tokens=-1))
check("max_new_tokens=0 仍然合法：只算 prompt、不生成",
      reject(max_new_tokens=0) is None)
check("prompt 里有浮点：拒绝，不再静默截断成整数",
      "必须是整数" in (reject(prompt_ids=[1, 2.5, 3]) or ""), reject(prompt_ids=[1, 2.5, 3]))
check("prompt 越界：拒绝（以前是 embedding 里的 IndexError）",
      "超出词表范围" in (reject(prompt_ids=[1, 5, 999]) or ""), reject(prompt_ids=[1, 5, 999]))
check("prompt 里有负数：拒绝",
      "超出词表范围" in (reject(prompt_ids=[1, -5]) or ""), reject(prompt_ids=[1, -5]))
check("bool 不是 token id，也不是 max_new_tokens",
      reject(prompt_ids=[True, 2]) is not None and reject(max_new_tokens=True) is not None)
check("prompt_ids 不可迭代：拒绝（以前是 TypeError）",
      reject(prompt_ids=None) is not None and reject(prompt_ids=5) is not None)
check("字符串 prompt_ids 按元素类型拦下（'abc' 不是三个 token）",
      reject(prompt_ids="abc") is not None, reject(prompt_ids="abc"))

# 删掉 num_uncomputed == 0 那一半之后：预算再紧也不能排出 0 token 的幽灵计划项，
# 否则 _check_progress 会把它当成「本轮有进展」，零进展守卫就废了。
for budget in (1, 2):
    ghost_engine = build(step55, 11, [1, 2, 3, 4, 5, 6], **dict(SPEC, max_num_batched_tokens=budget))
    ghost, _, _ = run(ghost_engine, [dict(REQUEST, max_new_tokens=4)])
    check(f"预算={budget}：没有 0 token 的计划项（num_uncomputed==0 的兜底已可省）",
          all(it["num_scheduled_tokens"] >= 1 for slot in ghost for it in slot["items"]))

# ------------------------------------------------ 5. 多请求：与普通贪心逐请求等价

# 需求 §5.C：同权重、同请求，比较**每个 request_id 的最终 token 序列**。
# 多请求投机每步输出数不同，所以不要求逐步轨迹或跨请求完成顺序相同。
# prompt 都带重复片段，n-gram 才提得出草稿（随机小模型下这不是自动成立的：
# 换一组 prompt 就可能一轮草稿都没有，所以下面还要断言「确实提过草稿」）
MULTI_REQS = [
    {"request_id": "R0", "prompt_ids": [3, 1, 4, 1, 5, 9, 2, 6] * 3, "max_new_tokens": 8},
    {"request_id": "R1", "prompt_ids": [7, 7, 1] * 4, "max_new_tokens": 8},
    {"request_id": "R2", "prompt_ids": [2, 4, 6, 8] * 3, "max_new_tokens": 8},
    {"request_id": "R3", "prompt_ids": [5, 5, 5, 5, 1, 2] * 2, "max_new_tokens": 8},
]


def multi_run(pkg, seed, device="cpu", dtype=torch.float32, spec=False):
    torch.manual_seed(seed)
    cfg = dict(max_num_seqs=4, max_num_batched_tokens=16, block_size=4, num_kv_blocks=32,
               enable_prefix_caching=False, scheduling_policy="fcfs")
    if spec:
        cfg.update(speculative_mode="ngram", num_speculative_tokens=2, prompt_lookup_n=2)
    engine = pkg.Engine(device=device, attention_backend="torch", dtype=dtype, **cfg, **DIMS)
    outs, events, finished, multi = {}, [], [], 0
    engine.on_token = lambda ev: events.append(
        (ev["request_id"], ev["token_id"], ev["output_index"]))
    engine.scheduler.on_finished = lambda rec: finished.append(rec["request_id"])
    for r in MULTI_REQS:
        engine.add_request(dict(r))
    n = 0
    while engine.has_unfinished_requests():
        engine.step()
        n += 1
        assert n < 200, "疑似活锁"
        for rec in engine.scheduler.step_done:
            outs[rec["request_id"]] = list(rec["output_ids"])
        for it in engine.scheduler.scheduled_items:
            multi += it["can_sample"] and len(it.get("draft_ids", [])) > 0
    return engine, outs, events, finished, multi


_, spec_outs, spec_events, spec_finished, spec_rounds = multi_run(step55, 29, spec=True)
_, plain_outs, plain_events, plain_finished, _ = multi_run(step51, 29, spec=False)
check("多请求：投机与普通贪心的**每个 request_id 最终 token 序列**相同",
      spec_outs == plain_outs and set(spec_outs) == {r["request_id"] for r in MULTI_REQS},
      f"\n  普通={plain_outs}\n  投机={spec_outs}")
check("多请求：这次真的在投机（有若干轮提出了草稿），不是走 fallback 的假通过",
      spec_rounds >= 3, f"提出草稿的轮数={spec_rounds}")
for rid in sorted(spec_outs):
    indices = [idx for r, _, idx in spec_events if r == rid]
    check(f"多请求：{rid} 的 output_index 连续、on_finished 恰好一次",
          indices == list(range(len(indices))) and spec_finished.count(rid) == 1,
          f"indices={indices} finished={spec_finished.count(rid)}")
LIMITS = {r["request_id"]: r["max_new_tokens"] for r in MULTI_REQS}
check("多请求：每条请求都产生了输出，且没有超过自己的 max_new_tokens",
      all(0 < len(v) <= LIMITS[rid] for rid, v in spec_outs.items()),
      str({rid: len(v) for rid, v in spec_outs.items()}))

if torch.cuda.is_available():
    for dtype in (torch.float32, torch.bfloat16):
        _, cuda_spec, _, _, cuda_rounds = multi_run(step55, 29, device="cuda",
                                                    dtype=dtype, spec=True)
        _, cuda_plain, _, _, _ = multi_run(step55, 29, device="cuda", dtype=dtype, spec=False)
        check(f"CUDA/{dtype}：多请求投机与普通贪心逐 request_id 相同"
              f"（提出草稿 {cuda_rounds} 轮）",
              cuda_spec == cuda_plain, f"\n  普通={cuda_plain}\n  投机={cuda_spec}")

# ------------------------------------------------ 6. 目录加载入口

# 这一节是补的：把加载流程从 engine.py 搬到 loading.py 时，`from_model_dir()` 里
# 还留着一处 `read_raw_config(...)` / `_load_weights_into(...)` 的调用没跟着改，
# 而当时所有 step55 脚本都是 `Engine(model=...)` 随机初始化，**没有一条走到过
# `from_model_dir`**，所以谁都没发现。补上这条端到端加载，以后就漏不掉了。
import tempfile

from step55.formats.native import save_model

tmpdir = tempfile.mkdtemp(prefix="step54_load_")
torch.manual_seed(7)
scratch = step55.TinyCausalLM(device="cpu", attention_backend="torch",
                              max_num_query_tokens=8, **DIMS)
save_model(scratch, tmpdir)

loaded = step55.Engine.from_model_dir(tmpdir, device="cpu", max_num_seqs=2,
                                 max_num_batched_tokens=16, block_size=4, num_kv_blocks=32,
                                 enable_prefix_caching=False, **SPEC)
loaded_out = []
loaded.scheduler.on_finished = lambda rec: loaded_out.extend(rec["output_ids"])
loaded.add_request({"request_id": "L", "prompt_ids": [1, 2, 3, 4, 1, 2, 3, 4],
                    "max_new_tokens": 6})
loaded_step = 0
while loaded.has_unfinished_requests():
    loaded.step()
    loaded_step += 1
    assert loaded_step < 60
check("from_model_dir：目录加载 + 投机跑通（这条能抓到加载流程搬家的漏改）",
      len(loaded_out) == 6 and loaded_step > 0, str(loaded_out))

# 同权重随机初始化的引擎应当给出同一串输出（加载没有悄悄改模型）
torch.manual_seed(7)
plain_same = step55.Engine(device="cpu", attention_backend="torch", max_num_seqs=2,
                      max_num_batched_tokens=16, block_size=4, num_kv_blocks=32,
                      enable_prefix_caching=False, **SPEC, **DIMS)
plain_out = []
plain_same.scheduler.on_finished = lambda rec: plain_out.extend(rec["output_ids"])
plain_same.add_request({"request_id": "L", "prompt_ids": [1, 2, 3, 4, 1, 2, 3, 4],
                        "max_new_tokens": 6})
while plain_same.has_unfinished_requests():
    plain_same.step()
check("from_model_dir：读回来的权重与保存前一致（输出逐 token 相同）",
      loaded_out == plain_out, f"\n  加载={loaded_out}\n  同权重={plain_out}")

from step55 import load_model_config, load_model_weights          # 公开导出
from step55.engine import load_model_config as engine_load        # 老 import 路径
from step55.loading import load_model_from_dir
check("加载入口的公开 import 路径都在（包级 + engine 重导出）",
      callable(load_model_config) and callable(load_model_weights)
      and callable(engine_load) and callable(load_model_from_dir))
check("load_model_config 读得出内部配置",
      load_model_config(tmpdir)["vocab_size"] == DIMS["vocab_size"])

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
