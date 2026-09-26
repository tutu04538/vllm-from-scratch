"""第 55 关：双模型的两套 KV（需求 §3、§6B）。

这一关最容易错的地方全在两条 KV 的**对齐**上，所以这里用**脚本模型**把轨迹钉死：
两个模型的前向照常跑（KV 真的写进各自的物理块），但每行的 argmax 由一个纯函数
`f(输入 token, 被预测的绝对位置) -> token` 给定。请求全是贪心（q 与 p 都是 one-hot），
于是「第几枚草稿被接受」完全确定，不用碰运气。

检查三件事：

1. **逐步轨迹**：每轮记录两套 `cache.length`、块表与下一轮真实输入——全接受时
   draft 必须比 target 少 1（最后一枚被接受的草稿还没进 draft 的 KV），下一轮
   提议前必须先补上；部分接受时两边必须齐平。不能重复喂 x，也不能漏算草稿。
2. **两套 KV 的正确性**（本关最重要的一条）：把两套 KV 的**有效部分**与「该模型对
   已提交前缀单独重算一遍」的结果逐个张量比对。只查长度/引用是不够的——被拒草稿的
   KV 留在块里、块号又恰好对得上时，长度照样是对的。
3. **边界**：草稿 EOS、跨块回滚、多请求不同 K、draft 池紧张后退回 target 路径。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import step55
from step55 import Engine
from step55.cache import CacheConfig, KVCachePool
from step55.request import SequenceConfig

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


TARGET_DIMS = dict(vocab_size=64, d_model=32, num_q_heads=4, num_kv_heads=2,
                   num_layers=3, intermediate_size=64, head_dim=8, max_seq_len=128,
                   eos_token_ids=[63])
DRAFT_DIMS = dict(vocab_size=64, d_model=16, num_q_heads=2, num_kv_heads=1,
                  num_layers=1, intermediate_size=32, head_dim=8, max_seq_len=128,
                  eos_token_ids=[63])
PROMPT = [1, 2, 3, 1, 2, 3]


class Scripted:
    """包住真模型：状态推进照旧，logits 由 `f(输入 token, 被预测的位置)` 给定。

    `f` 是**纯函数**，不消费任何队列，所以脚本不会「跑偏到耗尽」——出错时看到的是
    数值不符，而不是一个与本题无关的 `pop from empty list`。
    """

    def __init__(self, inner, f):
        self.inner = inner
        self.f = f

    def __getattr__(self, name):          # 其余属性/方法一律转发（max_seq_len 等）
        return getattr(self.inner, name)

    def _forward_append(self, input_ids, num_scheduled_tokens, past_kv, kv_cache_pool,
                        sample_rows=None):
        # 先照常跑：KV 真的写进给定的池子，position 与块表也照常前进
        starts = [c.length for c in past_kv]
        self.inner._forward_append(input_ids, num_scheduled_tokens, past_kv, kv_cache_pool,
                                   sample_rows)
        rows = list(range(len(input_ids))) if sample_rows is None else list(sample_rows)
        logits = torch.zeros((len(rows), self.vocab_size))
        # 打包行 -> (哪条请求, 请求内的第几行)，据此还原「被预测的绝对位置」
        row_owner, row_index = [], []
        for index, (start, count) in enumerate(zip(starts, num_scheduled_tokens)):
            row_owner.extend([index] * count)
            row_index.extend(range(count))
        for out_row, row in enumerate(rows):
            owner, index = row_owner[row], row_index[row]
            logits[out_row, self.f(int(input_ids[row]), starts[owner] + index + 1)] = 8.0
        return logits


def build(seed, target_f, draft_f, **cfg):
    base = dict(max_num_seqs=2, max_num_batched_tokens=16, block_size=4, num_kv_blocks=32,
                draft_num_kv_blocks=16, enable_prefix_caching=False, scheduling_policy="fcfs",
                speculative_mode="draft_model", num_speculative_tokens=2, prompt_lookup_n=2,
                device="cpu", attention_backend="torch")
    base.update(cfg)
    torch.manual_seed(seed)
    target = step55.TinyCausalLM(max_num_query_tokens=base["max_num_batched_tokens"],
                                 device="cpu", attention_backend="torch", **TARGET_DIMS)
    torch.manual_seed(seed + 1)
    draft = step55.TinyCausalLM(max_num_query_tokens=base["max_num_batched_tokens"],
                                device="cpu", attention_backend="torch", **DRAFT_DIMS)
    draft_arg = {} if base.pop("with_draft", True) else {"draft_model": None}
    if not draft_arg:
        draft_arg = {"draft_model": Scripted(draft, draft_f)}
    return Engine(model=Scripted(target, target_f), **draft_arg, **base)


def copy_mode(token, position):
    """复制模式：模型预测它刚吃进去的那个 token。draft 提什么、target 就同意什么。"""
    return token


def steps(engine, requests, arrivals=None, limit=40):
    """跑到没有未完成请求为止，逐步记录计划与请求状态。

    `requests` 是第 0 步到达的请求；`arrivals` 是后面第 N 步到达的（动态到达）。
    """
    for request in requests:
        engine.add_request(dict(request))
    arrivals = arrivals or {}
    for request in arrivals.get(0, []):
        engine.add_request(dict(request))

    # 提交那一刻的状态：那时两套 KV 都已经回滚/对齐完，而 post_step 还没判停、没回收
    # （请求跑完之后 `seq.cache` 会被置 None，只能在这里抓）
    committed = {}

    def on_token(event):
        for seq in engine.scheduler.running:
            if seq.request_id == event["request_id"]:
                committed[event["request_id"]] = (seq.cache.length, seq.draft_cache.length,
                                                  list(seq.output_ids))
                # 不变量：draft 绝不领先于 target 的真实计算边界（每一枚提交时刻都查）
                assert seq.draft_cache.length <= seq.cache.length, \
                    f"{seq.request_id}: draft={seq.draft_cache.length} > " \
                    f"target={seq.cache.length}" 

    engine.on_token = on_token
    trace, step = [], 0
    while engine.has_unfinished_requests():
        committed.clear()
        engine.step()
        step += 1
        assert step < limit, "疑似活锁"
        for request in arrivals.get(step, []):
            engine.add_request(dict(request))
        trace.append({
            "step": step,
            "plans": [(it["num_scheduled_tokens"], list(it["input_ids"]),
                       list(it["draft_ids"]), it["max_draft_k"])
                      for it in engine.scheduler.scheduled_items],
            "blocks": {seq.request_id: (len(seq.cache.block_table or []),
                                        len(seq.draft_cache.block_table or []))
                       for seq in live} if (live := engine.scheduler.running) else {},
            "reused": max((seq.reused_tokens for seq in live), default=0),
            "commit": dict(committed),
        })
    return trace


def spec_rounds(trace):
    """只留下真的投了机的那些步。"""
    return [slot for slot in trace if any(plan[2] for plan in slot["plans"])]


# ------------------------------------------------ 1. 全接受：draft 还差最后一枚

# 两个模型都用「整体 +1」模式：草稿与 x 不同，这样「下一轮有没有重复喂 x」才看得出来。
# 全接受时每轮提交 3 个 token：d0=x+1、d1=x+2、bonus=x+3。
shift = lambda token, position: token + 1
engine = build(1, shift, shift, num_speculative_tokens=2)
trace = steps(engine, [{"request_id": "A", "prompt_ids": PROMPT, "max_new_tokens": 6}])
rounds = spec_rounds(trace)
first = rounds[0]
# prompt 的最后一行预测 4（shift 模式），所以第一个输出 token 是 4，它就是这一轮的 x
check("全接受：第一轮投机真的发生，本轮输入 = [x, d0, d1]、K 枚草稿各带一行 q",
      first["plans"] == [(3, [4, 5, 6], [5, 6], 2)], str(first["plans"]))
target_len, draft_len, output = first["commit"]["A"]
check("全接受：target 追到 prompt 之后 3 个位置，**draft 只追到 2 个**"
      "（最后一枚被接受的草稿还没进 draft 的 KV，等下一轮补算）",
      target_len == len(PROMPT) + 3 and draft_len == len(PROMPT) + 2,
      f"target={target_len} draft={draft_len}")
check("全接受：提交的是 [d0, d1, bonus]",
      output == [4, 5, 6, 7], str(output))
second = rounds[1]
t2, d2, out2 = second["commit"]["A"]
check("全接受：第二轮从**上一轮最后一个提交的 token**（bonus=7）继续提议，不重复喂 x",
      second["plans"] == [(2, [7, 8], [8], 1)],
      f"第二轮输入 {second['plans'][0][1]}（x 应当是 7，不是上一轮的 4）")
check("全接受：补算把上一轮欠的那一枚补进了 draft（draft 追平后又提了 1 枚）",
      d2 == len(PROMPT) + 4 and t2 == d2 + 1, f"第二轮结束 target={t2} draft={d2}")

pool, draft_pool = engine.kv_cache_pool, engine.draft_kv_pool
check("跑完：两套池子的活动引用都归零",
      all(u == 0 for u in pool.block_usage) and all(u == 0 for u in draft_pool.block_usage),
      f"target 非零 {sum(1 for u in pool.block_usage if u)} 块、"
      f"draft 非零 {sum(1 for u in draft_pool.block_usage if u)} 块")

# ------------------------------------------------ 1b. 部分接受：两边应当齐平

# pos==8 那一行让 target 预测 d+1：第一枚接受、第二枚拒绝 -> target 只留 [x, d0]，
# draft 提议后正好也是 2 个位置，两边齐平（不需要夹）
reject_at_8 = lambda token, position: token + 1 if position == 8 else token
engine2 = build(2, reject_at_8, copy_mode, num_speculative_tokens=2)
trace2 = steps(engine2, [{"request_id": "A", "prompt_ids": PROMPT, "max_new_tokens": 6}])
rounds2 = spec_rounds(trace2)
first2 = rounds2[0]
target_len2, draft_len2, output2 = first2["commit"]["A"]
check("部分接受：接受 1 枚后拒绝（纠正 token 是 target 的 argmax=4），提交 [d0, c]",
      output2 == [3, 3, 4], str(output2))
check("部分接受：两边齐平——被拒的那枚草稿在 target 与 draft 上都退掉了",
      target_len2 == len(PROMPT) + 2 and draft_len2 == len(PROMPT) + 2,
      f"target={target_len2} draft={draft_len2}")

# ------------------------------------------------ 2. 有效 KV == 单独重算（强校验）

def gather_kv(pool, cache, length):
    """取出 `cache` 里 [0, length) 这些逻辑位置**实际**落在的 K/V。"""
    probe = CacheConfig(block_table=list(cache.block_table or []), length=0)
    slots = pool.build_slot_mapping([probe], [length])
    return pool.k_flat[:, slots].clone(), pool.v_flat[:, slots].clone()


def kv_diff(engine, seq, model_attr, pool_attr, length):
    """比对「模型现在这套 KV 的有效部分」与「它对同一段前缀单独重算」的结果。"""
    model = getattr(engine, model_attr)
    live_pool = getattr(engine, pool_attr)
    cache = seq.cache if model_attr == "model" else seq.draft_cache
    live = gather_kv(live_pool, cache, length)
    # 单独重算：同一份权重 + 全新池子 + 一次前向，只喂已提交前缀
    fresh = KVCachePool(engine.kv_cache_pool.block_size, length + 4, model.num_kv_heads,
                        model.head_dim, model.device, False, num_layers=model.num_layers,
                        dtype=model.dtype)
    ref_cache = CacheConfig()
    fresh.ensure_blocks_for(ref_cache, length)
    tokens = torch.tensor(list(seq.all_token_ids[:length]), dtype=torch.long,
                          device=model.device)
    model._forward_append(tokens, [length], [ref_cache], fresh, sample_rows=[])
    ref = gather_kv(fresh, ref_cache, length)
    return max(float((a - b).abs().max()) for a, b in zip(live, ref))


# 容差：同一个模型对同一段前缀做前向，**batch 形状不同**时 GEMM 的分块不同，
# 会有 FP32 级的舍入差（实测 ~3e-7，见下面的实测值）。所以按绝对容差比，
# 并配一条「故意改坏一个位置」的对照证明这个比对真的有判别力。
KV_ATOL = 1e-5

for label, target_f in (("全接受", copy_mode), ("部分接受", reject_at_8)):
    eng = build(5, target_f, copy_mode, num_speculative_tokens=2, max_num_seqs=1)
    eng.add_request({"request_id": "A", "prompt_ids": PROMPT, "max_new_tokens": 40})
    for _ in range(4):
        eng.step()
    seq = eng.scheduler.running[0]
    t_diff = kv_diff(eng, seq, "model", "kv_cache_pool", seq.cache.length)
    d_diff = kv_diff(eng, seq, "draft_model", "draft_kv_pool", seq.draft_cache.length)
    check(f"{label}：target 的有效 KV 与「对已提交前缀单独重算」一致（FP 噪声内）",
          t_diff <= KV_ATOL, f"最大逐元素差 {t_diff:.2e}")
    check(f"{label}：draft 的有效 KV 与单独重算一致（含补算进来的位置）",
          d_diff <= KV_ATOL,
          f"最大逐元素差 {d_diff:.2e}（draft 有效长度 {seq.draft_cache.length}）")

# 对照：把某个有效位置的 K 改掉一点点，同一段比对必须立刻炸——
# 否则上面那条 PASS 可能只是「比对本身没看东西」
seq = eng.scheduler.running[0]
probe = CacheConfig(block_table=list(seq.draft_cache.block_table), length=0)
slots = eng.draft_kv_pool.build_slot_mapping([probe], [seq.draft_cache.length])
eng.draft_kv_pool.k_flat[0, slots[2]] += 0.5
corrupted = kv_diff(eng, seq, "draft_model", "draft_kv_pool", seq.draft_cache.length)
check("对照：故意改坏一个有效位置的 KV，这条比对立刻失败（有判别力）",
      corrupted > 100 * KV_ATOL, f"改坏后最大差 {corrupted:.2e}")

# ------------------------------------------------ 3. 草稿 EOS

# 草稿在第二个位置提出 EOS：target 同意 -> 立即结束；target 不同意 -> 请求照常继续
# `% 64`：target 每一轮都会算满 K+1 行（bonus 行也在内，哪怕验证到 EOS 就停了），
# 而 bonus 行的输入正是那枚 EOS 草稿——脚本必须在 token=63 时也给得出合法 token
draft_eos = lambda token, position: 63 if position == 8 else (token + 1) % 64

accept_eos = build(7, draft_eos, draft_eos, num_speculative_tokens=2)
trace_eos = steps(accept_eos, [{"request_id": "A", "prompt_ids": PROMPT, "max_new_tokens": 8}])
round_eos = spec_rounds(trace_eos)[0]
check("草稿 EOS 被接受：提交到 EOS 为止（[d0, EOS]），本轮结束",
      round_eos["plans"] == [(3, [4, 5, 63], [5, 63], 2)]
      and round_eos["commit"]["A"][2] == [4, 5, 63],
      f"计划 {round_eos['plans']}，提交 {round_eos['commit']['A'][2]}")
check("草稿 EOS 被接受：draft 与 target 的长度一致（EOS 草稿的 KV 不留）",
      round_eos["commit"]["A"][0] == round_eos["commit"]["A"][1] == len(PROMPT) + 2,
      f"target={round_eos['commit']['A'][0]} draft={round_eos['commit']['A'][1]}")

reject_eos = build(8, lambda token, position: 9 if position == 8 else (token + 1) % 64,
                   draft_eos, num_speculative_tokens=2)
trace_rej = steps(reject_eos, [{"request_id": "A", "prompt_ids": PROMPT, "max_new_tokens": 8}])
first_rej = spec_rounds(trace_rej)[0]
rej_out = first_rej["commit"]["A"][2]
check("草稿 EOS 被拒绝：纠正 token 是 target 的 argmax（9），请求照常继续跑（不结束）",
      63 not in rej_out and rej_out[:3] == [4, 5, 9]
      and reject_eos.scheduler.has_unfinished_requests() is False,
      f"提交 {rej_out}（EOS 没进输出；请求由 max_new_tokens 正常结束）")

# ------------------------------------------------ 4. 多请求不同 K 与预算

# 预算 4、两条短 prompt 同一步就绪：先各留 1 个真实 token（共 2），余量 2 只够
# 一条请求提 2 枚草稿 -> 同一步里 K 不同（ragged batch），另一条照常走 1-token 路径
multi = build(9, shift, shift, max_num_batched_tokens=4, num_speculative_tokens=2,
              draft_num_kv_blocks=16, num_kv_blocks=32)
multi_trace = steps(multi, [
    {"request_id": "A", "prompt_ids": [1, 2], "max_new_tokens": 4},
    {"request_id": "B", "prompt_ids": [5, 6], "max_new_tokens": 4}])
first_multi = spec_rounds(multi_trace)[0]
check("同批不同 K：预算先保真实 token，余量只给一条请求 -> ragged batch（2 枚 / 0 枚）",
      [plan[3] for plan in first_multi["plans"]] == [2, 0]
      and [len(plan[2]) for plan in first_multi["plans"]] == [2, 0],
      str(first_multi["plans"]))
check("同批不同 K：拿到草稿的那条这一轮多提交了 3 枚，另一条只提交 1 枚",
      len(first_multi["commit"]["A"][2]) > len(first_multi["commit"]["B"][2]),
      f"A 累计 {first_multi['commit']['A'][2]}、B 累计 {first_multi['commit']['B'][2]}")
check("同批不同 K：两条请求都跑完",
      multi.scheduler.has_unfinished_requests() is False
      and len(multi_trace[-1]["commit"]["A"][2] if "A" in multi_trace[-1]["commit"]
              else multi_trace[-2]["commit"]["A"][2]) == 4)

# ------------------------------------------------ 5. 跨块回滚（拒绝掉一整个块）

# block_size=4、prompt 长 8：本轮 target 要算 [8, 13)（x + 4 枚草稿）-> 4 个块；
# 首枚就被拒 -> 只保留 [x] -> 回到 3 个块，最后那个块必须还回池子
prompt8 = [1, 2, 3, 4, 5, 6, 7, 8]
reject_first = lambda token, position: token + 5 if position == 9 else token + 1
cross = build(10, reject_first, shift, num_speculative_tokens=4, block_size=4, num_kv_blocks=16)
cross_trace = steps(cross, [{"request_id": "A", "prompt_ids": prompt8, "max_new_tokens": 6}])
round_cross = spec_rounds(cross_trace)[0]
check("跨块回滚：首枚被拒后 target 从 4 个块退到 3 个块（多占的整块真的还回去了）",
      round_cross["commit"]["A"][0] == len(prompt8) + 1
      and round_cross["blocks"]["A"][0] == 3,
      f"长度 {round_cross['commit']['A'][0]}、块数 {round_cross['blocks']['A'][0]}")
check("跨块回滚：draft 跟着夹到同一条边界",
      round_cross["commit"]["A"][1] == len(prompt8) + 1,
      f"draft={round_cross['commit']['A'][1]}")
check("跨块回滚：被拒的 4 枚草稿一枚都没进输出",
      round_cross["commit"]["A"][2] == [9, 14], str(round_cross["commit"]["A"][2]))

# ------------------------------------------------ 6. draft 池紧张：缩 K 与回退

# （a）连 prompt 都补不完的 draft 池：计划阶段就把 K 缩到 0，这条请求一直走普通路径
#     ——不能把一条本来能完成的 target 请求卡死，也不能为可选草稿去抢占别的请求
tight = build(11, shift, shift, draft_num_kv_blocks=1, num_speculative_tokens=2,
              num_kv_blocks=32)
tight_trace = steps(tight, [{"request_id": "A", "prompt_ids": PROMPT, "max_new_tokens": 8}])
check("draft 池放不下历史：计划阶段就把预留缩到 0（不排不可能的草稿）",
      all(plan[3] == 0 for slot in tight_trace for plan in slot["plans"]),
      str([slot["plans"] for slot in tight_trace][:3]))
check("draft 池放不下历史：请求照常跑完，没有活锁",
      tight.scheduler.has_unfinished_requests() is False
      and all(u == 0 for u in tight.draft_kv_pool.block_usage))

plain = build(11, shift, shift, speculative_mode=None, num_kv_blocks=32, with_draft=False)
plain_trace = steps(plain, [{"request_id": "A", "prompt_ids": PROMPT, "max_new_tokens": 8}])


def final_output(trace, request_id):
    return [slot["commit"][request_id][2] for slot in trace
            if request_id in slot["commit"]][-1]


check("draft 全程回退时，输出与纯 target 逐 token 相同",
      final_output(tight_trace, "A") == final_output(plain_trace, "A"),
      f"投机 {final_output(tight_trace, 'A')} / 纯 target {final_output(plain_trace, 'A')}")

# （b）运行时才不够：两条请求的计划**都**通过了只读容量检查，但第一条补算+提议吃掉
#     大部分池子，第二条当场补不动 -> 这一轮它没有草稿，target 一步都没耽误
#     （计划只读、执行才提交，所以这种「计划时够、执行时不够」真的会发生）
runtime_tight = build(12, shift, shift, draft_num_kv_blocks=3, num_speculative_tokens=2,
                      num_kv_blocks=32, max_num_batched_tokens=16)
runtime_trace = steps(runtime_tight, [
    {"request_id": "A", "prompt_ids": PROMPT, "max_new_tokens": 8},
    {"request_id": "B", "prompt_ids": list(PROMPT), "max_new_tokens": 8}])
fell_back = [slot for slot in runtime_trace
             if any(plan[3] > 0 and not plan[2] for plan in slot["plans"])]
check("draft 池运行时不够：真的走了回退（计划预留了草稿，却一枚都没提出来）",
      len(fell_back) > 0, f"{len(fell_back)} 步")
check("draft 池运行时不够：两条请求都照常跑完",
      runtime_tight.scheduler.has_unfinished_requests() is False
      and len(final_output(runtime_trace, "A")) == 8
      and len(final_output(runtime_trace, "B")) == 8,
      f"A={len(final_output(runtime_trace, 'A'))} B={len(final_output(runtime_trace, 'B'))}")
check("draft 池运行时不够：只影响草稿，target 的输出与纯 target 一致",
      final_output(runtime_trace, "A") == final_output(plain_trace, "A"),
      f"{final_output(runtime_trace, 'A')} vs {final_output(plain_trace, 'A')}")
check("draft 池运行时不够：结束时 draft 池引用归零",
      all(u == 0 for u in runtime_tight.draft_kv_pool.block_usage))

# ------------------------------------------------ 7. 抢占：两套 KV 一起回收、恢复后一致

# target 池只够两条请求各占一点：第三条更高优先级的请求到达时触发容量抢占。
# 被抢占的请求两套活动 KV 都要释放/失效，恢复后 draft 必须从已提交历史重新补算。
PRE = [1, 2, 3, 4, 1, 2, 3, 4]
pre = build(13, shift, shift, num_kv_blocks=6, draft_num_kv_blocks=16,
            num_speculative_tokens=2, max_num_seqs=3, scheduling_policy="priority",
            max_num_batched_tokens=16)
pre_trace = steps(pre, [
    {"request_id": "A", "prompt_ids": PRE, "max_new_tokens": 8, "priority": 0},
    {"request_id": "B", "prompt_ids": list(PRE), "max_new_tokens": 8, "priority": 0}])
check("抢占：构造出来了（容量压力下真的抢占过）",
      pre.scheduler.num_preemptions > 0, f"抢占 {pre.scheduler.num_preemptions} 次")

# priority + **动态到达**：一条严格更高优先级的请求中途插进来，应当顶掉低优先级的
# 名额（名额抢占），并且被顶掉那条的两套 KV 都要释放、之后能重新恢复
prio = build(19, shift, shift, num_kv_blocks=32, draft_num_kv_blocks=16,
             num_speculative_tokens=2, max_num_seqs=1, scheduling_policy="priority",
             max_num_batched_tokens=16)
prio_trace = steps(prio, [{"request_id": "low", "prompt_ids": PROMPT, "max_new_tokens": 10,
                           "priority": 0}],
                   arrivals={2: [{"request_id": "high", "prompt_ids": [7, 8, 7, 8],
                                  "max_new_tokens": 4, "priority": -1}]})
check("priority 动态到达：高优先级中途插进来，发生了**名额抢占**",
      prio.scheduler.num_priority_preemptions > 0,
      f"名额抢占 {prio.scheduler.num_priority_preemptions} 次")
check("priority 动态到达：两条请求都跑完、序号连续、两套池子归零",
      prio.scheduler.has_unfinished_requests() is False
      and len(final_output(prio_trace, "low")) == 10
      and len(final_output(prio_trace, "high")) == 4
      and all(u == 0 for u in prio.kv_cache_pool.block_usage)
      and all(u == 0 for u in prio.draft_kv_pool.block_usage),
      f"low={len(final_output(prio_trace, 'low'))} high={len(final_output(prio_trace, 'high'))}")
check("抢占：被抢占的请求两套 KV 都失效（draft 长度回到 0 并重新补算）",
      pre.draft_proposer.num_catchup_tokens > len(PRE),
      f"累计补算 {pre.draft_proposer.num_catchup_tokens} 个 token（> prompt 长度说明"
      f"至少重建过一次）")
check("抢占：跑完，两套池子的活动引用都归零",
      pre.scheduler.has_unfinished_requests() is False
      and all(u == 0 for u in pre.kv_cache_pool.block_usage)
      and all(u == 0 for u in pre.draft_kv_pool.block_usage))
check("抢占：两条请求的输出都是 8 个 token、序号连续",
      len(final_output(pre_trace, "A")) == 8 and len(final_output(pre_trace, "B")) == 8,
      f"A={len(final_output(pre_trace, 'A'))} B={len(final_output(pre_trace, 'B'))}")

# 恢复之后再跑一段，比对两套 KV 与「单独重算已提交前缀」
# 两条请求 + 6 块的小池子：争抢必然发生；被抢占后恢复，两套 KV 都要重新对齐
recover = build(14, shift, shift, num_kv_blocks=6, draft_num_kv_blocks=16,
                num_speculative_tokens=2, max_num_seqs=2)
for request_id in ("A", "B"):
    recover.add_request({"request_id": request_id, "prompt_ids": PROMPT, "max_new_tokens": 10})
seen_preemption = False
for _ in range(40):
    before = recover.scheduler.num_preemptions
    recover.step()
    seen_preemption = seen_preemption or recover.scheduler.num_preemptions > before
    if seen_preemption and recover.scheduler.running:
        break
seq = recover.scheduler.running[0] if recover.scheduler.running else None
check("抢占后恢复：抢占确实发生过（否则下面那条没测到东西）", seen_preemption)
check("抢占后恢复：两套 KV 与「对已提交前缀单独重算」一致",
      seq is not None
      and kv_diff(recover, seq, "model", "kv_cache_pool", seq.cache.length) <= KV_ATOL
      and kv_diff(recover, seq, "draft_model", "draft_kv_pool",
                  seq.draft_cache.length) <= KV_ATOL,
      f"target 长度 {seq.cache.length if seq else None}、"
      f"draft 长度 {seq.draft_cache.length if seq else None}")

# ------------------------------------------------ 8. target 前缀命中：draft 从真实历史追赶

# A 先跑出一段可复用的前缀并发布；B 带同样的前缀晚到 -> 命中（reused_tokens>0），
# 而 draft 池**不做**前缀共享，必须自己把那段历史补进自己的 KV
shared_prompt = [3, 1, 4, 1, 5, 9, 2, 6, 3, 1]
prefix = build(15, shift, shift, enable_prefix_caching=True, num_kv_blocks=32,
               draft_num_kv_blocks=16, num_speculative_tokens=2, max_num_seqs=2)
prefix_trace = steps(prefix, [{"request_id": "B", "prompt_ids": shared_prompt + [7, 8],
                               "max_new_tokens": 6}],
                     arrivals={3: [{"request_id": "C", "prompt_ids": shared_prompt,
                                    "max_new_tokens": 6}]})
reused = max(slot["reused"] for slot in prefix_trace)
check("前缀命中：C 真的借到了 A/B 已发布的块（reused_tokens > 0）", reused > 0,
      f"reused={reused}")
check("前缀命中：两条请求都跑完、序号连续",
      prefix.scheduler.has_unfinished_requests() is False
      and len(final_output(prefix_trace, "B")) == 6
      and len(final_output(prefix_trace, "C")) == 6,
      f"B={len(final_output(prefix_trace, 'B'))} C={len(final_output(prefix_trace, 'C'))}")
check("前缀命中：draft 累计补算的量覆盖了两条请求的完整历史（补算而不是读 target 的 KV）",
      prefix.draft_proposer.num_catchup_tokens >= 2 * len(shared_prompt),
      f"补算 {prefix.draft_proposer.num_catchup_tokens} 个 token")

# 命中那条请求跑到一半时，两套 KV 仍要与单独重算一致
hit = build(16, shift, shift, enable_prefix_caching=True, num_kv_blocks=32,
            draft_num_kv_blocks=16, num_speculative_tokens=2, max_num_seqs=1)
hit.add_request({"request_id": "A", "prompt_ids": shared_prompt, "max_new_tokens": 8})
for _ in range(3):
    hit.step()
hit.add_request({"request_id": "B", "prompt_ids": list(shared_prompt), "max_new_tokens": 8})
for _ in range(3):
    hit.step()
hit_seq = [s for s in hit.scheduler.running if s.request_id == "B"]
check("前缀命中：命中那条请求的两套 KV 与「对已提交前缀单独重算」一致",
      hit_seq and kv_diff(hit, hit_seq[0], "model", "kv_cache_pool",
                          hit_seq[0].cache.length) <= KV_ATOL
      and kv_diff(hit, hit_seq[0], "draft_model", "draft_kv_pool",
                  hit_seq[0].draft_cache.length) <= KV_ATOL,
      f"B: target={hit_seq[0].cache.length if hit_seq else None} "
      f"draft={hit_seq[0].draft_cache.length if hit_seq else None}")

# ------------------------------------------------ 9. 带惩罚项的 greedy：与普通路径逐 token 相同

# 惩罚项让**每一行**的历史都不同（真实生成 + 前 i 枚草稿），两个模型的提议与验证都必须
# 按这条规则喂历史，否则 argmax 会偏。这里用「整体 +1、但在位置 10 分叉」的目标脚本：
# 大部分草稿被接受、偶尔被拒，两条路径仍必须给出同一串 token。
target_diverge = lambda token, position: 3 if position == 10 else (token + 1) % 64
pen = dict(repetition_penalty=1.4, presence_penalty=0.4, frequency_penalty=0.3)
pen_spec = build(17, target_diverge, shift, num_speculative_tokens=2, num_kv_blocks=32)
pen_plain = build(17, target_diverge, shift, speculative_mode=None, num_kv_blocks=32,
                  with_draft=False)
pen_prompt = [5, 5, 1, 5, 5, 1, 5, 5]
spec_pen = steps(pen_spec, [dict({"request_id": "A", "prompt_ids": pen_prompt,
                                  "max_new_tokens": 10}, **pen)])
plain_pen = steps(pen_plain, [dict({"request_id": "A", "prompt_ids": pen_prompt,
                                    "max_new_tokens": 10}, **pen)])
check("带三种惩罚项的 greedy：draft_model 与普通路径**逐 token 相同**（逐行历史一致）",
      final_output(spec_pen, "A") == final_output(plain_pen, "A"),
      f"\n  投机 {final_output(spec_pen, 'A')}\n  普通 {final_output(plain_pen, 'A')}")
check("带惩罚项的 greedy：那条负载真的投了机（不是没草稿所以相等）",
      len(spec_rounds(spec_pen)) > 0
      and pen_spec.draft_proposer.num_proposed_tokens > 0,
      f"草稿 {pen_spec.draft_proposer.num_proposed_tokens} 枚")

# ------------------------------------------------ 10. CUDA FP32 / BF16 冒烟

if torch.cuda.is_available():
    for dtype in (torch.float32, torch.bfloat16):
        for mode in ("draft_model", "ngram", None):
            cuda = build(18, shift, shift, device="cuda", dtype=dtype,
                         num_speculative_tokens=2, num_kv_blocks=32, draft_num_kv_blocks=16,
                         max_num_seqs=2, speculative_mode=mode,
                         with_draft=(mode == "draft_model"))
            cuda_trace = steps(cuda, [
                {"request_id": "A", "prompt_ids": PROMPT, "max_new_tokens": 8},
                {"request_id": "B", "prompt_ids": [5, 6, 5, 6], "max_new_tokens": 8,
                 "temperature": 0.8, "top_k": 8, "seed": 3}])
            lengths = {rid: len(final_output(cuda_trace, rid)) for rid in ("A", "B")}
            pools_ok = all(u == 0 for u in cuda.kv_cache_pool.block_usage)
            draft_pool = getattr(cuda, "draft_kv_pool", None)
            if draft_pool is not None:
                pools_ok = pools_ok and all(u == 0 for u in draft_pool.block_usage)
            check(f"CUDA/{str(dtype).replace('torch.', '')}/{mode}：两条请求都跑完、两套池子归零",
                  lengths == {"A": 8, "B": 8} and pools_ok, str(lengths))
else:
    print("SKIP  CUDA 冒烟（本机没有 CUDA）")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
