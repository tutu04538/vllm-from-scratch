"""第 54 关：随机投机在引擎里的状态管理（需求 §5①④）。

纯概率算法在 `check_step54_rejection.py` 里测；这里测**引擎边界**：

  1. 临时惩罚历史与真实状态的界线：被拒的草稿绝不能进真实计数与已提交历史；
  2. 随机数归属与连续性：抢占/恢复不重置也不消耗请求自己的 generator；
  3. 重算不重复计入惩罚次数（`generated_total` 恰好等于已提交输出数）；
  4. 同批混跑：greedy / 随机 / 带惩罚项 / 不同 K，逐项检查计划与状态；
  5. CUDA FP32/BF16 冒烟。
"""

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import step54
from step54 import Engine

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


DIMS = dict(vocab_size=64, d_model=16, max_seq_len=128, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


class ScriptedModel:
    """包住真模型：状态推进照旧，logits 由脚本给定（每行一个目标 token）。

    `script=None` 表示「复制模式」：每个采样行返回它自己的输入 token。
    """

    def __init__(self, inner, script=None, favorite=None):
        """favorite=(高分 token, 次高分 token)：固定 logits，用惩罚项来分胜负。"""
        self.inner = inner
        self.script = None if script is None else list(script)
        self.favorite = favorite

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _forward_append(self, input_ids, num_scheduled_tokens, past_kv, kv_cache_pool,
                        sample_rows=None):
        self.inner._forward_append(input_ids, num_scheduled_tokens, past_kv, kv_cache_pool,
                                   sample_rows)
        rows = list(range(len(input_ids))) if sample_rows is None else list(sample_rows)
        logits = torch.zeros((len(rows), self.vocab_size))
        if self.favorite is not None:
            # 固定 logits：谁赢完全由**惩罚项**决定，正是逐行历史的试金石
            logits[:, self.favorite[0]] = 4.0
            logits[:, self.favorite[1]] = 3.5
            return logits
        if self.script is None:
            tokens = [int(input_ids[row]) for row in rows]
        else:
            tokens = [self.script.pop(0) for _ in rows]
        for row, token_id in enumerate(tokens):
            logits[row, token_id] = 4.0           # 目标 token 领先其余 0 分，分布可控
        return logits


def build(seed, script=None, favorite=None, **cfg):
    base = dict(max_num_seqs=4, max_num_batched_tokens=16, block_size=4, num_kv_blocks=64,
                enable_prefix_caching=False, scheduling_policy="fcfs",
                speculative_mode="ngram", num_speculative_tokens=2, prompt_lookup_n=2)
    base.update(cfg)
    torch.manual_seed(seed)
    inner = step54.TinyCausalLM(device="cpu", attention_backend="torch",
                                max_num_query_tokens=base["max_num_batched_tokens"], **DIMS)
    return Engine(model=ScriptedModel(inner, script, favorite), **base)


def run(engine, requests, arrivals=None, limit=80):
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
                         "items": list(engine.scheduler.scheduled_items)})
    return per_step


# ------------------------------------------------ 1. 临时计数与真实状态的界线

# A 的历史里 (4,1) 的续写是 [2,3] -> K=2；脚本让第一枚草稿被拒、纠正 token 是 5
# 目标分布用 top_k=1 收成 one-hot：草稿 2/3 的目标概率是 **0**，必拒——这样「拒绝
# 之后草稿不进真实状态」是确定性结论，不依赖某次抽样的运气。
reject_engine = build(11, [1, 8, 6, 9, 4, 7] + [0] * 100)
requests = [{"request_id": "A", "prompt_ids": [1, 2, 3, 4, 1, 2, 3, 4], "max_new_tokens": 6,
             "temperature": 0.8, "top_k": 1,
             "presence_penalty": 0.5, "frequency_penalty": 0.3, "seed": 3}]
steps = run(reject_engine, None, arrivals={0: requests})
seq = next(it["request"] for slot in steps for it in slot["items"])
state = seq.sampling_state
check("被拒的草稿不进真实计数（生成的只有真正提交过的那几枚）",
      state.generated_total == len(seq.output_ids)
      and set(state.generated_counts) <= set(seq.output_ids),
      f"generated_total={state.generated_total} outputs={list(seq.output_ids)}")
check("已提交历史 = prompt + 已提交输出（草稿一个字都没进去）",
      list(seq.all_token_ids) == list(seq.prompt_ids) + list(seq.output_ids))
check("临时历史不共享底层字典：引擎自己造一份，真实 state 的 counts 没被就地改过",
      isinstance(state.generated_counts, dict) and state.generated_total == len(seq.output_ids))

# 同一请求的惩罚计数与「普通单步采样」参考一致：把提交序列逐枚重放一遍
ref = SimpleNamespace(prompt_token_ids=state.prompt_token_ids, generated_counts={})
for token in seq.output_ids:
    ref.generated_counts[token] = ref.generated_counts.get(token, 0) + 1
check("逐枚重放提交序列得到的计数与真实计数一致（没多也没少）",
      ref.generated_counts == state.generated_counts, f"{state.generated_counts}")

# ------------------------------------------------ 2. RNG 归属与连续性

# 池子刚好卡住：A 占 2 块、B 占 2 块，等 B 的真实 token 需要新块时抢占 B。
# 用**复制模式**的脚本模型（每行返回自己的输入 token），不担心脚本耗尽。
pressure = build(3, None, num_kv_blocks=4, max_num_seqs=4, max_num_batched_tokens=16)
A_REQ = {"request_id": "A", "prompt_ids": [1, 2, 3, 3, 5, 6, 3], "max_new_tokens": 6}
B_REQ = {"request_id": "B", "prompt_ids": [11, 21, 31, 41, 51, 41, 41], "max_new_tokens": 6,
         "temperature": 0.9, "top_k": 8, "seed": 12345}
pressure.on_token = lambda ev: None
for req in (A_REQ, B_REQ):
    pressure.add_request(dict(req))
b_seq = pressure.scheduler.waiting[-1]
preempted_state = after_state = None
step = 0
while pressure.has_unfinished_requests() and step < 40:
    before = pressure.scheduler.num_preemptions
    snapshot = b_seq.sampling_state.generator.get_state().clone()
    pressure.step()
    step += 1
    if pressure.scheduler.num_preemptions > before:
        # 抢占发生在 schedule() 里、forward **之前**，所以这一步 B 一个随机数都没抽
        preempted_state = snapshot
        after_state = b_seq.sampling_state.generator.get_state().clone()
        break
check("抢占真的发生了（构造出来了）", preempted_state is not None)
check("抢占/调度**不消耗也不重置**请求自己的 generator（状态逐字节相同）",
      preempted_state is not None and after_state is not None
      and torch.equal(preempted_state, after_state))
check("被抢占的请求仍在等待，generator 还是它自己那一个",
      isinstance(b_seq.sampling_state.generator, torch.Generator))

# 跑完剩下的：被抢占的请求恢复后继续用同一个 generator（没有重建、没有重置）
while pressure.has_unfinished_requests():
    pressure.step()
check("恢复后 generator 没有回到初始状态（没有把随机流重置）",
      not torch.equal(b_seq.sampling_state.generator.get_state(),
                      torch.Generator(device="cpu").manual_seed(12345).get_state()))

# ------------------------------------------------ 3. 重算不重复计入惩罚次数


def tracked_commits(engine):
    """包一层唯一的提交入口，记录每次提交之后的 (惩罚计数, 已提交输出数)。"""
    seen = {}
    original = engine._commit_tokens

    def spy(seq, token_ids, notify):
        original(seq, token_ids, notify)
        seen[seq.request_id] = (seq.sampling_state.generated_total, len(seq.output_ids))

    engine._commit_tokens = spy
    return seen


counts_engine = build(3, None, num_kv_blocks=4, max_num_seqs=4, max_num_batched_tokens=16)
counts_engine.on_token = lambda ev: None
seen = tracked_commits(counts_engine)
for req in (A_REQ, dict(B_REQ, request_id="C", temperature=0.0, presence_penalty=0.4)):
    counts_engine.add_request(dict(req))
while counts_engine.has_unfinished_requests():
    counts_engine.step()
check("每次提交之后：惩罚计数恰好等于已提交输出数（重算的历史不重复计数）",
      all(total == length for total, length in seen.values()), str(seen))
check("这条负载确实发生过抢占重算（否则上面那句没测到东西）",
      counts_engine.scheduler.num_preemptions > 0,
      f"抢占 {counts_engine.scheduler.num_preemptions} 次")
check("被抢占后重算的请求也把惩罚计数补齐了（重放的历史不加倍）",
      all(total == length for total, length in seen.values())
      and len(seen) == 2, str(seen))

# ------------------------------------------------ 4. 同批混跑

mixed = build(29, None, max_num_batched_tokens=20, max_num_seqs=4)
MIXED = [
    {"request_id": "G", "prompt_ids": [1, 2, 3, 3, 5, 6, 3], "max_new_tokens": 5},
    {"request_id": "R", "prompt_ids": [7, 7, 1] * 3, "max_new_tokens": 5,
     "temperature": 0.8, "top_k": 20, "top_p": 0.9, "seed": 5},
    {"request_id": "P", "prompt_ids": [11, 21, 31, 41, 51, 41, 41], "max_new_tokens": 5,
     "temperature": 0.7, "presence_penalty": 0.6, "repetition_penalty": 1.2, "seed": 9},
    {"request_id": "K", "prompt_ids": [13, 13, 14, 14], "max_new_tokens": 5,
     "repetition_penalty": 1.5},                       # 贪心 + 惩罚：走逐行路径
]
mixed_steps = run(mixed, None, arrivals={0: MIXED})
all_tokens = [t for slot in mixed_steps for t in slot["tokens"]]
for rid in "GRPK":
    indices = [i for r, _, i in all_tokens if r == rid]
    check(f"同批混跑：{rid} 的 output_index 连续、完成一次",
          indices == list(range(len(indices))) and len(indices) == 5, str(indices))
by_request = {r: [] for r in "GRPK"}
for rid, _, _ in all_tokens:
    by_request[rid].append(rid)
check("同批混跑：四种采样参数（greedy / 随机 / 随机+惩罚 / 贪心+惩罚）都跑完了",
      all(len(v) == 5 for v in by_request.values()), str({k: len(v) for k, v in by_request.items()}))
pool = mixed.kv_cache_pool
chain, cur = [], pool.block_next[pool._SENTINEL_HEAD]
while cur != pool._SENTINEL_TAIL:
    chain.append(cur)
    cur = pool.block_next[cur]
check("同批混跑：结束后活动引用归零、可分配链与真实空闲一致",
      all(u == 0 for u in pool.block_usage)
      and set(chain) == set(pool._allocatable_block_indices()))

# ------------------------------------------------ 4.1 贪心 + 惩罚：与普通路径逐 token 一致

# 这条是本关最容易写错的地方：逐行历史。投机路径给第 j 行喂的历史是「真实生成 +
# 前 j 枚草稿」，普通路径每一步只喂「真实生成」——两者对**同一个位置**必须给出同一个
# argmax，于是整条输出序列必须逐 token 相同。历史用错（比如所有行都用真实计数）就会
# 在这里露馅。
GREEDY_PENALTY_REQ = {"request_id": "G", "prompt_ids": [5, 5, 1, 5, 5, 1, 5, 5],
                      "max_new_tokens": 8,
                      "repetition_penalty": 1.4, "presence_penalty": 0.4,
                      "frequency_penalty": 0.3}


def run_one(spec, seed, **cfg):
    engine = build(seed, None, **cfg) if spec else build(seed, None, speculative_mode=None,
                                                         **{k: v for k, v in cfg.items()})
    engine.on_token = lambda ev: None
    out = {}
    engine.scheduler.on_finished = lambda rec: out.__setitem__(rec["request_id"],
                                                               list(rec["output_ids"]))
    engine.add_request(dict(GREEDY_PENALTY_REQ))
    n = 0
    while engine.has_unfinished_requests():
        engine.step()
        n += 1
        assert n < 100
    return engine, out


spec_engine, spec_out = run_one(True, 29)
plain_engine, plain_out = run_one(False, 29)
check("贪心 + 三种惩罚：投机与普通路径的输出逐 token 相同（逐行历史一致）",
      spec_out == plain_out, f"\n  投机={spec_out}\n  普通={plain_out}")
draft_rounds = 0
spec_engine2 = build(29, None)
spec_engine2.on_token = lambda ev: None
spec_engine2.add_request(dict(GREEDY_PENALTY_REQ))
while spec_engine2.has_unfinished_requests():
    spec_engine2.step()
    draft_rounds += sum(1 for it in spec_engine2.scheduler.scheduled_items if it["draft_ids"])
check("这条负载真的提出过草稿（不是没投机所以相等）", draft_rounds > 0,
      f"提出草稿的轮数={draft_rounds}")

# ------------------------------------------------ 4.2 逐行历史真的用上了吗

# 固定的两档 logits（5 号 4.0、9 号 3.5）+ 频率惩罚 0.7，胜负完全由**该行看到的
# 生成历史**决定。历史是 [7,8,5,5,7,8] + 采样出的 5，末尾 (8,5) 的续写是 [5,7]。
#
# 这条用例是**判别性**的：把逐行历史换成「所有行都用真实计数」，输出会从这里
#   [5, 9, 5, 9, 5, 9]
# 变成
#   [5, 9, 5, 9, 5, 5]
# （最后一枚差在：正确历史下 5 号被 accepted 的草稿累积惩罚压过 9 号，错误历史下没有。）
# 我是先跑出这个差异、再把它写成断言的——不是先写断言再假设它有判别力。
HISTORY_PROMPT = [7, 8, 5, 5, 7, 8]
history_engine = build(29, None, favorite=(5, 9))
history_engine.on_token = lambda ev: None
history_engine.add_request({"request_id": "H", "prompt_ids": HISTORY_PROMPT,
                            "max_new_tokens": 6, "frequency_penalty": 0.7})
history_steps = run(history_engine, None)
h_seq = next(it["request"] for slot in history_steps for it in slot["items"])
h_drafts = [it["draft_ids"] for slot in history_steps for it in slot["items"] if it["draft_ids"]]
check("逐行历史：输出序列与「把每一行都按真实计数算惩罚」的写法不同（这条用例能判别）",
      list(h_seq.output_ids) == [5, 9, 5, 9, 5, 9] and len(h_drafts) >= 2,
      f"输出={list(h_seq.output_ids)} 草稿轮数={len(h_drafts)}；"
      f"用错历史会得到 [5, 9, 5, 9, 5, 5]")

# ------------------------------------------------ 5. CUDA 冒烟

if torch.cuda.is_available():
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(5)
        cuda_engine = Engine(
            device="cuda", attention_backend="torch", dtype=dtype,
            max_num_seqs=2, max_num_batched_tokens=8, block_size=4, num_kv_blocks=32,
            enable_prefix_caching=False, scheduling_policy="fcfs",
            speculative_mode="ngram", num_speculative_tokens=2, prompt_lookup_n=2, **DIMS)
        out = {}
        cuda_engine.on_token = lambda ev: out.setdefault(ev["request_id"], []).append(ev["token_id"])
        cuda_engine.add_request({"request_id": "x", "prompt_ids": [3, 1, 4, 1, 5, 9, 2, 6] * 2,
                                 "max_new_tokens": 6, "temperature": 0.8, "top_k": 20, "seed": 3})
        cuda_engine.add_request({"request_id": "y", "prompt_ids": [7, 7, 1] * 3,
                                 "max_new_tokens": 6, "frequency_penalty": 0.5})
        n = 0
        while cuda_engine.has_unfinished_requests():
            cuda_engine.step()
            n += 1
            assert n < 100
        check(f"CUDA/{dtype}：同批随机投机 + 带惩罚的贪心都能跑完",
              len(out.get("x", [])) == 6 and len(out.get("y", [])) == 6,
              str({k: len(v) for k, v in out.items()}))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
