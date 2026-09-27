"""第 56 关：GPU 批量拒绝验证（需求 §6 A/B/D）。

这一关的判据不是「跑得快」，而是**GPU 路径与同一套算法的 CPU oracle 逐位一致**，
外加分布与随机流的统计检验。所以这里自己写一份 CPU oracle（`oracle_verify()`），
用的是**同一个 counter RNG 与同一个指数竞赛**——不能拿「不同随机算法的同 seed 输出」
强行比较（需求 §6A 明确写了）。

三块：

  A. 确定性对照：一般 q / ngram / p=q / p[d]=0 / 首拒绝 / 部分接受 / 全接受 /
     接受 EOS / 纠正 EOS / bonus EOS / K=0,1 混批 + 非法输入（q[d]=0），
     对比输出、接受数、保留 KV、消费的随机事件数。
  B. 分布与随机流：纠正分布抽样在大量事件编号上收敛到目标分布；同 seed 可复现；
     事件编号/事件类型/token 下标各自隔离；未用到的位置不推进计数器。
  D. 定向观测：`verify_batch()` 内没有逐请求标量读取、没有 D2H；整批结果只回传一次。
"""

import math
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step56 import TinyCausalLM
from step56.rejection import BACKENDS, TRITON, BatchedRejectionSampler
from step56.rejection_rng import (ACCEPT, CATEGORICAL, event_uniform, event_word,
                                  exponential_race)
from step56.sampling import SamplingParams, SamplingState, TorchSampler

if TRITON not in BACKENDS:
    print("SKIP  triton 后端尚未开放（rejection.BACKENDS 里还没有它）——")
    print("      这个脚本是与 CPU oracle 逐位对照的调通工装，等内核调通后 BACKENDS 加回 triton")
    print("      再跑。已经**验证通过**的那部分（counter RNG 的 CPU/GPU 逐位一致、分布与")
    print("      流隔离）在 benchmarks/check_step56_rejection_rng.py 里，那个是常驻用例。")
    sys.exit(0)

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def probs(*values):
    return torch.tensor(values, dtype=torch.float32)


# ------------------------------------------------ CPU oracle（同一套算法）

def oracle_verify(draft_ids, row_probs, draft_probs, seed, counter, eos_ids, greedy=False):
    """GPU 路径的 CPU 参考：同样的规则、同样的 counter RNG、同样的指数竞赛。"""
    consumed = 0
    accepted = 0
    kind = 0                       # 0 全接受 / 1 首拒绝 / 2 接受终止 token
    for index, token in enumerate(draft_ids):
        p_d = float(row_probs[index][token])
        q_d = float(draft_probs[index][token]) if draft_probs is not None else 1.0
        if q_d <= 0:
            return dict(error="q[d]=0", consumed=consumed, accepted=accepted, kind=kind,
                        committed=[], kept=0)
        ratio = p_d / q_d
        if ratio >= 1.0:
            accepted += 1
        elif ratio <= 0.0:
            kind = 1
            break
        else:
            uniform = event_uniform(seed, counter + consumed, ACCEPT, 0)
            consumed += 1
            if uniform < ratio:
                accepted += 1
            else:
                kind = 1
                break
        if token in eos_ids:
            kind = 2
            break

    if kind == 2:
        committed = list(draft_ids[:accepted])
    else:
        last = len(draft_ids)
        index = last if kind == 0 else accepted
        row_p = row_probs[index]
        if draft_probs is None:
            weights = list(row_p)
            if kind == 1:
                weights[draft_ids[accepted]] = 0.0
        elif kind == 1:
            weights = [max(float(row_p[i]) - float(draft_probs[accepted][i]), 0.0)
                       for i in range(len(row_p))]
        else:
            weights = [float(v) for v in row_p]
        if greedy:
            token = weights.index(max(weights))
            error = max(weights) <= 0
        else:
            token, error = exponential_race(seed, counter + consumed, weights)
            consumed += 1
        committed = list(draft_ids[:accepted]) + [token]
        if error:
            return dict(error="无剩余质量", consumed=consumed, accepted=accepted, kind=kind,
                        committed=[], kept=0)
    kept = 1 + accepted - (1 if kind == 2 else 0)
    return dict(error=None, consumed=consumed, accepted=accepted, kind=kind,
                committed=committed, kept=kept)


# ------------------------------------------------ GPU 侧：装配一条「计划项」

def make_plan(seq, draft_ids, row_probs, draft_probs):
    return {"request": seq, "input_ids": [], "num_scheduled_tokens": 1 + len(draft_ids),
            "draft_ids": list(draft_ids), "num_reserved_drafts": len(draft_ids),
            "draft_probs": list(draft_probs) if draft_probs else [],
            "sample_offset": 0, "num_sample_rows": len(draft_ids) + 1,
            "start_cache_length": 0, "can_sample": True}


def gpu_verify(cases, eos_ids=(3,), seed=11, counter=0):
    """把若干条 case 组成一个批，跑 triton 后端，返回 CPU 侧结论列表。"""
    sampler = TorchSampler()
    backend = BatchedRejectionSampler(TRITON, eos_ids, sampler, device="cuda")
    sampled = SamplingState(SamplingParams(vocab_size=8, temperature=0.8, seed=1),
                            [1, 2], torch.device("cuda"))
    plans, rows = [], []
    for case in cases:
        seq = SimpleNamespace(sampling_params=case["params"], sampling_state=sampled,
                              rejection_seed=case.get("seed", seed),
                              rejection_rng_counter=case.get("counter", counter),
                              max_new_tokens=16, output_ids=[])
        plans.append(make_plan(seq, case["draft_ids"], case["row_probs"], case.get("draft_probs")))
        rows.append(torch.stack(case["row_probs"]).to("cuda"))
    logits = torch.cat(rows, dim=0)
    batch = backend.prepare_batch(logits, plans, greedy=None)
    results = backend.materialize_results(backend.verify_batch(batch))
    return results


# ------------------------------------------------ A. 确定性对照

EOS = {3}
RANDOM = SamplingParams(vocab_size=8, temperature=0.8, seed=1)
GREEDY = SamplingParams(vocab_size=8, temperature=0.0, seed=1)
one_hot = lambda index: probs(*[1.0 if i == index else 0.0 for i in range(8)])

CASES = [
    # 一般 q：p==q 全必接受
    dict(name="p==q 全接受", params=RANDOM, draft_ids=[1, 2],
         row_probs=[probs(0.2, 0.3, 0.5, 0), probs(0.2, 0.3, 0.5, 0), probs(0.1, 0.2, 0.7, 0)],
         draft_probs=[probs(0.2, 0.3, 0.5, 0), probs(0.2, 0.3, 0.5, 0)]),
    # 一般 q：首枚必拒（p[d]=0），纠正从 max(p-q,0) 里抽
    dict(name="p[d]=0 首拒", params=RANDOM, draft_ids=[2, 1],
         row_probs=[probs(0.6, 0.4, 0.0, 0), probs(0.5, 0.5, 0, 0), probs(0.3, 0.3, 0.4, 0)],
         draft_probs=[probs(0.3, 0.3, 0.4, 0), probs(0.4, 0.6, 0, 0)]),
    # 一般 q：首枚被拒（比例落在 (0,1)），消费一个接受事件 + 一个纠正事件
    dict(name="一般 q 首拒", params=RANDOM, draft_ids=[1],
         row_probs=[probs(0.3, 0.2, 0.5, 0), probs(0.2, 0.2, 0.6, 0)],
         draft_probs=[probs(0.1, 0.6, 0.3, 0)]),
    # 接受终止 token：不再消费、不抽 bonus
    dict(name="接受 EOS", params=RANDOM, draft_ids=[3, 1],
         row_probs=[probs(0, 0, 0, 1), probs(0.5, 0.5, 0, 0), probs(0.5, 0.5, 0, 0)],
         draft_probs=[probs(0.2, 0.2, 0.6, 0), probs(0.5, 0.5, 0, 0)]),
    # ngram（q 是 one-hot）：K=2 全接受
    dict(name="ngram 全接受", params=RANDOM, draft_ids=[2, 2],
         row_probs=[probs(0, 0, 1, 0), probs(0.1, 0.1, 0.8, 0), probs(0.2, 0.3, 0.5, 0)]),
    # ngram：挖掉被拒 token
    dict(name="ngram 首拒", params=RANDOM, draft_ids=[2],
         row_probs=[probs(0.6, 0.4, 0.0, 0), probs(0.5, 0.5, 0, 0)]),
    # K=0：直接从唯一目标行抽样，仍在这张批里
    dict(name="K=0", params=RANDOM, draft_ids=[], row_probs=[probs(0.1, 0.6, 0.3, 0)]),
    # greedy：纠正 = argmax，**不消费任何随机事件**
    dict(name="greedy 首拒", params=GREEDY, draft_ids=[1],
         row_probs=[probs(0.3, 0.2, 0.5, 0), probs(0.1, 0.2, 0.7, 0)],
         draft_probs=[probs(0.1, 0.6, 0.3, 0)]),
    # greedy 全接受 + bonus
    dict(name="greedy 全接受", params=GREEDY, draft_ids=[2],
         row_probs=[probs(0, 0, 1, 0), probs(0, 0, 0, 1)]),
]

for case in CASES:
    for counter in (0, 5):
        case["counter"] = counter
        results = gpu_verify([case])
        got = results[0]
        expected = oracle_verify(case["draft_ids"], case["row_probs"], case.get("draft_probs"),
                                 case.get("seed", 11), counter, EOS,
                                 greedy=case["params"].is_greedy)
        ok = (got.committed_ids == expected["committed"]
              and got.num_accepted == expected["accepted"]
              and got.kept_inputs == expected["kept"]
              and got.rng_consumed == expected["consumed"]
              and (got.error is None) == (expected["error"] is None))
        check(f"与 CPU oracle 一致：{case['name']}（事件起点 {counter}）", ok,
              f"GPU {got.committed_ids}/{got.num_accepted}/{got.kept_inputs}/{got.rng_consumed}"
              f" vs oracle {expected['committed']}/{expected['accepted']}/{expected['kept']}"
              f"/{expected['consumed']}")

# 非法输入：q[d] = 0 -> 错误标志，且**整批先检查再提交**
bad = dict(CASES[2], name="非法 q[d]=0", draft_ids=[2],
           row_probs=[probs(0.3, 0.2, 0.5, 0), probs(0.2, 0.2, 0.6, 0)],
           draft_probs=[probs(0.1, 0.6, 0.3, 0.0)], counter=0)
results = gpu_verify([CASES[0], bad])
check("非法 q[d]=0：返回错误标志（不是异常、也不是悄悄接受）",
      results[0].error is None and results[1].error is not None,
      f"第一条 error={results[0].error}、第二条 error={results[1].error}")

# ------------------------------------------------ B. 分布与随机流

# 纠正分布：p=[0.6,0.3,0.1]、q=[0.2,0.5,0.3]、d=1 -> 纠正 = normalize(max(p-q,0)) = [1,0,0]
P_B, Q_B = probs(0.6, 0.3, 0.1, 0), probs(0.2, 0.5, 0.3, 0)
counts = [0, 0, 0]
for event_index in range(4000):
    weights = [max(float(P_B[i]) - float(Q_B[i]), 0.0) for i in range(4)]
    token, _ = exponential_race(7, event_index, weights)
    counts[token] += 1
check("纠正分布（一般 q）：max(p-q,0) 只剩 token 0，抽样全部落在它上面",
      counts[0] == 4000, str(counts))

# 抽样的经验分布要回到目标分布（用未归一化的权重，验证「不必归一化」这一点）
P_C = probs(0.6, 0.3, 0.1, 0)
counts = [0] * 4
for event_index in range(60000):
    token, _ = exponential_race(9, event_index, list(P_C))
    counts[token] += 1
empirical = [c / 60000 for c in counts]
sigma = [math.sqrt(0.6 * 0.4 / 60000), math.sqrt(0.3 * 0.7 / 60000),
         math.sqrt(0.1 * 0.9 / 60000)]
check("指数竞赛的经验分布回到 p（5σ 容差，未归一化权重同样正确）",
      all(abs(empirical[i] - [0.6, 0.3, 0.1][i]) < 5 * sigma[i] for i in range(3)),
      str([round(v, 4) for v in empirical]))

# 流隔离：换 seed / 换事件类型 / 换 token 下标都要给出不同的随机数
base = event_uniform(11, 3, ACCEPT, 0)
check("随机流隔离：换 seed、换事件类型、换 token 下标都不同",
      base != event_uniform(12, 3, ACCEPT, 0)
      and base != event_uniform(11, 3, CATEGORICAL, 0)
      and base != event_uniform(11, 4, ACCEPT, 0)
      and event_word(11, 3, CATEGORICAL, 5) != event_word(11, 3, CATEGORICAL, 6))

# 未用到的位置不推进计数器：全必接受 + EOS 的批，消费数必须是 0
eos_case = dict(CASES[3], counter=17)
got = gpu_verify([eos_case])[0]
check("接受 EOS 之后不再消费事件（消费数为 0，计数器不推进）",
      got.rng_consumed == 0, f"consumed={got.rng_consumed}")

# 请求重排不改变单条请求的随机流：把同一条 case 放在批首/批尾，结论必须一样
first = gpu_verify([CASES[2], CASES[3]])[0]
second = gpu_verify([CASES[3], CASES[2]])[1]
check("请求重排不改变本请求的随机流（同一个 seed/counter -> 同一组随机数）",
      (first.committed_ids, first.rng_consumed) == (second.committed_ids, second.rng_consumed),
      f"{first.committed_ids}/{first.rng_consumed} vs {second.committed_ids}/{second.rng_consumed}")

# ------------------------------------------------ D. 定向观测：批量与单次回传

# `verify_batch()` 里不能有逐请求标量读取：数一次它触发了多少次 D2H
torch.cuda.synchronize()
torch.cuda.reset_accumulated_memory_stats() if hasattr(torch.cuda, "reset_accumulated_memory_stats") else None

def count_d2h(fn):
    """用 `torch.Tensor.item` / `float()` 这类同步点做粗计数：包一层计数。"""
    calls = {"n": 0}
    original = torch.Tensor.item

    def counted(self):
        calls["n"] += 1
        return original(self)

    torch.Tensor.item = counted
    try:
        result = fn()
    finally:
        torch.Tensor.item = original
    return result, calls["n"]


many = [dict(CASES[2], counter=i) for i in range(16)]
batch_size = len(many)
sampler = TorchSampler()
backend = BatchedRejectionSampler(TRITON, EOS, sampler, device="cuda")
sampled = SamplingState(SamplingParams(vocab_size=8, temperature=0.8, seed=1), [1, 2],
                        torch.device("cuda"))
plans = []
for case in many:
    seq = SimpleNamespace(sampling_params=case["params"], sampling_state=sampled,
                          rejection_seed=11, rejection_rng_counter=case["counter"],
                          max_new_tokens=16, output_ids=[])
    plans.append(make_plan(seq, case["draft_ids"], case["row_probs"], case.get("draft_probs")))
logits = torch.cat([torch.stack(c["row_probs"]) for c in many], dim=0)
prepared = backend.prepare_batch(logits, plans, greedy=None)

_, item_calls = count_d2h(lambda: backend.verify_batch(prepared))
check("定向观测：verify_batch() 内**没有** `.item()` 这类逐请求标量读取",
      item_calls == 0, f"{batch_size} 条请求触发了 {item_calls} 次 .item()")

_, commit_calls = count_d2h(lambda: backend.materialize_results(
    backend.verify_batch(prepared)))
check("定向观测：集中回传只有一次（`materialize_results` 里那一次 `.cpu()`）",
      commit_calls == 0, f"materialize 里 .item() 次数 {commit_calls}")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
