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
     事件编号/token 下标各自隔离；未用到的位置不推进计数器。
  D. 定向观测：`verify_batch()` 内没有逐请求标量读取、没有 D2H；整批结果只回传一次。
"""

import math
import sys
import warnings
from types import SimpleNamespace

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step56 import TinyCausalLM
from step56 import rejection_triton as rt
from step56.rejection import BACKENDS, TRITON, BatchedRejectionSampler
from step56.rejection_rng import event_uniform, event_word, exponential_race
from step56.sample_runtime import SampleRuntime
from step56.sampling import (SamplingParams, SamplingState, TorchSampler)

if TRITON not in BACKENDS:
    print("FAIL  triton 后端没开（rejection.BACKENDS 里没有它）——这个脚本是它的调通工装")
    sys.exit(1)

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def probs(*values):
    return torch.tensor(values, dtype=torch.float32)


# ------------------------------------------------ CPU oracle（同一套算法）

def oracle_verify(draft_ids, row_probs, draft_probs, seed, counter, eos_ids, greedy=False):
    """GPU 路径的 CPU 参考：同样的规则、同样的 counter RNG、同样的指数竞赛。

    两处与内核**逐条对齐**的约定（都写在 rejection_triton.py 的 docstring 里）：

    - 贪心不消费随机数：`ratio` 落进 (0,1) 时报错（内核是错误码 3）。贪心的目标分布
      是 one-hot，`ratio` 只能取 `1/q[d] >= 1` 或 `0`，这个分支实际不可达——真落进去
      就是「目标分布不是 one-hot」这个前提被破坏了，两边都**报出来**而不是安静地
      当成必拒绝。torch 参考路径在那条分支上同样是直接抛异常。
    - 报错项返回**空结论**（committed=[]、kept=0），与 `materialize_results()` 一致。
    """
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
        elif greedy:
            # 与内核同一条规则：贪心落进 (0,1) = 前提被破坏，报错而不是安静必拒
            return dict(error="贪心落进随机分支", consumed=consumed, accepted=accepted,
                        kind=kind, committed=[], kept=0)
        else:
            uniform = event_uniform(seed, counter + consumed, 0)
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

def make_plan(seq, draft_ids, draft_probs):
    return {"request": seq, "input_ids": [], "num_scheduled_tokens": 1 + len(draft_ids),
            "draft_ids": list(draft_ids), "num_reserved_drafts": len(draft_ids),
            "draft_probs": list(draft_probs) if draft_probs else [],
            "sample_offset": 0, "num_sample_rows": len(draft_ids) + 1,
            "start_cache_length": 0, "can_sample": True}


class ScriptedSampler:
    """按**预设行**回答的打桩采样器：`_row_probs()` 会逐行问它要目标分布。

    为什么不直接传 logits 让真的 `TorchSampler` 算：那会再过一遍 softmax / 过滤 /
    惩罚，拿到的分布与用例里写的 p **不是同一个**（对着它比就只是比「两边都算了
    同一遍 softmax」）。这里把 p 原样钉住，对照的才是验证逻辑本身。
    `distribution()` 的真实实现由第五十二关起的常驻用例覆盖，这一关不必重复。
    """

    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = 0

    def distribution(self, row, params, state):
        value = self.rows[self.calls]
        self.calls += 1
        return value


def make_batch(cases, eos_ids=(3,), seed=11, counter=0):
    """把若干条 case 组成一个批：返回 `(backend, 已备好的批)`。

    D 段要在 `prepare_batch` 与 `verify_batch` **之间**做观测（数同步点），所以组批
    与验证分成两步。
    """
    sampler = ScriptedSampler([r for case in cases for r in case["row_probs"]])
    backend = BatchedRejectionSampler(TRITON, eos_ids, sampler, device="cuda")
    sampled = SamplingState(SamplingParams(vocab_size=8, temperature=0.8, seed=1),
                            [1, 2], torch.device("cuda"))
    plans, offset = [], 0
    for case in cases:
        seq = SimpleNamespace(sampling_params=case["params"], sampling_state=sampled,
                              rejection_seed=case.get("seed", seed),
                              rejection_rng_counter=case.get("counter", counter),
                              max_new_tokens=16, output_ids=[])
        plan = make_plan(seq, case["draft_ids"], case.get("draft_probs"))
        # 每个 item 在筛选后 logits 里的行区间（`prepare_batch()` 按它切片）
        plan["sample_offset"] = offset
        plan["num_sample_rows"] = len(case["row_probs"])
        offset += len(case["row_probs"])
        plans.append(plan)
    # logits 只用来定位行（`_row_probs()` 按 `sample_offset` 切它）；值由打桩采样器给，
    # 所以这里只要行数对得上就行
    logits = torch.zeros(offset, 8, device="cuda")
    return backend, backend.prepare_batch(logits, plans, greedy=None)


def gpu_verify(cases, eos_ids=(3,), seed=11, counter=0):
    """把若干条 case 组成一个批，跑 triton 后端，返回 CPU 侧结论列表。"""
    backend, batch = make_batch(cases, eos_ids, seed, counter)
    return backend.materialize_results(backend.verify_batch(batch))


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
    # 接受终止 token：不再消费、不抽 bonus（q[3] 必须为正——草稿是从 q 里抽出来的）
    dict(name="接受 EOS", params=RANDOM, draft_ids=[3, 1],
         row_probs=[probs(0, 0, 0, 1), probs(0.5, 0.5, 0, 0), probs(0.5, 0.5, 0, 0)],
         draft_probs=[probs(0.2, 0.2, 0.2, 0.4), probs(0.5, 0.5, 0, 0)]),
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
# 草稿 token 是 2，而它的提议分布给 token 2 的质量是 0 —— 「从 q 里抽不出一个
# q 质量为零的 token」，这是调用方的错，必须报错而不是除零、也不是当成必接受
bad = dict(CASES[2], name="非法 q[d]=0", draft_ids=[2],
           row_probs=[probs(0.3, 0.2, 0.5, 0), probs(0.2, 0.2, 0.6, 0)],
           draft_probs=[probs(0.1, 0.6, 0.0, 0.3)], counter=0)
results = gpu_verify([CASES[0], bad])
check("非法 q[d]=0：返回错误标志（不是异常、也不是悄悄接受）",
      results[0].error is None and results[1].error is not None,
      f"第一条 error={results[0].error}、第二条 error={results[1].error}")

# 贪心 + 非 one-hot 的目标分布：ratio 落进 (0,1)，这是「贪心不抽随机数」的前提被破坏。
# 用例是**故意违约**的（脚本采样器可以喂任意分布），目的是证明这道闸门真的会响：
# 报错误码 3，而不是安静地当成必拒绝（那样会给出一个看着合理、其实偏掉的分布）。
bad_greedy = dict(CASES[2], name="贪心落进随机分支", params=GREEDY, draft_ids=[1],
                  row_probs=[probs(0.5, 0.3, 0.2, 0), probs(0.4, 0.3, 0.3, 0)],
                  draft_probs=[probs(0.2, 0.6, 0.2, 0)], counter=0)
results = gpu_verify([bad_greedy])
got = results[0]
check("贪心落进随机分支：报错误码 3（不是安静地当成必拒绝）",
      got.error is not None and "贪心" in got.error and got.rng_consumed == 0,
      f"error={got.error}、rng_consumed={got.rng_consumed}")
expected = oracle_verify(bad_greedy["draft_ids"], bad_greedy["row_probs"],
                         bad_greedy["draft_probs"], 11, 0, EOS, greedy=True)
check("贪心落进随机分支：与 CPU oracle 一致（两边都报错、都不消费随机事件）",
      (got.error is None) == (expected["error"] is None) and got.rng_consumed == expected["consumed"],
      f"GPU error={got.error} / oracle error={expected['error']}")

# ragged 批：K 各不相同（2 / 0 / 2 / 1），混在一张批里逐项与 oracle 对齐。
# 这条最容易出错的地方就是「按项的行区间」——收尾行的下标、被拒草稿的位置都在
# 这个区间上算，一旦错位就会读到别人的行（甚至越界）。
RAGGED = [dict(CASES[0], name="ragged K=2"), dict(CASES[6], name="ragged K=0"),
          dict(CASES[4], name="ragged ngram K=2"), dict(CASES[2], name="ragged K=1")]
for case, counter in zip(RAGGED, (0, 0, 3, 7)):
    case["counter"] = counter
got_all = gpu_verify(RAGGED)
for index, case in enumerate(RAGGED):
    expected = oracle_verify(case["draft_ids"], case["row_probs"], case.get("draft_probs"),
                             case.get("seed", 11), case["counter"], EOS,
                             greedy=case["params"].is_greedy)
    got = got_all[index]
    check(f"ragged 批逐项与 oracle 一致：{case['name']}（K={len(case['draft_ids'])}）",
          (got.committed_ids == expected["committed"]
           and got.num_accepted == expected["accepted"]
           and got.kept_inputs == expected["kept"]
           and got.rng_consumed == expected["consumed"]),
          f"GPU {got.committed_ids}/{got.num_accepted}/{got.kept_inputs}/{got.rng_consumed}"
          f" vs oracle {expected['committed']}/{expected['accepted']}/{expected['kept']}"
          f"/{expected['consumed']}")

# 整批 K 全为 0：一张批里一个草稿位置都没有（输出列只剩哨兵 + 五个字段），
# 混批与空批两种退化形状都要能走通
all_zero = gpu_verify([dict(CASES[6], counter=i) for i in range(3)])
check("整批 K=0：三项都从各自唯一的目标行抽样（列布局要留住哨兵列）",
      all(all_zero[i].committed_ids == [oracle_verify(
          [], [probs(0.1, 0.6, 0.3, 0)], None, 11, i, EOS)["committed"][0]]
          for i in range(3)),
      str([r.committed_ids for r in all_zero]))

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

# 抽样内核（词表分块指数竞赛）与 CPU 参考**逐事件**对照：同一条权重行、同一个事件
# 编号，两边必须给出同一个 token。这是唯一能证明「归一化省掉也对」的地方——
# 权重故意不给成归一化的。
W_B = probs(0.6, 0.3, 0.1, 0)
EVENTS = list(range(512))
cpu_tokens = [exponential_race(7, event_index, list(W_B))[0] for event_index in EVENTS]
blind_backend, _ = make_batch([CASES[0]])
gpu_tokens, gpu_errs = blind_backend._sample(
    rt, torch.stack([W_B] * len(EVENTS)).to("cuda"),
    torch.full((len(EVENTS),), 7, dtype=torch.int64, device="cuda"),
    torch.zeros(len(EVENTS), dtype=torch.int64, device="cuda"),
    torch.tensor(EVENTS, dtype=torch.int64, device="cuda"))
check("抽样内核与 CPU 参考逐事件一致（512 个事件、未归一化权重）",
      gpu_tokens.cpu().tolist() == cpu_tokens and not gpu_errs.any().item(),
      f"前 8 个 GPU {gpu_tokens.cpu().tolist()[:8]} vs CPU {cpu_tokens[:8]}")

# 整行权重为零：报「无剩余质量」（错误码 2），不返回一个 token 0 冒充结论
zero_tokens, zero_errs = blind_backend._sample(
    rt, torch.zeros(1, 4, device="cuda"),
    torch.zeros(1, dtype=torch.int64, device="cuda"),
    torch.zeros(1, dtype=torch.int64, device="cuda"),
    torch.zeros(1, dtype=torch.int64, device="cuda"))
check("抽样内核：整行权重为零时报错（不是悄悄返回 token 0）",
      zero_errs.cpu().tolist() == [2], f"错误码 {zero_errs.cpu().tolist()}")

# 流隔离：换 seed / 换事件类型 / 换 token 下标都要给出不同的随机数
base = event_uniform(11, 3, 0)
check("随机流隔离：换 seed、换事件编号、换 token 下标都不同",
      base != event_uniform(12, 3, 0)
      and base != event_uniform(11, 4, 0)
      and event_word(11, 3, 5) != event_word(11, 3, 6))

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

# 观测手段一：PyTorch 的同步调试模式会给「任何触发设备同步的调用」发警告
# （`.item()` / `.cpu()` / `.tolist()` / `bool(张量)` / 数据相关的掩码索引 / H2D 拷贝
# 都算）。它自称 prototype、「不能覆盖所有同步操作」，所以这里同时保留 `.item()` 与
# `.cpu()/.tolist()` 两个**独立**计数器。
#
# 要分清两类同步，它们不是一个量级的东西：
#
#   * **每步 O(1) 的元数据上传**（把本轮的位置、种子、计数器搬到设备上）——
#     本关没有消除，见 docs/step56_gpu_rejection.md 的「剩余同步点」；
#   * **每请求一次的结果回传**——这是本关要消灭的（验收方在 189 里数过：
#     16 条请求 80 次 `aten::_local_scalar_dense`）。
#
# 判据因此是「同步点数与批大小无关」，而不是「同步次数为 0」。
torch.cuda.synchronize()
torch.cuda.set_sync_debug_mode("warn")

def count_sync(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, sum("synchronizing CUDA operation" in str(w.message) for w in caught)


def count_return_trips(fn):
    """数「回传」类的调用：`.item()` / `.cpu()` / `.tolist()`——三个都换成计数版。

    只数**设备张量**上的调用：`host.cpu().tolist()` 里那个 `.tolist()` 发生在已经
    回到主机的张量上，是纯主机操作，不算一次回传。
    """
    calls = {"item": 0, "cpu": 0, "tolist": 0}
    originals = (torch.Tensor.item, torch.Tensor.cpu, torch.Tensor.tolist)

    def wrap(name, original):
        def counted(self, *args, **kwargs):
            if self.is_cuda:
                calls[name] += 1
            return original(self, *args, **kwargs)
        return counted

    torch.Tensor.item = wrap("item", originals[0])
    torch.Tensor.cpu = wrap("cpu", originals[1])
    torch.Tensor.tolist = wrap("tolist", originals[2])
    try:
        result = fn()
    finally:
        torch.Tensor.item, torch.Tensor.cpu, torch.Tensor.tolist = originals
    return result, calls["item"] + calls["cpu"] + calls["tolist"]


many = [dict(CASES[2], counter=i) for i in range(16)]
batch_size = len(many)
backend, prepared = make_batch(many)
alone_backend, alone_batch = make_batch([CASES[2]])

_, syncs_many = count_sync(lambda: backend.verify_batch(prepared))
_, syncs_one = count_sync(lambda: alone_backend.verify_batch(alone_batch))
check("定向观测：verify_batch() 的同步点数与批大小无关（不是每请求一个）",
      syncs_many == syncs_one,
      f"{batch_size} 条请求 {syncs_many} 次 vs 1 条请求 {syncs_one} 次（都是每步 O(1) 的元数据上传）")

_, returns = count_return_trips(lambda: backend.verify_batch(prepared))
check("定向观测：verify_batch() 内**没有**任何回传（`.item()` / `.cpu()` / `.tolist()`）",
      returns == 0, f"{batch_size} 条请求触发了 {returns} 次回传")

_, commit_returns = count_return_trips(lambda: backend.materialize_results(
    backend.verify_batch(prepared)))
check("定向观测：整批结论只回传一次（`materialize_results` 里那一次 `.cpu()`）",
      commit_returns == 1, f"materialize 的回传次数 {commit_returns}")

# 统计口径：消费的随机事件数在**回传之后**从主机侧那张表里累加（不额外同步）
check("定向观测：消费的随机事件数从回传结果里统计（16 条 × 各 1 个接受事件 + 1 次抽样）",
      backend.num_rng_events >= 2 * batch_size,
      f"num_rng_events={backend.num_rng_events}")

torch.cuda.set_sync_debug_mode("default")

# ------------------------------------------------ E. 后端组合与 K=0 回退项

DIMS = dict(vocab_size=64, d_model=16, max_seq_len=96, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


def rejected(**cfg):
    """构造引擎，返回错误信息（没报错就返回 None）。"""
    from step56 import Engine
    try:
        Engine(attention_backend="torch", **DIMS, **cfg)
        return None
    except ValueError as error:
        return str(error)


check("组合：triton 后端 + CPU 设备在构造时明确拒绝",
      (rejected(device="cpu", speculative_mode="ngram", rejection_backend="triton") or "")
      .startswith("rejection_backend='triton' 需要 CUDA 设备"),
      str(rejected(device="cpu", speculative_mode="ngram", rejection_backend="triton")))
check("组合：triton 后端 + 不开投机在构造时明确拒绝（它会空转，不能悄悄成功）",
      "只在投机验证里有意义"
      in (rejected(device="cuda", rejection_backend="triton") or ""),
      str(rejected(device="cuda", rejection_backend="triton")))
check("组合：未知后端名在构造时明确拒绝",
      "未知的 rejection_backend" in (rejected(device="cuda", rejection_backend="cuda") or ""))

# K=0 回退项：计划要投机、实际一枚草稿都没提出来（补算吃光预算 / draft 池不够 /
# 草稿就是空的）。triton 后端必须把它也纳入验证批——它的采样要用**同一套 counter
# RNG**，中途切回 target 的 torch generator 会让这条请求的随机流换一条；
# torch 后端保持第五十五关的行为（没有草稿就不走验证）。
triton_runtime = SampleRuntime(TorchSampler(), None, {3}, None, "triton")
torch_runtime = SampleRuntime(TorchSampler(), None, {3}, None, "torch")
check("K=0 回退项：triton 后端把它纳入验证批（用同一套 counter RNG）",
      triton_runtime._needs_verification({"draft_ids": [], "num_reserved_drafts": 2})
      and not torch_runtime._needs_verification({"draft_ids": [], "num_reserved_drafts": 2})
      and not triton_runtime._needs_verification({"draft_ids": [],
                                                 "num_reserved_drafts": 0}),
      f"triton 有草稿 {triton_runtime._needs_verification({'draft_ids': [1], 'num_reserved_drafts': 2})}、"
      f"triton 无草稿有预留 "
      f"{triton_runtime._needs_verification({'draft_ids': [], 'num_reserved_drafts': 2})}、"
      f"torch 无草稿有预留 "
      f"{torch_runtime._needs_verification({'draft_ids': [], 'num_reserved_drafts': 2})}")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
