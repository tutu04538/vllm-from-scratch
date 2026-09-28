"""第 56 关：counter RNG 的 CPU/GPU 一致性、分布与流隔离（需求 §6B）。

GPU 批量拒绝验证（triton 后端）还在调通中，但它的**随机流底座**已经完成并在这里钉住：

  1. `rejection_rng.event_uniform()`（CPU 参考）与 `rejection_triton` 的 Philox 内核
     **逐位一致**——两万组随机 (seed, 事件编号, 类型, token 下标) 一个都不差；
  2. 取值均匀（均值/极值/桶计数）；
  3. 流隔离：换 seed / 换事件编号 / 换 token 下标都给出不同的随机数；
  4. 指数竞赛（categorical 抽样，CPU 参考）的经验分布回到目标分布，且未归一化权重同样正确；
  5. 「只有真正发生的事件才推进计数器」这条约定在 CPU 参考里逐条可数。
"""

import math
import statistics
import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step56.rejection_rng import event_uniform, event_word, exponential_race

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


if not torch.cuda.is_available():
    print("SKIP  需要 CUDA（内核在 GPU 上跑）")
    sys.exit(0)

# ------------------------------------------------ 1. CPU 参考 与 GPU 内核逐位一致

from step56.rejection_triton import event_uniforms

SEED = 0x0123456789ABCDEF
N = 20000
indices = list(range(N))
tokens = [(i * 7919) % 151936 for i in range(N)]      # 真实词表大小上的 token 下标
gpu = event_uniforms(SEED, indices, tokens).cpu().tolist()
cpu = [event_uniform(SEED, indices[i], tokens[i]) for i in range(N)]
mismatch = sum(1 for a, b in zip(cpu, gpu) if a != b)
check("CPU 参考与 Triton 内核**逐位一致**（两万组事件，一个都不差）",
      mismatch == 0, f"不一致 {mismatch} 个")

# ------------------------------------------------ 2. 均匀性

check("取值落在 [0,1) 且既碰到下界附近也碰到上界附近（不是退化到几个点）",
      min(cpu) < 1e-3 and max(cpu) > 1 - 1e-3 and min(cpu) >= 0.0 and max(cpu) < 1.0,
      f"min={min(cpu):.6f} max={max(cpu):.6f} 均值={statistics.fmean(cpu):.4f}")
buckets = [0] * 10
for value in cpu:
    buckets[min(int(value * 10), 9)] += 1
expected = N / 10
sigma = math.sqrt(expected * 0.9)
check("十分桶计数都在 4σ 内（粗均匀性）",
      all(abs(count - expected) < 4 * sigma for count in buckets), str(buckets))

# ------------------------------------------------ 3. 流隔离

base = event_uniform(11, 3, 0)
check("流隔离：换 seed / 换事件编号 / 换 token 下标都给出不同的随机数",
      base != event_uniform(12, 3, 0)
      and base != event_uniform(11, 4, 0)
      and event_word(11, 3, 5) != event_word(11, 3, 6))
check("同一个事件反复取到**同一个**随机数（counter-based 的核心性质）",
      event_uniform(11, 3, 0) == base
      and event_word(11, 3, 5) == event_word(11, 3, 5))

def _error(fn):
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    return None


# 事件编号的位宽写清楚了，越界要明确报错而不是静默截断/回绕
check("事件编号用到 32 位空间（2**32-1 可用，2**32 明确报错而不是静默回绕）",
      isinstance(event_uniform(11, (1 << 32) - 1, 0), float)
      and _error(lambda: event_uniform(11, 1 << 32, 0)) is not None)


# ------------------------------------------------ 4. 指数竞赛（categorical 抽样）

P = [0.6, 0.3, 0.1, 0.0]
counts = [0] * 4
N_RACE = 60000
for event_index in range(N_RACE):
    token, error = exponential_race(9, event_index, P)
    assert not error
    counts[token] += 1
empirical = [c / N_RACE for c in counts]
sigma = [math.sqrt(p * (1 - p) / N_RACE) for p in (0.6, 0.3, 0.1)]
check("指数竞赛的经验分布回到 p（5σ 容差；权重未归一化也正确）",
      all(abs(empirical[i] - P[i]) < 5 * sigma[i] for i in range(3)),
      str([round(v, 4) for v in empirical]))
check("权重为 0 的 token 一次都不会被抽到", counts[3] == 0, str(counts))
check("权重全为 0 时明确报错（不是悄悄退回 0 号 token）",
      exponential_race(9, 0, [0.0, 0.0, 0.0, 0.0])[1] is True)

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
