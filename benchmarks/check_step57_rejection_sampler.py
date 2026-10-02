"""57E 验收（对应需求里的 `test_rejection_sampler.py`）：验证算法本身。

按 199 §7/§8 分三段：

  1. **两条路径的形状**：greedy 逐位置比对 argmax；random 用 `min(1, p/q)` 接受、
     拒绝时用 `max(p - q, 0)` 恢复。固定 p/q + **注入**均匀随机数与 recovered 值，
     这样"算法错"和"随机流不同"不会混在一起（199 §8 允许注入）。
  2. **分布正确性**：接受概率与 recovered 分布对着**独立 CPU 公式**逐值比，再用大量抽样
     做统计检查（不要求与旧代码同 seed 同 token）。
  3. **形状与边界**：K=0、ragged、q[d]=0（防御性拒绝）、点质量提议（`draft_probs=None`）、
     bonus 只在全接受时追加、无效位置填 -1。
"""

import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch

from step57.sample import Sampler, SamplingMetadata
from step57.spec_decode.metadata import SpecDecodeMetadata
from step57.spec_decode.rejection_sampler import RejectionSampler

FAIL = []
V = 5


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def metadata_for(drafts: list[list[int]]) -> SpecDecodeMetadata:
    req_ids = [f"r{index}" for index in range(len(drafts))]
    scheduled = {req_id: (len(draft) + 1 if draft else 1)
                 for req_id, draft in zip(req_ids, drafts)}
    return SpecDecodeMetadata.from_scheduled(
        {req_id: draft for req_id, draft in zip(req_ids, drafts) if draft},
        scheduled, req_ids)


def sampling_metadata(temperatures, drafts, output_token_ids=None, **kwargs):
    count = len(temperatures)
    temperature = torch.tensor(temperatures, dtype=torch.float32)
    return SamplingMetadata(
        temperature=None if all(value < 1e-5 for value in temperatures) else temperature,
        all_greedy=all(value < 1e-5 for value in temperatures),
        all_random=all(value >= 1e-5 for value in temperatures),
        top_k=None, top_p=None, generators={}, no_penalties=True,
        prompt_token_ids=[[] for _ in range(count)],
        output_token_ids=output_token_ids or [[] for _ in range(count)],
        min_tokens=[0] * count, stop_token_ids=[[] for _ in range(count)],
        spec_token_ids=[list(draft) for draft in drafts],
        **kwargs)


sampler = RejectionSampler(Sampler())


def run(drafts, logits, temperature=0.0, *, uniforms=None, recoveries=None,
        draft_probs=None, history=None):
    return sampler.forward(metadata_for(drafts), torch.tensor(logits, dtype=torch.float32),
                           draft_probs, sampling_metadata([temperature] * len(drafts), drafts,
                                                          history),
                           uniforms=uniforms, recoveries=recoveries).sampled_token_ids


# ------------------------------------------------ 1. greedy：逐位置比对 argmax

# target argmax = [0, 1, 2]（行 0/1 是验证行、行 2 是 bonus 行）
logits = [[9.0, 0, 0, 0, 0], [0, 9.0, 0, 0, 0], [0, 0, 9.0, 0, 0]]
out = run([[0, 1]], logits)
check("1. greedy 全接受：两枚草稿 + bonus，长度 K+1",
      out.tolist() == [[0, 1, 2]], str(out.tolist()))

out = run([[0, 1]], logits)[:, :2]
out_first_reject = run([[1, 1]], logits)
check("1. greedy 首枚就不同：用 target 的 argmax 顶替，**后面全部丢掉**（不追加 bonus）",
      out_first_reject.tolist() == [[0, -1, -1]], str(out_first_reject.tolist()))

out_mid = run([[0, 0]], logits)
check("1. greedy 中间被拒（第 2 枚不同）：接受 1 枚 + 顶替 1 枚，长度 2",
      out_mid.tolist() == [[0, 1, -1]], str(out_mid.tolist()))

# ------------------------------------------------ 2. random：接受判定与恢复

# 构造 p：row0 上草稿 token 0 的 p=0.9（one-hot 附近），q = 点质量（1.0）
logits_random = [[9.0, 0, 0, 0, 0], [0, 0, 9.0, 0, 0]]
u_accept = torch.tensor([0.5], dtype=torch.float64)
u_reject = torch.tensor([0.9999], dtype=torch.float64)
out_accept = run([[0]], logits_random, temperature=1.0, uniforms=u_accept)
out_reject = run([[0]], logits_random, temperature=1.0, uniforms=u_reject,
                 recoveries=torch.tensor([3]))
check("2. random 接受（u < p/q）：提交草稿 + bonus",
      out_accept.tolist() == [[0, 2]], str(out_accept.tolist()))
check("2. random 拒绝（u > p/q）：提交 recovered token，**不追加 bonus**",
      out_reject.tolist() == [[3, -1]], str(out_reject.tolist()))

# 边界：u 恰好等于 p/q → 接受（判定是 >=）
p_draft = float(torch.softmax(torch.tensor(logits_random[0]), dim=-1)[0])
out_edge = run([[0]], logits_random, temperature=1.0,
               uniforms=torch.tensor([p_draft], dtype=torch.float64))
check("2. 边界：u == p/q 时接受（判定用 `>=`，与 vLLM 内核一致）",
      out_edge.tolist() == [[0, 2]], f"p/q={p_draft:.6f}、结果={out_edge.tolist()}")

# 带 q 的接受概率：q 是**实际提议**的分布，不是重新算的 softmax。
# 用均匀的 target（p=0.2 每项）与更"自信"的 q（0.6/0.4），把 p/q 压到 1 以下，
# 这样 u 落在 [0,1) 里就能同时造出"刚好接受"和"刚好拒绝"两个边界
flat_logits = [[0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 9.0]]
draft_probs = torch.tensor([[0.6, 0.4, 0, 0, 0], [0, 0, 1.0, 0, 0]])
ratio = 0.2 / 0.6
out_q = run([[0]], flat_logits, temperature=1.0, draft_probs=draft_probs,
            uniforms=torch.tensor([ratio - 1e-6], dtype=torch.float64))
out_q_reject = run([[0]], flat_logits, temperature=1.0, draft_probs=draft_probs,
                   uniforms=torch.tensor([ratio + 1e-6], dtype=torch.float64),
                   recoveries=torch.tensor([4]))
check("2. 有 q 时按 p/q 判定：刚好低于阈值接受、刚好高于拒绝",
      out_q.tolist() == [[0, 4]] and out_q_reject.tolist() == [[4, -1]],
      f"p/q={ratio:.4f} → {out_q.tolist()} / {out_q_reject.tolist()}")

# 不做接受判定时（u 极小）也要按 q 归一：这里顺便确认 bonus 行取的是最后一行的 logits
out_q_bonus = run([[0]], flat_logits, temperature=1.0, draft_probs=draft_probs,
                  uniforms=torch.tensor([0.0], dtype=torch.float64))
check("2. 接受到底时 bonus 取的是 bonus 行（不是验证行）",
      out_q_bonus.tolist() == [[0, 4]], str(out_q_bonus.tolist()))

# q[d] == 0：防御性拒绝（不能算出 NaN/Inf）
zero_q = torch.tensor([[0.0, 1.0, 0, 0, 0], [0, 0, 1.0, 0, 0]])
out_zero_q = run([[0]], logits_random, temperature=1.0, draft_probs=zero_q,
                 uniforms=torch.tensor([1e-9], dtype=torch.float64),
                 recoveries=torch.tensor([2]))
check("2. q[d] == 0：即使 u 极小也**拒绝**（vLLM 内核同样是防御性拒绝）",
      out_zero_q.tolist() == [[2, -1]], str(out_zero_q.tolist()))

# ------------------------------------------------ 3. 独立 CPU 公式对照

def cpu_reference(draft_token, p_row, q_row, uniform, recovered):
    """按定义写一遍：接受概率 min(1, p[d]/q[d])，否则用 max(p-q, 0) 采样。"""
    draft_prob = 1.0 if q_row is None else q_row[draft_token]
    accept_probability = min(1.0, p_row[draft_token] / draft_prob) if draft_prob > 0 else 0.0
    if uniform < accept_probability:
        return draft_token, True
    return recovered, False


p_row = torch.softmax(torch.tensor(logits_random[0]), dim=-1)
q_row = torch.tensor([0.4, 0.6, 0, 0, 0])
for uniform in (0.01, 0.5, 0.99):
    expected_token, expected_accept = cpu_reference(0, p_row.tolist(), q_row.tolist(), uniform, 4)
    got = run([[0]], logits_random, temperature=1.0, draft_probs=q_row.unsqueeze(0),
              uniforms=torch.tensor([uniform], dtype=torch.float64),
              recoveries=torch.tensor([4]))
    check(f"3. 与 CPU 公式逐值一致（u={uniform}）",
          (got[0, 0].item() == expected_token) and (got[0, 1].item() == 2) == expected_accept,
          f"CPU={expected_token}/接受 {expected_accept}，实现={got.tolist()}")

# recovered 的分布 ∝ max(p - q, 0)：统计检查
draft_probs_big = torch.zeros(1, V)
draft_probs_big[0, 0] = 0.5
draft_probs_big[0, 1] = 0.5
target_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])       # p 均匀 = 0.2
counts = torch.zeros(V)
draws = 4000
two_rows = torch.cat([target_logits, target_logits], dim=0)      # 验证行 + bonus 行
for _ in range(draws):
    got = sampler.forward(metadata_for([[0]]), two_rows, draft_probs_big,
                          sampling_metadata([1.0], [[0]]),
                          uniforms=torch.tensor([0.9], dtype=torch.float64)
                          ).sampled_token_ids
    counts[got[0, 0].item()] += 1
expected = torch.tensor([0.0, 0.0, 1 / 3, 1 / 3, 1 / 3])
check("3. recovered 分布 ∝ max(p - q, 0)：大量抽样后频率收敛（草稿位置的 p-q 为 0，永不出现）",
      (counts / draws - expected).abs().max().item() < 0.03,
      f"实测 {[round(x, 3) for x in (counts / draws).tolist()]} vs 期望 {[round(x, 3) for x in expected.tolist()]}")

# ------------------------------------------------ 4. 形状、ragged 与点质量

ragged_logits = [[9.0, 0, 0, 0, 0], [0, 9.0, 0, 0, 0], [0, 0, 9.0, 0, 0], [0, 0, 0, 9.0, 0]]
out_ragged = run([[0, 1], []], ragged_logits)
check("4. ragged：一条 K=2、一条 K=0；padded 到 max_spec_len+1，无效位填 -1",
      out_ragged.tolist() == [[0, 1, 2], [3, -1, -1]], str(out_ragged.tolist()))

out_k0 = run([[]], [[0, 0, 9.0, 0, 0]])
check("4. 全批 K=0：等价于普通解码（就是 bonus 那一行）",
      out_k0.tolist() == [[2]], str(out_k0.tolist()))

# 点质量提议（ngram）：draft_probs=None → q[d] = 1
out_point = run([[0]], logits_random, temperature=1.0, draft_probs=None,
                uniforms=torch.tensor([0.5], dtype=torch.float64))
check("4. `draft_probs=None` 表示点质量提议（q[d]=1）：接受判定退化成 p[d] >= u",
      out_point.tolist() == [[0, 2]], str(out_point.tolist()))

# 混合批：一条贪心、一条随机
# 两条请求各 K=1 → 4 行：r0 的验证行 + r0 的 bonus 行 + r1 的验证行 + r1 的 bonus 行
mixed_logits = [[9.0, 0, 0, 0, 0], [0, 0, 9.0, 0, 0], [0, 9.0, 0, 0, 0], [0, 0, 0, 9.0, 0]]
mixed_meta = metadata_for([[0], [1]])   # r0 的草稿 0、r1 的草稿 1
mixed_sampling = SamplingMetadata(
    temperature=torch.tensor([0.0, 1.0]), all_greedy=False, all_random=False,
    top_k=None, top_p=None, generators={}, no_penalties=True,
    prompt_token_ids=[[], []], output_token_ids=[[], []], min_tokens=[0, 0],
    stop_token_ids=[[], []], spec_token_ids=[[0], [1]])
# 注入的均匀随机数按**扁平草稿位置**给（P 个），与 vLLM 内核的索引方式一致
out_mixed = sampler.forward(mixed_meta, torch.tensor(mixed_logits), None, mixed_sampling,
                            uniforms=torch.tensor([0.0, 0.5], dtype=torch.float64),
                            recoveries=torch.tensor([1, 1])).sampled_token_ids
check("4. greedy/random 混批：greedy 行按 argmax 判、random 行按 p/q 判（都不看对方的路径）",
      out_mixed.tolist() == [[0, 2], [1, 3]], str(out_mixed.tolist()))

# ------------------------------------------------ 5. min_tokens 在投机路径上同样生效

# 草稿就是停止 token（4）、target 的 argmax 也是它：min_tokens 没到就不该提交
# 两条请求各 K=1 → 4 行（各自 1 个验证行 + 1 个 bonus 行）
eos_logits = [[0.0, 0, 0, 0, 9.0], [0.0, 0, 0, 0, 9.0], [0.0, 0, 0, 0, 9.0]]
censor_sampling = SamplingMetadata(
    temperature=None, all_greedy=True, all_random=False, top_k=None, top_p=None,
    generators={}, no_penalties=True, prompt_token_ids=[[], []],
    output_token_ids=[[], []], min_tokens=[3, 0], stop_token_ids=[[4], [4]],
    spec_token_ids=[[4], [4]])
eos_logits.append([0.0, 0, 0, 0, 9.0])
out_censored = sampler.forward(metadata_for([[4], [4]]), torch.tensor(eos_logits), None,
                               censor_sampling).sampled_token_ids
check("5. min_tokens 未到时，停止 token 即使在草稿里也不会被提交（验证侧同样要屏蔽）",
      out_censored[0, 0].item() != 4 and out_censored[1, 0].item() == 4,
      f"未到 min_tokens 的行={out_censored[0, 0].item()}、已到的行={out_censored[1, 0].item()}")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
