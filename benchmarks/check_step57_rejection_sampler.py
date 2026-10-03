"""57E 验收（59 关改造后重写）：拒绝采样的算法语义 + 与上游/参考实现的差分。

59 关把生产入口搬到了 `minivllm/sample/rejection_sampler.py`，实现换成 **Triton 批量内核**
（上游同一条路径），Torch 版只作为参考实现留在 `minivllm/testing/`。用例没变，但改为对着
**新入口**跑，并且：

  - 注入随机数（199 §8 允许）逐值核对 `min(1, p/q)` 与 recovered 的选择；
  - 上游 `vllm.v1.sample.rejection_sampler.rejection_sample` 与 Torch 参考实现各做一遍差分；
  - 顺手确认内核路径**没有逐候选 D2H**（059 §3.6）。

完整用例集在 `tests/step59/`（pytest）。
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from minivllm.sample import Sampler, SamplingMetadata, expand_batch_to_tokens
from minivllm.sample import rejection_sampler as mine
from minivllm.testing.spec_metadata import make_metadata
from minivllm.testing.torch_rejection_sampler import TorchRejectionSampler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLACEHOLDER = -1
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


if DEVICE != "cuda":
    print("拒绝采样内核需要 CUDA（上游同样只有 GPU 路径）：本机没有 CUDA")
    sys.exit(1)


def sampling_metadata(temperatures, drafts, *, device=DEVICE, output_token_ids=None):
    count = len(temperatures)
    all_greedy = all(value < 1e-5 for value in temperatures)
    return SamplingMetadata(
        temperature=None if all_greedy else torch.tensor(temperatures, dtype=torch.float32,
                                                         device=device),
        all_greedy=all_greedy, all_random=all(value >= 1e-5 for value in temperatures),
        top_k=None, top_p=None, no_penalties=True,
        prompt_token_ids=[[] for _ in range(count)],
        output_token_ids=output_token_ids or [[] for _ in range(count)],
        min_tokens=[0] * count, stop_token_ids=[[] for _ in range(count)],
        spec_token_ids=[list(draft) for draft in drafts])


def case(drafts, rows, *, q_rows=None, bonus_tokens=None, temperature=0.0):
    """`rows` 按**紧凑行序**给（每请求 K_i 个验证行 + 紧跟它自己的 1 个 bonus 行）。"""
    meta = make_metadata(drafts, device=DEVICE)
    logits = torch.tensor(rows, dtype=torch.float32, device=DEVICE).log()
    q = None
    if q_rows is not None:
        q = torch.tensor(q_rows, dtype=torch.float32, device=DEVICE)
    if bonus_tokens is None:
        bonus_tokens = [int(np.argmax(row)) for row in rows[-len(drafts):]]
    bonus = torch.tensor(bonus_tokens, dtype=torch.int32, device=DEVICE).unsqueeze(1)
    return meta, logits, q, bonus, sampling_metadata([temperature] * len(drafts), drafts)


def run(meta, logits, q, bonus, sm):
    return mine.rejection_sample(meta.draft_token_ids, meta.num_draft_tokens,
                                 meta.max_spec_len, meta.cu_num_draft_tokens, q,
                                 logits[meta.target_logits_indices], bonus, sm)


def one_hot(token, vocab):
    row = [0.01] * vocab
    row[token] = 0.9
    return row


# ------------------------------------------------ 1. greedy 路径的形状

meta, logits, _, bonus, sm = case([[1, 2, 3], [], [4]],
                                  [one_hot(1, 5), one_hot(0, 5), one_hot(3, 5),
                                   one_hot(2, 5), one_hot(3, 5), one_hot(4, 5), one_hot(1, 5)],
                                  bonus_tokens=[2, 3, 1])
out = run(meta, logits, None, bonus, sm)
check("1. A/B/C：A 第 2 枚被拒 → 用 target argmax 顶替、后面全丢；B（K=0）只有 bonus",
      out.tolist() == [[1, 0, PLACEHOLDER, PLACEHOLDER],
                       [3, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER],
                       [4, 1, PLACEHOLDER, PLACEHOLDER]],
      str(out.tolist()))
check("1. 输出是 int32 的 `[B, max_spec_len+1]`，无效位置填 -1",
      out.dtype == torch.int32 and tuple(out.shape) == (3, 4)
      and PLACEHOLDER == mine.PLACEHOLDER_TOKEN_ID, f"{out.dtype} {tuple(out.shape)}")

meta, logits, _, bonus, sm = case([[1, 2], [3]],
                                  [one_hot(1, 5), one_hot(2, 5), one_hot(0, 5),
                                   one_hot(3, 5), one_hot(4, 5)], bonus_tokens=[0, 4])
check("1. 全接受 → 追加 bonus token",
      run(meta, logits, None, bonus, sm).tolist() == [[1, 2, 0], [3, 4, PLACEHOLDER]])

meta, logits, _, bonus, sm = case([[1, 2, 3]],
                                  [one_hot(0, 5), one_hot(1, 5), one_hot(2, 5),
                                   one_hot(4, 5)], bonus_tokens=[4])
check("1. 首枚就被拒 → 只提交 1 个 token",
      run(meta, logits, None, bonus, sm).tolist() == [[0, PLACEHOLDER, PLACEHOLDER,
                                                       PLACEHOLDER]])

# ------------------------------------------------ 2. 形状退化与 ragged

meta, logits, _, bonus, sm = case([[], []], [one_hot(3, 5), one_hot(4, 5)],
                                  bonus_tokens=[3, 4])
check("2. 全批 K=0：等价于普通解码（输出就是 bonus 行）",
      run(meta, logits, None, bonus, sm).tolist() == [[3], [4]])

meta, logits, _, bonus, sm = case([[2]], [one_hot(2, 5), one_hot(1, 5)], bonus_tokens=[1])
check("2. B=1：形状 [1, 2]",
      run(meta, logits, None, bonus, sm).tolist() == [[2, 1]])

meta, logits, _, bonus, sm = case([[1, 2], [], [3], [4, 0, 2]],
                                  [one_hot(1, 5), one_hot(2, 5), one_hot(0, 5),
                                   one_hot(1, 5), one_hot(3, 5), one_hot(0, 5),
                                   one_hot(4, 5), one_hot(1, 5), one_hot(2, 5),
                                   one_hot(3, 5)], bonus_tokens=[0, 1, 0, 3])
ragged = run(meta, logits, None, bonus, sm)
check("2. ragged（K=[2,0,1,3]）：每行有效长度 = K_i+1，其余是 -1，互不串行",
      ragged.tolist() == [[1, 2, 0, PLACEHOLDER],
                          [1, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER],
                          [3, 0, PLACEHOLDER, PLACEHOLDER],
                          [4, 1, PLACEHOLDER, PLACEHOLDER]],
      str(ragged.tolist()))

# ------------------------------------------------ 3. random 路径（注入随机数）

P = mine.generate_uniform_probs
R = mine.sample_recovered_tokens
try:
    mine.sample_recovered_tokens = lambda *a, **k: torch.tensor([0], dtype=torch.int32,
                                                                device=DEVICE)
    meta, logits, q, bonus, sm = case([[1]], [[0.5, 0.5], [0.9, 0.1]], q_rows=[[0.5, 1.0]],
                                      bonus_tokens=[0], temperature=1.0)
    mine.generate_uniform_probs = lambda *a, **k: torch.tensor([0.49], dtype=torch.float64,
                                                               device=DEVICE)
    accepted = run(meta, logits, q, bonus, sm)
    mine.generate_uniform_probs = lambda *a, **k: torch.tensor([0.51], dtype=torch.float64,
                                                               device=DEVICE)
    rejected = run(meta, logits, q, bonus, sm)
    check("3. `p[d]/q[d]=0.5`：u=0.49 接受、u=0.51 拒绝（`min(1, p/q)` 的边界）",
          accepted.tolist() == [[1, 0]] and rejected.tolist() == [[0, PLACEHOLDER]],
          f"{accepted.tolist()} / {rejected.tolist()}")

    meta, logits, q, bonus, sm = case([[1]], [[0.5, 0.5], [0.9, 0.1]], q_rows=[[1.0, 0.0]],
                                      bonus_tokens=[0], temperature=1.0)
    mine.generate_uniform_probs = lambda *a, **k: torch.tensor([0.0], dtype=torch.float64,
                                                               device=DEVICE)
    check("3. q[d]=0 → 防御性拒绝（否则 p/q 出 NaN），用 recovered token 顶替",
          run(meta, logits, q, bonus, sm).tolist() == [[0, PLACEHOLDER]])

    mine.sample_recovered_tokens = R
    meta, logits, q, bonus, sm = case([[1, 2, 0]], [[0.2, 0.3, 0.5]] * 4,
                                      q_rows=[[0.2, 0.3, 0.5]] * 3, bonus_tokens=[2],
                                      temperature=1.0)
    check("3. q == p（draft 与 target 同分布）→ 全部接受并追加 bonus",
          run(meta, logits, q, bonus, sm).tolist() == [[1, 2, 0, 2]])

    meta, logits, q, bonus, sm = case([[1]], [[0.4, 0.6], [0.9, 0.1]], q_rows=None,
                                      bonus_tokens=[0], temperature=1.0)
    mine.sample_recovered_tokens = lambda *a, **k: torch.tensor([0], dtype=torch.int32,
                                                                device=DEVICE)
    mine.generate_uniform_probs = lambda *a, **k: torch.tensor([0.59], dtype=torch.float64,
                                                               device=DEVICE)
    point_mass_accept = run(meta, logits, None, bonus, sm)
    mine.generate_uniform_probs = lambda *a, **k: torch.tensor([0.61], dtype=torch.float64,
                                                               device=DEVICE)
    point_mass_reject = run(meta, logits, None, bonus, sm)
    check("3. 点质量提议（ngram，draft_probs=None）：判定退化成 `p[d] >= u`",
          point_mass_accept.tolist() == [[1, 0]]
          and point_mass_reject.tolist() == [[0, PLACEHOLDER]],
          f"{point_mass_accept.tolist()} / {point_mass_reject.tolist()}")

    meta, logits, q, bonus, sm = case([[1, 2, 3]], [[0.4, 0.6]] * 4,
                                      q_rows=[[0.5, 1.0]] * 3, bonus_tokens=[0],
                                      temperature=1.0)
    mine.generate_uniform_probs = lambda *a, **k: torch.tensor([0.9, 0.0, 0.0],
                                                               dtype=torch.float64,
                                                               device=DEVICE)
    mine.sample_recovered_tokens = lambda *a, **k: torch.tensor([0, 1, 1],
                                                                dtype=torch.int32,
                                                                device=DEVICE)
    check("3. 第 1 枚被拒 → 后面预先算好的 recovered 不许泄漏（尾部保持 -1）",
          run(meta, logits, q, bonus, sm).tolist() == [[0, PLACEHOLDER, PLACEHOLDER,
                                                        PLACEHOLDER]])
finally:
    mine.generate_uniform_probs = P
    mine.sample_recovered_tokens = R

# ------------------------------------------------ 4. 与上游 / 参考实现差分

from vllm.v1.sample.logits_processor import LogitsProcessors           # noqa: E402
from vllm.v1.sample.metadata import SamplingMetadata as VllmMeta       # noqa: E402
from vllm.v1.sample.rejection_sampler import rejection_sample as upstream_sample  # noqa: E402


def upstream_call(meta, logits, q, bonus, temperatures):
    batch = len(meta.num_draft_tokens)
    drafts, start = [], 0
    tokens = meta.draft_token_ids.tolist()
    for num in meta.num_draft_tokens:
        drafts.append(tokens[start:start + num])
        start += num
    vmeta = VllmMeta(
        temperature=None if all(t < 1e-5 for t in temperatures) else torch.tensor(
            temperatures, dtype=torch.float32, device=DEVICE),
        all_greedy=all(t < 1e-5 for t in temperatures),
        all_random=all(t >= 1e-5 for t in temperatures),
        top_p=None, top_k=None, generators={}, max_num_logprobs=None, no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch, device=DEVICE),
        presence_penalties=torch.zeros(batch, device=DEVICE),
        repetition_penalties=torch.ones(batch, device=DEVICE),
        output_token_ids=[[] for _ in range(batch)],
        allowed_token_ids_mask=None, bad_words_token_ids={},
        logitsprocs=LogitsProcessors(), spec_token_ids=drafts)
    return upstream_sample(meta.draft_token_ids, meta.num_draft_tokens, meta.max_spec_len,
                           meta.cu_num_draft_tokens, q, logits[meta.target_logits_indices],
                           bonus, vmeta)


drafts = [[1, 2, 3], [], [4]]
rows = [one_hot(1, 5), one_hot(0, 5), one_hot(3, 5), one_hot(2, 5),
        one_hot(3, 5), one_hot(4, 5), one_hot(1, 5)]
meta, logits, _, bonus, sm = case(drafts, rows, bonus_tokens=[2, 3, 1])
check("4. greedy：与上游 `rejection_sample` 逐值一致",
      run(meta, logits, None, bonus, sm).tolist()
      == upstream_call(meta, logits, None, bonus, [0.0, 0.0, 0.0]).tolist())

torch.manual_seed(7)
q = torch.rand(4, 5, device=DEVICE)
q = (q / q.sum(-1, keepdim=True)).float()
rows = [[0.1 + 0.1 * ((index + token) % 3) for token in range(5)] for index in range(7)]
meta, logits, q, bonus, sm = case(drafts, rows, q_rows=q.tolist(), bonus_tokens=[3, 1, 0],
                                  temperature=1.0)
torch.manual_seed(99)
ours = run(meta, logits, q, bonus, sm)
torch.manual_seed(99)
theirs = upstream_call(meta, logits, q, bonus, [1.0] * 3)
check("4. random：同一 seed 下与上游逐值一致（随机数调用顺序也一致）",
      ours.tolist() == theirs.tolist(), f"{ours.tolist()} / {theirs.tolist()}")

# 参考实现要在 CPU 上跑，用一份 CPU 的元数据副本 + 同一批 logits
greedy_rows = [one_hot(1, 5), one_hot(0, 5), one_hot(3, 5), one_hot(2, 5),
               one_hot(3, 5), one_hot(4, 5), one_hot(1, 5)]
meta_g, logits_g, _, bonus_g, sm_g = case(drafts, greedy_rows, bonus_tokens=[2, 3, 1])
reference = TorchRejectionSampler(Sampler()).forward(
    make_metadata(drafts, device="cpu"), None, logits_g.cpu(),
    sampling_metadata([0.0] * 3, drafts, device="cpu"))
check("4. greedy：与 Torch 参考实现逐值一致（参考实现只给测试用）",
      run(meta_g, logits_g, None, bonus_g, sm_g).tolist()
      == reference.sampled_token_ids.tolist(),
      f"{run(meta_g, logits_g, None, bonus_g, sm_g).tolist()}"
      f" / {reference.sampled_token_ids.tolist()}")

# ------------------------------------------------ 5. 交付边界与性能口径


def parse_case():
    from minivllm.sample import RejectionSampler
    padded = torch.tensor([[1, 0, PLACEHOLDER, PLACEHOLDER],
                           [3, 99, PLACEHOLDER, PLACEHOLDER]], dtype=torch.int32,
                          device=DEVICE)
    outputs, _ = RejectionSampler.parse_output(padded, vocab_size=10, discard_req_indices=[1])
    return outputs


check("5. `parse_output` 一次过滤 -1 与越界 id，并按行丢弃（中间 prefill 块）",
      parse_case() == [[1, 0], []], str(parse_case()))

meta, logits, q, bonus, sm = case([[1, 2, 3, 4, 5]] * 8,
                                  [[0.2, 0.3, 0.5]] * 48,
                                  q_rows=[[0.5, 0.3, 0.2]] * 40, temperature=1.0)
for _ in range(3):
    run(meta, logits, q, bonus, sm)
torch.cuda.synchronize()
from torch.profiler import ProfilerActivity, profile   # noqa: E402

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        run(meta, logits, q, bonus, sm)
    torch.cuda.synchronize()
d2h = sum(1 for event in prof.events() if "DtoH" in event.name) / 5
check("5. B=8、K=5（40 个候选）的内核路径没有任何 D2H（逐候选取值会变成 40+ 次）",
      d2h == 0, f"实测 {d2h} 次/步")

check("5. `expand_batch_to_tokens` 与上游逐值一致（温度/top-k/top-p 的按请求展开）",
      expand_batch_to_tokens(torch.tensor([0.0, 2.0], device=DEVICE),
                             torch.tensor([1, 4], dtype=torch.int32, device=DEVICE), 4,
                             replace_from=0, replace_to=1).tolist() == [1.0, 2.0, 2.0, 2.0])

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
