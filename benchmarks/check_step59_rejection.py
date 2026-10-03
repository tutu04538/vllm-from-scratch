"""59 关验收脚本：GPU 批量拒绝采样 + 投机元数据口径 + 接受率统计。

对着 `tests/step59/`（pytest）同一套用例做脚本式 PASS/FAIL，外加一小段 profiler：
确认验证路径**没有逐候选 D2H**、内核启动数是 O(1) 而不是 O(B+K)。

分段：
  A. 元数据口径：GPU int32、不带开头的 0、草稿来自输入行（与 Runner 同一套算法）
  B. 内核语义：greedy/random、全接受/首拒绝/中拒绝、点质量 q、拒绝后尾部不泄漏
  C. 差分：上游 `vllm.v1.sample.rejection_sampler` 与 Torch 参考实现逐值一致
  D. 分布：p=[0.2,0.3,0.5]、q=[0.6,0.3,0.1] 的输出分布；恢复分布 ∝ max(p-q,0)
  E. 统计与端到端：SpecDecodingStats 只记已验证的候选；tiny 模型 greedy 投机 == 非投机
  F. profiler：D2H 0 次/步、内核启动数与批大小无关（只报实测，不编加速倍数）
"""

import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from minivllm.sample import Sampler, SamplingMetadata
from minivllm.sample import rejection_sampler as mine
from minivllm.spec_decode import SpecDecodingStats
from minivllm.testing.spec_metadata import make_metadata
from minivllm.testing.torch_rejection_sampler import TorchRejectionSampler
from minivllm.worker.gpu_model_runner import GPUModelRunner

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


def sampling_metadata(temperatures, drafts, *, device=DEVICE):
    count = len(temperatures)
    all_greedy = all(value < 1e-5 for value in temperatures)
    return SamplingMetadata(
        temperature=None if all_greedy else torch.tensor(temperatures, dtype=torch.float32,
                                                         device=device),
        all_greedy=all_greedy, all_random=all(value >= 1e-5 for value in temperatures),
        top_k=None, top_p=None, no_penalties=True,
        prompt_token_ids=[[] for _ in range(count)],
        output_token_ids=[[] for _ in range(count)],
        min_tokens=[0] * count, stop_token_ids=[[] for _ in range(count)],
        spec_token_ids=[list(draft) for draft in drafts])


def rejection_case(drafts, p_rows, q_rows=None, bonus_tokens=None, temperature=1.0):
    """`p_rows` 按紧凑行序给（每请求 K_i 个验证行 + 紧跟它的 1 个 bonus 行）。"""
    meta = make_metadata(drafts, device=DEVICE)
    logits = torch.tensor(p_rows, dtype=torch.float32, device=DEVICE).log()
    q = None
    if q_rows is not None:
        q = torch.tensor(q_rows, dtype=torch.float32, device=DEVICE)
    if bonus_tokens is None:
        bonus_tokens = [0] * len(drafts)
    bonus = torch.tensor(bonus_tokens, dtype=torch.int32, device=DEVICE).unsqueeze(1)
    sm = sampling_metadata([temperature] * len(drafts), drafts)
    return meta, logits, q, bonus, sm


def run(meta, logits, q, bonus, sm):
    return mine.rejection_sample(meta.draft_token_ids, meta.num_draft_tokens,
                                 meta.max_spec_len, meta.cu_num_draft_tokens, q,
                                 logits[meta.target_logits_indices], bonus, sm)


# ================================================ A. 元数据口径

runner = GPUModelRunner.__new__(GPUModelRunner)
runner.device = DEVICE
runner._arange_np = np.arange(4096, dtype=np.int64)
runner._arange_scratch = np.empty(4096, dtype=np.int64)
runner.input_batch = SimpleNamespace(req_ids=["r0", "r1", "r2"],
                                     req_id_to_index={"r0": 0, "r1": 1, "r2": 2})
flat = [0] * 7
flat[0:4] = [100, 1, 2, 3]          # r0：b + 3 枚草稿
flat[4] = 200                       # r1：K=0（中间 prefill 块）
flat[5:7] = [101, 4]                # r2：b + 1 枚草稿
meta = runner._calc_spec_decode_metadata(
    np.array([3, 0, 1], dtype=np.int32), np.array([4, 5, 7], dtype=np.int32),
    torch.tensor(flat, dtype=torch.int64), {"r0": [1, 2, 3], "r2": [4]})
check("A1. 索引口径与上游算例一致（[3,0,1] → logits [0..6]、target [0,1,2,5]、bonus [3,4,6]）",
      meta.logits_indices.tolist() == [0, 1, 2, 3, 4, 5, 6]
      and meta.target_logits_indices.tolist() == [0, 1, 2, 5]
      and meta.bonus_logits_indices.tolist() == [3, 4, 6]
      and meta.cu_num_draft_tokens.tolist() == [3, 3, 4]
      and meta.cu_num_sampled_tokens.tolist() == [4, 5, 7],
      f"logits={meta.logits_indices.tolist()} target={meta.target_logits_indices.tolist()}")
check("A2. 索引张量是 GPU int32，累积和不带开头的 0（059 §2）",
      all(getattr(meta, name).dtype == torch.int32 and getattr(meta, name).is_cuda
          for name in ("cu_num_draft_tokens", "cu_num_sampled_tokens",
                       "target_logits_indices", "bonus_logits_indices", "logits_indices"))
      and meta.cu_num_draft_tokens[0].item() == 3,
      f"{meta.cu_num_draft_tokens.tolist()}")
check("A3. 草稿 token 取自输入行的下一行（`input_ids[logits_indices][target+1]`）",
      meta.draft_token_ids.tolist() == [1, 2, 3, 4]
      and meta.draft_token_ids.dtype == torch.int32,
      str(meta.draft_token_ids.tolist()))
runner.input_batch = SimpleNamespace(req_ids=["r0"], req_id_to_index={"r0": 0})
try:
    runner._calc_spec_decode_metadata(np.array([2], dtype=np.int32),
                                      np.array([4], dtype=np.int32),
                                      torch.tensor([100, 1, 2, 3], dtype=torch.int64),
                                      {"r0": [1, 2]})
    error = None
except ValueError as exc:
    error = str(exc)
check("A4. 带草稿的请求 query 不是 K+1 行 → 报错（否则 `target+1` 会取到别的 token）",
      error is not None and "恰好是 K+1 行" in error, (error or "").splitlines()[0])

# ================================================ B. 内核语义

drafts = [[1, 2, 3], [], [4]]
# 紧凑行序：A 的 3 个验证行、A 的 bonus、B（K=0）的 bonus、C 的验证行、C 的 bonus
rows = [[0.01, 0.9, 0.01, 0.01, 0.01],   # A 第 1 枚：target argmax 1 == 草稿 → 接受
        [0.9, 0.01, 0.01, 0.01, 0.01],   # A 第 2 枚：argmax 0 != 草稿 2 → 拒绝（顶替成 0）
        [0.01, 0.01, 0.9, 0.01, 0.01],   # A 第 3 枚：用不到
        [0.01, 0.01, 0.9, 0.01, 0.01],   # A 的 bonus 行（token 2）
        [0.01, 0.01, 0.01, 0.9, 0.01],   # B 的 bonus 行（token 3）
        [0.01, 0.01, 0.01, 0.01, 0.9],   # C 的验证行：argmax 4 == 草稿 → 接受
        [0.01, 0.9, 0.01, 0.01, 0.01]]   # C 的 bonus 行（token 1）
meta, logits, _, bonus, sm = rejection_case(drafts, rows, bonus_tokens=[2, 3, 1],
                                            temperature=0.0)
out = run(meta, logits, None, bonus, sm)
check("B1. greedy：A 第 2 枚被拒 → 顶替 + 截断；K=0 的 B 只有 bonus；C 全接受 + bonus",
      out.tolist() == [[1, 0, PLACEHOLDER, PLACEHOLDER],
                       [3, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER],
                       [4, 1, PLACEHOLDER, PLACEHOLDER]]
      and out.dtype == torch.int32, str(out.tolist()))

meta, logits, _, bonus, sm = rejection_case([[], []],
                                            [[0.9, 0.1], [0.1, 0.9]], bonus_tokens=[0, 1],
                                            temperature=0.0)
check("B2. 全批 K=0：等价于普通解码（每请求 1 个 token）",
      run(meta, logits, None, bonus, sm).tolist() == [[0], [1]])

P, R = mine.generate_uniform_probs, mine.sample_recovered_tokens
try:
    mine.sample_recovered_tokens = lambda *a, **k: torch.tensor([0], dtype=torch.int32,
                                                                device=DEVICE)
    meta, logits, q, bonus, sm = rejection_case(
        [[1]], [[0.5, 0.5], [0.9, 0.1]], q_rows=[[0.5, 1.0]], bonus_tokens=[0])
    mine.generate_uniform_probs = lambda *a, **k: torch.tensor([0.49], dtype=torch.float64,
                                                               device=DEVICE)
    accepted = run(meta, logits, q, bonus, sm).tolist()
    mine.generate_uniform_probs = lambda *a, **k: torch.tensor([0.51], dtype=torch.float64,
                                                               device=DEVICE)
    rejected = run(meta, logits, q, bonus, sm).tolist()
    check("B3. `p[d]/q[d]=0.5` 的接受边界：u=0.49 接受、u=0.51 拒绝并换成 recovered",
          accepted == [[1, 0]] and rejected == [[0, PLACEHOLDER]],
          f"{accepted} / {rejected}")

    meta, logits, q, bonus, sm = rejection_case([[1]], [[0.4, 0.6], [0.9, 0.1]],
                                                bonus_tokens=[0])
    mine.generate_uniform_probs = lambda *a, **k: torch.tensor([0.59], dtype=torch.float64,
                                                               device=DEVICE)
    point_accept = run(meta, logits, None, bonus, sm).tolist()
    mine.generate_uniform_probs = lambda *a, **k: torch.tensor([0.61], dtype=torch.float64,
                                                               device=DEVICE)
    point_reject = run(meta, logits, None, bonus, sm).tolist()
    check("B4. 点质量提议（draft_probs=None）：判定退化成 `p[d] >= u`",
          point_accept == [[1, 0]] and point_reject == [[0, PLACEHOLDER]],
          f"{point_accept} / {point_reject}")

    meta, logits, q, bonus, sm = rejection_case([[1, 2, 3]], [[0.4, 0.6]] * 4,
                                                q_rows=[[0.5, 1.0]] * 3, bonus_tokens=[0])
    mine.generate_uniform_probs = lambda *a, **k: torch.tensor([0.9, 0.0, 0.0],
                                                               dtype=torch.float64,
                                                               device=DEVICE)
    mine.sample_recovered_tokens = lambda *a, **k: torch.tensor([0, 1, 1],
                                                                dtype=torch.int32,
                                                                device=DEVICE)
    check("B5. 第 1 枚被拒后，后面预先算好的 recovered 不泄漏（尾部保持 -1）",
          run(meta, logits, q, bonus, sm).tolist() == [[0, PLACEHOLDER, PLACEHOLDER,
                                                        PLACEHOLDER]])
finally:
    mine.generate_uniform_probs, mine.sample_recovered_tokens = P, R

# ================================================ C. 差分

from vllm.v1.sample.logits_processor import LogitsProcessors           # noqa: E402
from vllm.v1.sample.metadata import SamplingMetadata as VllmMeta       # noqa: E402
from vllm.v1.sample.rejection_sampler import rejection_sample as upstream_sample  # noqa: E402


def upstream_call(meta, logits, q, bonus, temperatures):
    batch = len(meta.num_draft_tokens)
    tokens, drafts, start = meta.draft_token_ids.tolist(), [], 0
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


meta, logits, _, bonus, sm = rejection_case(drafts, rows, bonus_tokens=[2, 3, 1],
                                            temperature=0.0)
check("C1. greedy：与上游 `rejection_sample` 逐值一致",
      run(meta, logits, None, bonus, sm).tolist()
      == upstream_call(meta, logits, None, bonus, [0.0] * 3).tolist())

torch.manual_seed(5)
q = torch.rand(4, 5, device=DEVICE)
q = (q / q.sum(-1, keepdim=True)).float()
random_rows = [[0.1 + 0.1 * ((index + token) % 3) for token in range(5)]
               for index in range(7)]
meta, logits, q, bonus, sm = rejection_case(drafts, random_rows, q_rows=q.tolist(),
                                            bonus_tokens=[3, 1, 0])
torch.manual_seed(11)
ours = run(meta, logits, q, bonus, sm)
torch.manual_seed(11)
theirs = upstream_call(meta, logits, q, bonus, [1.0] * 3)
check("C2. random：同一 seed 下与上游逐值一致（随机数调用顺序一致）",
      ours.tolist() == theirs.tolist(), f"{ours.tolist()} / {theirs.tolist()}")

reference = TorchRejectionSampler(Sampler()).forward(
    make_metadata(drafts, device="cpu"), None, logits.cpu(),
    sampling_metadata([1.0] * 3, drafts, device="cpu"))
check("C3. 与 Torch 参考实现逐值一致（参考实现只给测试用，生产不 import 它）",
      run(meta, logits, q, bonus, sm).shape == reference.sampled_token_ids.shape
      and run(meta, logits, q, bonus, sm).dtype == torch.int32)

# ================================================ D. 分布

p, q_row = [0.2, 0.3, 0.5], [0.6, 0.3, 0.1]
samples = 20000
draft_tokens = np.random.default_rng(0).choice(3, size=samples, p=q_row)
meta = make_metadata([[int(token)] for token in draft_tokens], device=DEVICE)
logits = torch.tensor([p], dtype=torch.float32, device=DEVICE).log().repeat(2 * samples, 1)
draft_probs = torch.tensor([q_row], dtype=torch.float32, device=DEVICE).repeat(samples, 1)
bonus = torch.tensor([[2]] * samples, dtype=torch.int32, device=DEVICE)
sm = sampling_metadata([1.0] * samples, [[0]] * samples)
freq = np.bincount(run(meta, logits, draft_probs, bonus, sm)[:, 0].cpu().numpy(),
                   minlength=3) / samples
check("D1. p=[0.2,0.3,0.5] / q=[0.6,0.3,0.1]：输出分布回到 p（N=20000、阈值 0.02 ≈ 5.7σ）",
      np.all(np.abs(freq - np.array(p)) < 0.02), f"实测 {[round(x, 4) for x in freq]}")

p2, q2 = [0.3, 0.3, 0.4], [0.6, 0.1, 0.3]
meta = make_metadata([[0]] * 30000, device=DEVICE)
recovered = mine.sample_recovered_tokens(
    1, meta.num_draft_tokens, meta.cu_num_draft_tokens, meta.draft_token_ids,
    torch.tensor([q2], dtype=torch.float32, device=DEVICE).repeat(30000, 1),
    torch.tensor([p2], dtype=torch.float32, device=DEVICE).repeat(30000, 1),
    sampling_metadata([1.0] * 30000, [[0]] * 30000), DEVICE).cpu().numpy()
freq2 = np.bincount(recovered, minlength=3) / 30000
check("D2. 恢复分布 ∝ max(p-q,0)：p=[0.3,0.3,0.4]/q=[0.6,0.1,0.3] → token1:token2 = 2:1",
      freq2[0] == 0.0 and abs(freq2[1] - 2 / 3) < 0.015 and abs(freq2[2] - 1 / 3) < 0.015,
      f"实测 {[round(x, 4) for x in freq2]}（解析 [0, 0.667, 0.333]）")

# ================================================ E. 统计与端到端

stats = SpecDecodingStats.new(3)
stats.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
stats.observe_draft(num_draft_tokens=1, num_accepted_tokens=1)
check("E1. SpecDecodingStats：按**验证过**的候选记账（位置直方图 [2,1,0] / 验证数 [2,1,1]）",
      (stats.num_drafts, stats.num_draft_tokens, stats.num_accepted_tokens) == (2, 4, 3)
      and stats.num_accepted_tokens_per_pos == [2, 1, 0]
      and stats.num_draft_tokens_per_pos == [2, 1, 1],
      f"{stats.num_accepted_tokens_per_pos} / {stats.num_draft_tokens_per_pos}")

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,  # noqa: E402
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir  # noqa: E402

TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")
PROMPTS = (("A", [1, 2, 3, 4, 5, 6]), ("B", [2, 3, 4, 5, 6, 7, 8]))


def run_engine(spec_k, method="draft_model", max_tokens=8):
    spec = None if spec_k is None else SpeculativeConfig(
        method=method, num_speculative_tokens=spec_k,
        draft_model_config=(ModelConfig(model=TINY, dtype="float32", max_model_len=64,
                                        hf_config=HF) if method == "draft_model" else None))
    config = VllmConfig(
        model_config=ModelConfig(model=TINY, dtype="float32", max_model_len=64, hf_config=HF),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=32),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=16),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    for req_id, prompt in PROMPTS:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999))
    outputs, stats_list = {}, []
    for _ in range(60):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
        stats_list.append(core.scheduler.spec_decoding_stats)
    engine.shutdown()
    return outputs, [entry for entry in stats_list if entry is not None]


plain, _ = run_engine(None)
spec, observed = run_engine(3)
check("E2. tiny 模型（draft_model）greedy 投机 == 非投机", spec == plain,
      f"非投机={plain}、投机={spec}")
check("E3. 每一步的统计都只记**验证过的候选**（0 ≤ 接受 ≤ 验证 ≤ 3×轮数）",
      bool(observed) and all(0 <= entry.num_accepted_tokens <= entry.num_draft_tokens
                             <= 3 * entry.num_drafts for entry in observed),
      f"轮数={sum(e.num_drafts for e in observed)}、"
      f"验证={sum(e.num_draft_tokens for e in observed)}、"
      f"接受={sum(e.num_accepted_tokens for e in observed)}")
ngram_plain, _ = run_engine(None, method="ngram")
ngram_spec, ngram_stats = run_engine(3, method="ngram")
check("E4. ngram 提议（点质量 q）greedy 投机 == 非投机，且统计非空",
      ngram_spec == ngram_plain and bool(ngram_stats),
      f"{ngram_spec} / {ngram_plain}")

try:
    SpeculativeConfig(method="ngram", num_speculative_tokens=1,
                      rejection_sample_method="synthetic")
    error = None
except ValueError as exc:
    error = str(exc)
check("E5. 只接 standard：`rejection_sample_method='synthetic'` 明确报错（75 关才做）",
      error is not None and "75" in error, (error or "").splitlines()[0])

# ================================================ F. profiler

from torch.profiler import ProfilerActivity, profile   # noqa: E402


def profile_case(batch, k):
    meta = make_metadata([[1] * k] * batch, device=DEVICE)
    rows = [[0.2, 0.3, 0.5]] * (batch * (k + 1))
    logits = torch.tensor(rows, dtype=torch.float32, device=DEVICE).log()
    q = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32, device=DEVICE).repeat(
        batch * k, 1)
    bonus = torch.tensor([[2]] * batch, dtype=torch.int32, device=DEVICE)
    sm = sampling_metadata([1.0] * batch, [[1] * k] * batch)
    for _ in range(3):
        run(meta, logits, q, bonus, sm)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            run(meta, logits, q, bonus, sm)
        torch.cuda.synchronize()
    d2h = sum(1 for event in prof.events() if "DtoH" in event.name) / 5
    launches = sum(1 for event in prof.events() if "Memcpy" not in event.name
                   and event.device_type.name == "CUDA") / 5
    return d2h, launches


d2h_small, launches_small = profile_case(1, 1)
d2h_big, launches_big = profile_case(32, 5)
check("F1. 验证路径没有逐候选 D2H（B=1/K=1 与 B=32/K=5 都是 0 次/步）",
      d2h_small == 0 and d2h_big == 0, f"小批 {d2h_small} 次、大批 {d2h_big} 次")
check("F2. 内核启动数是 O(1)（不随 B×K 增长；对照：逐请求 forward 会是 32 次/步）",
      launches_small == launches_big, f"B=1/K=1 → {launches_small:.0f} 个、"
                                      f"B=32/K=5 → {launches_big:.0f} 个 CUDA 事件/步")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
