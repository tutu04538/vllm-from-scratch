"""60 关验收脚本：CPU/GPU ngram 提议与历史增量维护。

分段：
  A. CPU ngram 与上游冻结实现逐值一致（tie / min_n / max_model_len / 无匹配 / 重复全同）
  B. GPU ngram：匹配一致、固定宽度 + 有效个数、scatter 与"只读长度"
  C. GPU 历史：行重排 [A,B,C] → [C,A] → [D,C,A]、只拷新增、D2H 不随历史长度增长
  D. 调度侧对齐：`update_scheduler_for_invalid_drafts` 把占位裁到有效（-1 出不去）
  E. 端到端：tiny 模型上两条 ngram 路径 greedy 投机 == 非投机
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step59"))
sys.path.insert(0, str(ROOT / "tests" / "step60"))

from ngram_helpers import FakeInputBatch, make_config, make_engine, make_rows  # noqa: E402

from minivllm import (LLMEngine, SamplingParams, SpeculativeConfig, UniProcExecutor,  # noqa: E402
                      Worker)
from minivllm.spec_decode.ngram_proposer import (  # noqa: E402
    NgramProposer, _find_longest_matched_ngram_and_propose_tokens as ours_find)
from minivllm.spec_decode.ngram_proposer_gpu import (  # noqa: E402
    NgramProposerGPU, update_ngram_gpu_tensors_incremental,
    update_scheduler_for_invalid_drafts)
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir  # noqa: E402
from vllm.v1.spec_decode.ngram_proposer import (  # noqa: E402
    _find_longest_matched_ngram_and_propose_tokens as upstream_find,
    batch_propose_numba)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


class _Cfg:
    def __init__(self, k=3, min_n=2, max_n=4, max_model_len=64, max_num_seqs=4,
                 method="ngram"):
        self.speculative_config = SpeculativeConfig(
            method=method, num_speculative_tokens=k,
            prompt_lookup_min=min_n, prompt_lookup_max=max_n)
        self.model_config = type("M", (), {"max_model_len": max_model_len})()
        self.scheduler_config = type("S", (), {"max_num_seqs": max_num_seqs})()


# ================================================ A. CPU ngram

rng = np.random.default_rng(0)
cases = [([1, 2, 3, 1, 2, 9, 1, 2], 2, 2, 4), ([5, 5, 5, 5, 5, 5], 2, 5, 3),
         ([7, 8, 9, 1, 2, 3, 7, 8, 9], 5, 6, 2), ([1, 2], 5, 5, 3),
         ([1, 2, 3], 1, 3, 4)]
for _ in range(20):
    tokens = rng.integers(0, 4, size=int(rng.integers(1, 30))).tolist()
    low, high = sorted((int(rng.integers(1, 4)), int(rng.integers(1, 6))))
    cases.append((tokens, low, high, int(rng.integers(1, 5))))
diff = 0
for tokens, min_n, max_n, k in cases:
    array = np.array(tokens, dtype=np.int32)
    for mml in (10 ** 9, len(tokens) + 2, len(tokens)):
        if ours_find(array, min_n, max_n, mml, k).tolist() != \
                upstream_find(array, min_n, max_n, mml, k).tolist():
            diff += 1
check("A1. 单条序列匹配与上游逐值一致（含 max_model_len 边界）", diff == 0,
      f"{len(cases) * 3} 组，差异 {diff}")
check("A2. 同长度多处匹配取**最早**那处（60 关之前本机取的是最近一次）",
      ours_find(np.array([1, 2, 3, 1, 2, 9, 1, 2], dtype=np.int32), 2, 2, 10 ** 9,
                3).tolist() == [3, 1, 2])

proposer = NgramProposer(_Cfg(k=3, min_n=2, max_n=4, max_num_seqs=8))
batch, max_len = 6, 32
token_ids = np.zeros((batch, 64), dtype=np.int32)
lengths = np.zeros(batch, dtype=np.int32)
for i in range(batch):
    n = int(rng.integers(2, max_len))
    token_ids[i, :n] = rng.integers(0, 4, size=n)
    lengths[i] = n
valid = [0, 1, 3, 5]
theirs_draft = np.zeros((8, 3), dtype=np.int32)
theirs_num = np.zeros(8, dtype=np.int32)
batch_propose_numba(valid, lengths, token_ids, proposer.min_n, proposer.max_n,
                    proposer.max_model_len, 3, theirs_draft, theirs_num)
theirs = [theirs_draft[i, :theirs_num[i]].tolist() if i in valid else [] for i in range(batch)]
check("A3. `batch_propose` 与上游 numba 版逐值一致（只算 valid 行）",
      proposer.batch_propose(batch, valid, lengths, token_ids, 3) == theirs)

histories = [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]]
buf = FakeInputBatch(histories, 64)
prop2 = NgramProposer(_Cfg(k=2, min_n=2, max_n=2))
drafts = prop2.propose(2, [[7], [], [7]], buf.num_tokens_no_spec.numpy(),
                       buf.token_ids_cpu.numpy())
check("A4. 跳过规则：本轮没采样 → 不提（第 2 条为空）", drafts[1] == [] and drafts[0] == [1, 2],
      str(drafts))
lengths_full = buf.num_tokens_no_spec.numpy().copy()
lengths_full[2] = 64
check("A5. 跳过规则：已到 max_model_len → 不提",
      prop2.propose(2, [[7]] * 3, lengths_full, buf.token_ids_cpu.numpy())[2] == [])
check("A6. 协议入口尊重 `history_end`（不拿上一轮遗留的草稿区去匹配）",
      prop2.propose_drafts([dataclasses.replace(row, history_end=2)
                            for row in make_rows(histories)],
                           {f"r{i}": t for i, t in enumerate(histories)}).draft_token_ids[0] == [])

spec = SpeculativeConfig(method="ngram", num_speculative_tokens=3)
check("A7. `prompt_lookup_min/max` 默认 5/5，只给一个时另一个跟随",
      (spec.prompt_lookup_min, spec.prompt_lookup_max) == (5, 5)
      and (SpeculativeConfig(method="ngram", num_speculative_tokens=3,
                             prompt_lookup_max=8).prompt_lookup_min) == 8)

# ================================================ B. GPU ngram（匹配 + 宽度/有效）

gpu_prop = NgramProposerGPU(_Cfg(k=3, min_n=2, max_n=4, method="ngram_gpu"), torch.device(DEVICE))
tokens = [1, 2, 3, 1, 2, 9, 1, 2]
gpu_batch, gpu_ids, gpu_lens = FakeInputBatch([tokens], 64, dtype=torch.int32), None, None
gpu_ids = torch.zeros(1, 64, dtype=torch.int32, device=DEVICE)
gpu_lens = torch.zeros(1, dtype=torch.int32, device=DEVICE)
gpu_ids[0, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32, device=DEVICE)
gpu_lens[0] = len(tokens)
drafts_g, valid_g = gpu_prop.kernel(gpu_lens, gpu_ids,
                                    torch.ones(1, dtype=torch.bool, device=DEVICE))
expected = upstream_find(np.array(tokens, dtype=np.int32), gpu_prop.min_n, gpu_prop.max_n,
                         10 ** 9, gpu_prop.k).tolist()
check("B1. GPU 匹配与上游冻结实现逐值一致",
      drafts_g[0].tolist()[:len(expected)] == expected
      and int(valid_g[0]) == len(expected), f"{drafts_g[0].tolist()} vs {expected}")

narrow = NgramProposerGPU(_Cfg(k=4, min_n=2, max_n=2, method="ngram_gpu"),
                          torch.device(DEVICE))
ids2 = torch.zeros(1, 64, dtype=torch.int32, device=DEVICE)
ids2[0, :5] = torch.tensor([1, 2, 9, 1, 2], dtype=torch.int32, device=DEVICE)
lens2 = torch.tensor([5], dtype=torch.int32, device=DEVICE)
d2, v2 = narrow.kernel(lens2, ids2, torch.ones(1, dtype=torch.bool, device=DEVICE))
check("B2. 固定宽度 + 有效个数：宽度 4、有效 3，第 4 格是占位 -1",
      d2.shape == (1, 4) and d2[0].tolist() == [9, 1, 2, -1] and int(v2[0]) == 3,
      f"{d2[0].tolist()} / valid={int(v2[0])}")

before_lens = gpu_lens.clone()
sampled = torch.tensor([[9, 9]], dtype=torch.int32, device=DEVICE)
counts = torch.tensor([2], dtype=torch.int32, device=DEVICE)
gpu_prop.propose(3, gpu_lens, gpu_ids, sampled, counts)
check("B3. 新采样 token 落进历史，但**长度表不被写回**（只读输入）",
      gpu_ids[0, len(tokens):len(tokens) + 2].tolist() == [9, 9]
      and gpu_lens.tolist() == before_lens.tolist(),
      f"lens={gpu_lens.tolist()}")

# ================================================ C. GPU 历史增量

hist_prop = NgramProposerGPU(_Cfg(k=2, min_n=1, max_n=1, method="ngram_gpu",
                                  max_num_seqs=4), torch.device(DEVICE))
hist_batch = FakeInputBatch([], 64, dtype=torch.int32, max_num_reqs=4)
hist_ids = torch.zeros(4, 64, dtype=torch.int32, device=DEVICE)
hist_lens = torch.zeros(4, dtype=torch.int32, device=DEVICE)


def rebuild(rows, prev, new_ids):
    hist_batch.req_ids = [req_id for req_id, _ in rows]
    hist_batch.req_id_to_index = {req_id: i for i, (req_id, _) in enumerate(rows)}
    for i, (req_id, toks) in enumerate(rows):
        hist_batch.set_row(i, toks, req_id=req_id)
    update_ngram_gpu_tensors_incremental(hist_batch, hist_ids, hist_lens, new_ids, prev,
                                         DEVICE)
    return {req_id: i for i, (req_id, _) in enumerate(rows)}


state = rebuild([("A", [1, 1, 1]), ("B", [2, 2, 2]), ("C", [3, 3, 3])], None, {"A", "B", "C"})
rebuild([("C", [3, 3, 3]), ("A", [1, 1, 1])], state, set())
rebuild([("D", [4, 4, 4, 4]), ("C", [3, 3, 3]), ("A", [1, 1, 1])], {"C": 0, "A": 1}, {"D"})
check("C1. 批序列 [A,B,C]→[C,A]→[D,C,A]：历史/长度跟着请求走（不跟行号）",
      hist_ids[0, :4].tolist() == [4, 4, 4, 4] and hist_ids[1, :3].tolist() == [3, 3, 3]
      and hist_ids[2, :3].tolist() == [1, 1, 1] and hist_lens[:3].tolist() == [4, 3, 3],
      f"lens={hist_lens[:3].tolist()}")

if DEVICE == "cuda":
    from torch.profiler import ProfilerActivity, profile

    big = NgramProposerGPU(_Cfg(k=3, min_n=1, max_n=3, method="ngram_gpu",
                                max_model_len=1024), torch.device(DEVICE))
    big_ids = torch.zeros(1, 1024, dtype=torch.int32, device=DEVICE)
    big_ids[0, :512] = torch.tensor(([1, 2, 3, 4] * 128)[:512], dtype=torch.int32,
                                    device=DEVICE)
    big_lens = torch.tensor([512], dtype=torch.int32, device=DEVICE)
    one = torch.tensor([[7]], dtype=torch.int32, device=DEVICE)
    one_count = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    big.propose(3, big_lens, big_ids, one, one_count)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        big.propose(3, big_lens, big_ids, one, one_count)
        torch.cuda.synchronize()
    d2h = sum(1 for event in prof.events() if "DtoH" in event.name)
    check("C2. 提议本身没有 D2H（不把整段历史搬回 CPU；历史 512 token）", d2h == 0,
          f"实测 {d2h} 次")
else:
    check("C2. 提议本身没有 D2H（不把整段历史搬回 CPU）", False, "本机没有 CUDA")

# ================================================ D. 调度侧对齐

check("D1. `update_scheduler_for_invalid_drafts`：按有效数截断并滤掉哨兵",
      update_scheduler_for_invalid_drafts([1, 2, 3, -1], 2) == [1, 2]
      and update_scheduler_for_invalid_drafts([1, 2, 3, -1], 4) == [1, 2, 3]
      and update_scheduler_for_invalid_drafts([1, 2, 3], 0) == []
      and update_scheduler_for_invalid_drafts([1, 2, 3], None) == [1, 2, 3])

# ================================================ E. 端到端

TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")
PROMPTS = (("A", [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]), ("B", [2, 3, 2, 3, 2, 3, 9, 9]))


def run(method, spec_k, min_n=None, max_n=None):
    config = make_config(tiny_dir=TINY, hf_config=HF, spec_k=spec_k, method=method,
                         device=DEVICE, max_num_seqs=2)
    if spec_k is not None and min_n is not None:
        object.__setattr__(config.speculative_config, "prompt_lookup_min", min_n)
        object.__setattr__(config.speculative_config, "prompt_lookup_max", max_n)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    for req_id, prompt in PROMPTS:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=999))
    outputs, stats, sentinels = {}, [], []
    original = core.scheduler.update_draft_token_ids

    def spy(drafts):
        original(drafts)
        for request in list(core.scheduler.running):
            sentinels.extend(request.spec_token_ids)
    core.scheduler.update_draft_token_ids = spy
    for _ in range(60):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
        stats.append(core.scheduler.spec_decoding_stats)
    engine.shutdown()
    hits = [entry for entry in stats if entry is not None]
    return outputs, hits, sentinels


plain, _, _ = run("ngram", None)
cpu_out, cpu_stats, cpu_sent = run("ngram", 3, 1, 3)
gpu_out, gpu_stats, gpu_sent = run("ngram_gpu", 3, 1, 3)
check("E1. CPU ngram：greedy 投机 == 非投机，且真的提了草稿",
      cpu_out == plain and sum(s.num_draft_tokens for s in cpu_stats) > 0,
      f"{cpu_out}")
check("E2. GPU ngram：greedy 投机 == 非投机，且真的提了草稿",
      gpu_out == plain and sum(s.num_draft_tokens for s in gpu_stats) > 0,
      f"{gpu_out}")
check("E3. 两条提议路径输出一致（同模型同 prompt）", cpu_out == gpu_out)
check("E4. Scheduler 手里的 `spec_token_ids` 从不含哨兵 -1",
      bool(gpu_sent) and all(token >= 0 for token in gpu_sent), f"看到 {len(gpu_sent)} 个草稿")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
