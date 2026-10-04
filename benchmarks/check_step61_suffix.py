"""61 关验收脚本：Suffix Decoding 的请求内 / 跨请求历史（依赖 arctic_inference）。

分段（对应需求 061 §3/§4）：
  A. 依赖接入：版本/可导入性、缓冲类型口径（1 维连续 int32）
  B. 请求内：建 prompt 树 → 输出只追加一次 → 逐步候选（与**冻结的上游 proposer** 逐步比对）
  C. 跨请求：全局树命中、`max_cached_requests=0` 关全局树、容量溢出与 FIFO 淘汰、同 ID 重用
  D. 边界：partial prefill、离批/重进、到达 max_model_len、六步调用顺序
  E. 参数生效：K / max_spec_factor / min_token_prob / max_tree_depth 真的改变候选
  F. 端到端：tiny 模型 greedy 投机 == 非投机，且真的提了候选（错误候选不改最终答案）

不比对上游的部分会显式写明"反证/单侧"，其余每条都同时校验 `ours == upstream`。
"""

import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step59"))
sys.path.insert(0, str(ROOT / "tests" / "step61"))

from suffix_helpers import (FakeBatch, our_proposer, run_segments_on, run_trace,  # noqa: E402
                            run_trace_on, step59_helpers, upstream_proposer)

from minivllm.config import (ModelConfig, SpeculativeConfig, VllmConfig,  # noqa: E402
                             has_arctic_inference)
from minivllm.spec_decode.ngram_proposer import NgramProposer  # noqa: E402
from minivllm.spec_decode.suffix_decoding import _as_int32  # noqa: E402
from minivllm.spec_decode.utils import TargetRows  # noqa: E402
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 8
REPEAT_PROMPT = {"r0": [1, 2, 3, 4, 1, 2, 3, 4]}
CROSS = {"A": [1, 2, 3, 9, 9, 1, 2, 3], "B": [7, 8, 1, 2, 3]}
CROSS_STEPS = [{"order": ["A"], "sampled": {"A": [4]}},
               {"order": ["A"], "sampled": {"A": [5]}},
               {"order": ["B"], "sampled": {"B": [4]}}]
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def both(prompts, steps, *, k=K, max_model_len=64, **config):
    """同一串事件跑我们与上游两个提议者，返回 `(ours, upstream, 是否逐步一致)`。"""
    ours = run_trace_on("ours", max_model_len=max_model_len, prompts=prompts, steps=steps,
                        num_speculative_tokens=k, **config)
    upstream = run_trace_on("upstream", max_model_len=max_model_len, prompts=prompts, steps=steps,
                            num_speculative_tokens=k, **config)
    same = ours["drafts"] == upstream["drafts"] and ours["states"] == upstream["states"]
    return ours, upstream, same


# ================================================ A. 依赖接入
print("== A. 依赖接入 ==")
check("A1. `arctic_inference` 已安装且能被探测到",
      has_arctic_inference() and importlib.metadata.version("arctic_inference") == "0.3.0",
      f"version={importlib.metadata.version('arctic_inference')}")
check("A2. 本仓库缓冲 → 依赖包要求的 1 维连续 int32",
      (lambda array: array.dtype == np.int32 and array.ndim == 1
       and array.flags["C_CONTIGUOUS"])(_as_int32(torch.tensor([[1, 2, 3]], dtype=torch.int64)[0])))
import minivllm.config as _config_module  # noqa: E402

_original_has = _config_module.has_arctic_inference
_config_module.has_arctic_inference = lambda: False
try:
    SpeculativeConfig(method="suffix")
    missing_ok = False
except ImportError as error:
    missing_ok = "arctic-inference==0.1.1" in str(error)
finally:
    _config_module.has_arctic_inference = _original_has
check("A3. 未安装依赖时显式 ImportError（不静默降级）", missing_ok)

# ================================================ B. 请求内
print("\n== B. 请求内（prompt 树 + 输出追加 + 逐步候选）==")
steps = [{"order": ["r0"], "sampled": {"r0": [token]}} for token in (1, 2, 3, 4, 1)]
ours, upstream, same = both(REPEAT_PROMPT, steps)
check("B1. 五步逐步候选与上游一致", same, f"{ours['drafts']}")
check("B2. 每步都真的产出候选（非空、长度动态）",
      all(draft for step in ours["drafts"] for draft in step), f"{ours['drafts']}")
check("B3. 第一步候选 = prompt 里重复模式的后续 [2,3,4,1]",
      ours["drafts"][0] == [[2, 3, 4, 1]])
check("B4. 缓存状态与上游一致（活跃集 / 全局缓存集）", ours["states"] == upstream["states"],
      f"{ours['states'][-1]}")

# 输出只追加一次：同一批再调一次 propose 会把同样的 token 再追加一遍（所以一轮只能调一次）
probe = our_proposer(num_speculative_tokens=K, max_model_len=64)
batch = FakeBatch(1, 64)
histories = {"r0": [1, 2, 3, 4, 1, 2, 3, 4, 1]}
batch.layout(["r0"], histories, {"r0": 8})
first = probe.propose(K, batch, [[1]])
second = probe.propose(K, batch, [[1]])
check("B5. `add_active_response` 只加本轮采样：重复调用会重复追加（故一轮只调一次）",
      first[0] == [2, 3, 4, 1] and second[0] != first[0], f"{first[0]} → {second[0]}")

# ================================================ C. 跨请求
print("\n== C. 跨请求（全局树 / 容量 / FIFO / 同 ID 重用）==")
ours, upstream, same = both(CROSS, CROSS_STEPS)
check("C1. B 从全局树拿到 A 的『1 2 3 → 4 5』：[5]，逐步与上游一致",
      same and ours["drafts"][2] == [[5]], f"{ours['drafts']}")

ngram_config = VllmConfig(
    model_config=ModelConfig(model="fake/tiny", dtype="float32", max_model_len=64),
    speculative_config=SpeculativeConfig(method="ngram", num_speculative_tokens=K,
                                         prompt_lookup_min=3, prompt_lookup_max=3))
ngram_batch = FakeBatch(1, 64)
ngram_history = {"r0": [7, 8, 1, 2, 3, 4]}
ngram_batch.layout(["r0"], ngram_history, {"r0": 5})
ngram_rows = [TargetRows(req_id="r0", row=0, start=0, target_rows=1, num_rejected=0,
                         history_end=6, next_token_id=4, ready=True)]
ngram_drafts = NgramProposer(ngram_config).propose_drafts(
    ngram_rows, ngram_history, ngram_batch).draft_token_ids
check("C2. 同一份历史上 ngram（60 关）拿不到跨请求模式 → []（本关的痛点）",
      ngram_drafts == [[]], f"{ngram_drafts}")

closed, closed_up, closed_same = both(CROSS, CROSS_STEPS, max_cached_requests=0)
check("C3. `max_cached_requests=0`：跨请求候选消失，prompt 树照旧（逐步同上游）",
      closed_same and closed["drafts"][2] == [[]] and closed["states"][2] == ({"B"}, set()))
local, local_up, local_same = both(REPEAT_PROMPT, [{"order": ["r0"], "sampled": {"r0": [1]}}],
                                   max_cached_requests=0)
check("C4. 关掉全局树后请求内候选不受影响（[2,3,4,1]）",
      local_same and local["drafts"][0][0] == [2, 3, 4, 1])

fifo_prompts = {"A": [1, 2, 3, 9, 9, 1, 2, 3], "B": [7, 8, 1, 2, 3], "C": [9, 9, 1, 2, 3]}
fifo_steps = CROSS_STEPS + [{"order": ["C"], "sampled": {"C": [4]}}]
cap2, cap2_up, cap2_same = both(fifo_prompts, fifo_steps, max_cached_requests=2)
cap3, cap3_up, cap3_same = both(fifo_prompts, fifo_steps, max_cached_requests=3)
check("C5. 容量 2：C 进来先 FIFO 淘汰最早的 A → C 拿不到 [5]（逐步同上游）",
      cap2_same and cap2["drafts"][3] == [[]] and cap2["states"][3][1] == {"B", "C"})
check("C6. 容量 3：A 还在 → C 拿到 [5]",
      cap3_same and cap3["drafts"][3] == [[5]] and cap3["states"][3][1] == {"A", "B", "C"})

p1, p2 = [1, 2, 3, 4, 1, 2, 3, 4], [7, 7, 1, 2, 3, 4]
segments = [{"prompts": {"r0": p1},
             "steps": [{"order": ["r0"], "sampled": {"r0": [9]}},
                       {"order": ["r0"], "sampled": {"r0": [5]}}],
             "stop": ["r0"]},
            {"prompts": {"r0": p2}, "steps": [{"order": ["r0"], "sampled": {"r0": [9]}}]}]
reuse = run_segments_on("ours", max_model_len=64, segments=segments,
                        num_speculative_tokens=K)
reuse_up = run_segments_on("upstream", max_model_len=64, segments=segments,
                           num_speculative_tokens=K)
check("C7. 同 ID 重用：旧响应被 evict，新请求拿不到上一条的续写（逐步同上游）",
      reuse["drafts"] == reuse_up["drafts"] == [[[]], [[]], [[]]], f"{reuse['drafts']}")
stale_proposer = our_proposer(num_speculative_tokens=K, max_model_len=64)
run_trace(stale_proposer, max_model_len=64, prompts={"r0": p1}, steps=segments[0]["steps"],
          num_speculative_tokens=K)
stale_proposer.suffix_cache.evict_cached_response = lambda req_id: None  # 反证：不 evict
stale_proposer.suffix_cache.stop_request("r0")
stale = run_trace(stale_proposer, max_model_len=64, prompts={"r0": p2},
                  steps=segments[1]["steps"], num_speculative_tokens=K)
check("C8. 反证（单侧）：不 evict 时新请求会拿到旧请求的续写 [5]",
      stale["drafts"] == [[[5]]], f"{stale['drafts']}")

# ================================================ D. 边界与调用顺序
print("\n== D. 边界与调用顺序 ==")
prefill, prefill_up, prefill_same = both(
    REPEAT_PROMPT, [{"order": ["r0"], "sampled": {}, "visible": {"r0": 4}},
                    {"order": ["r0"], "sampled": {}, "visible": {"r0": 8}},
                    {"order": ["r0"], "sampled": {"r0": [1]}}])
check("D1. partial prefill：不建树、不追加、不提议；第一枚输出后才开始猜",
      prefill_same and prefill["drafts"][:2] == [[[]], [[]]]
      and prefill["drafts"][2] == [[2, 3, 4, 1]] and prefill["states"][0] == (set(), set()))
absent, absent_up, absent_same = both(
    {"rA": [1, 2, 3, 4, 1, 2, 3, 4], "rB": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]},
    [{"order": ["rA", "rB"], "sampled": {"rA": [1], "rB": [1]}},
     {"order": ["rA"], "sampled": {"rA": [1]}},                    # rB 离批（被抢占/预算不够）
     {"order": ["rB", "rA"], "sampled": {"rB": [1], "rA": [1]}}])  # rB 重新进入
check("D2. 离批 → 末尾 stop（活跃集只剩 rA），重新进入 → 重建树；逐步同上游",
      absent_same and absent["states"][1][0] == {"rA"} and absent["states"][2][0] == {"rB", "rA"})
at_limit, at_limit_up, at_limit_same = both({"rA": [1, 2, 3, 4, 1], "rB": [1, 2, 3]},
                                            [{"order": ["rA", "rB"], "sampled": {"rA": [9], "rB": [1]}}],
                                            max_model_len=6)
check("D3. 到达 max_model_len 的行：候选为空且**不建树**",
      at_limit_same and at_limit["drafts"][0][0] == []
      and at_limit["states"][0] == ({"rB"}, {"rB"}))

order_probe = our_proposer(num_speculative_tokens=K, max_model_len=64)
cache = order_probe.suffix_cache
names = ("start_request", "evict_cached_response", "add_active_response", "speculate",
         "stop_request")
originals = {name: getattr(cache, name) for name in names}
calls = []


def _wrap(name):
    original = originals[name]

    def recorded(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)

    return recorded


for _name in names:
    setattr(cache, _name, _wrap(_name))
run_trace(order_probe, max_model_len=64, prompts=REPEAT_PROMPT,
          steps=[{"order": ["r0"], "sampled": {"r0": [1]}}], num_speculative_tokens=K)
first_round = list(calls)
calls.clear()
run_trace(order_probe, max_model_len=64, prompts=REPEAT_PROMPT,
          steps=[{"order": [], "sampled": {}}], num_speculative_tokens=K)
absent_round = list(calls)
for _name, _original in originals.items():
    setattr(cache, _name, _original)
check("D4. §3 六步顺序：start → add → speculate（空采样什么都不调；离批末尾 stop）",
      first_round == ["start_request", "add_active_response", "speculate"]
      and absent_round == ["stop_request"], f"{first_round} / {absent_round}")

# ================================================ E. 参数生效
print("\n== E. 参数生效 ==")
factor_runs = {factor: both(REPEAT_PROMPT, [{"order": ["r0"], "sampled": {"r0": [1]}}],
                            max_spec_factor=factor)[0]["drafts"][0][0]
               for factor in (1.0, 0.5, 0.1)}
check("E1. `max_spec_factor` 控制长度：1.0→4 枚、0.5→2 枚、0.1→0 枚",
      factor_runs == {1.0: [2, 3, 4, 1], 0.5: [2, 3], 0.1: []}, f"{factor_runs}")
prob_prompt = {"r0": [7, 7, 1, 2, 3, 1, 2, 3, 5, 1, 2, 3, 5]}
prob_runs = {prob: both(prob_prompt, [{"order": ["r0"], "sampled": {"r0": [3]}}],
                        min_token_prob=prob)[0]["drafts"][0][0]
             for prob in (0.1, 0.5, 0.51)}
check("E2. `min_token_prob` 控制概率过滤：0.5→[5]，0.51→[]",
      prob_runs == {0.1: [5], 0.5: [5], 0.51: []}, f"{prob_runs}")
depth_runs = {depth: both({"r0": [7, 7, 1, 2, 3, 4, 1, 2, 3, 4]},
                          [{"order": ["r0"], "sampled": {"r0": [1]}}],
                          max_tree_depth=depth)[0]["drafts"][0][0]
              for depth in (24, 3)}
check("E3. `max_tree_depth` 改变上下文与匹配：24→4 枚、3→1 枚",
      depth_runs == {24: [2, 3, 4, 1], 3: [2]}, f"{depth_runs}")
k_runs = {k: both(REPEAT_PROMPT, [{"order": ["r0"], "sampled": {"r0": [1]}}],
                  k=k)[0]["drafts"][0][0] for k in (1, 2, 8)}
check("E4. `num_speculative_tokens` 是上限：1→[2]、2→[2,3]、8→[2,3,4,1]",
      k_runs == {1: [2], 2: [2, 3], 8: [2, 3, 4, 1]}, f"{k_runs}")

# ================================================ F. 端到端
print("\n== F. 端到端（tiny 模型，greedy）==")
tiny_dir = tiny_qwen3_dir("tiny_gqa")
hf_config = tiny_qwen3_config("tiny_gqa")
prompts = [("a", [1, 2, 3, 4, 1, 2, 3, 4]), ("b", [5, 6, 5, 6, 5, 6])]
spec_outputs, stats = step59_helpers.run_prompts(
    tiny_dir=tiny_dir, hf_config=hf_config, prompts=prompts, max_tokens=8, temperature=0.0,
    method="suffix", spec_k=8, device=DEVICE, collect_stats=True)
plain_outputs, _ = step59_helpers.run_prompts(
    tiny_dir=tiny_dir, hf_config=hf_config, prompts=prompts, max_tokens=8, temperature=0.0,
    spec_k=None, device=DEVICE)
hits = [entry for entry in stats if entry is not None]
drafts = sum(entry.num_drafts for entry in hits)
draft_tokens = sum(entry.num_draft_tokens for entry in hits)
accepted = sum(entry.num_accepted_tokens for entry in hits)
check("F1. greedy 下投机输出 == 非投机输出（错误候选不改最终答案）",
      spec_outputs == plain_outputs, f"{spec_outputs}")
check("F2. 真引擎里提了候选（drafts>0 且 draft_tokens>0）",
      drafts > 0 and draft_tokens > 0, f"drafts={drafts} tokens={draft_tokens} accepted={accepted}")
check("F3. 候选确实被 target 采纳（accepted ≥ 1，不是『提了就丢』）", accepted >= 1,
      f"accepted={accepted}/{draft_tokens}")

print()
print(f"设备={DEVICE}  依赖=arctic_inference {importlib.metadata.version('arctic_inference')}")
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
