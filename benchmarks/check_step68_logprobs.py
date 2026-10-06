"""68 关验收脚本（A）：logprobs 与采样约束的投机语义（需求 068 §3.1/§3.2/§3.5、§4）。

脚本式 PASS/FAIL（供回归用），口径与 `tests/step68/` 一致：

  A. 四种模式：贪心行手算 log_softmax / logits、名次、并列名次、全词表模式
  B. 非投机差分：与 site-packages 里真的 `Sampler.forward` 逐值比对（4 种模式）
  C. 投机索引："第 j 个位置读第 j 行"（接受 / 恢复 / bonus）、与上游
     `RejectionSampler._get_logprobs_tensors` 差分、投机+全词表的拒绝
  D. 截断：`parse_output` 用同一张 valid_mask 裁 token 与 logprobs、多请求各自的偏移
  E. 约束：min_p 算式与位置、逐行"假设历史"不重复应用、min_tokens 逐行屏蔽
  F. 未接入项：请求期显式拒绝（白名单 / bad words / logit_bias / 思考预算 /
     logprob_token_ids / prompt_logprobs / 超 max_logprobs）
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step68"))

import spec68_helpers as h  # noqa: E402
import test_spec_logprobs as logs  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as err:
        return True, str(err)[:80]
    except Exception as err:              # noqa: BLE001 —— 别的异常不算"预期失败"
        return False, f"{type(err).__name__}: {err}"
    return False, "没有抛异常"


# ---------------------------------------------------------------- A. 四种模式
logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, 0.5]])
hand_row0 = logs.hand_log_softmax([2.0, 1.0, 0.0, -1.0])
for mode, expected in (("raw_logprobs", hand_row0), ("raw_logits", [2.0, 1.0, 0.0, -1.0]),
                       ("processed_logprobs", hand_row0), ("processed_logits", [2.0, 1.0, 0.0, -1.0])):
    out = h.ours_sampler_output(
        logits, h.sampling_metadata([0.0, 0.0], [[], []], max_num_logprobs=2,
                                    logprobs_mode=mode))
    tensors = out.logprobs_tensors
    ok = (abs(tensors.logprobs[0, 0].item() - expected[0]) < 1e-6
          and tensors.selected_token_ranks.tolist() == [1, 1]
          and tensors.logprob_token_ids[0].tolist() == [0, 0, 1])
    check(f"A1. {mode}：选中 token 的 logprob/名次/列序都按手算", ok,
          f"got={tensors.logprobs[0,0].item():.6f} want={expected[0]:.6f}")

# 名次并列算同档
tie = h.ours_sampler_output(torch.tensor([[1.0, 1.0, 0.0, -1.0]]),
                            h.sampling_metadata([0.0], [[]], max_num_logprobs=1))
check("A2. 名次 = 大于等于选中 logprob 的个数（并列同档）",
      tie.logprobs_tensors.selected_token_ranks.tolist() == [2])

full = h.ours_sampler_output(
    torch.tensor([[2.0, 1.0, 0.0, -1.0]]),
    h.sampling_metadata([0.0], [[]], max_num_logprobs=-1))
check("A3. logprobs=-1：采样器交出整份分布（形状与上游一致、不排名次）",
      full.logprobs_tensors.logprobs.shape == (1, 4)
      and full.logprobs_tensors.logprob_token_ids.shape == (0,))

# ---------------------------------------------------------------- B. 非投机差分
diff_ok = []
for mode in logs.MODES:
    common = dict(output_token_ids=[[1], []], prompt_token_ids=[[0], [0]],
                  no_penalties=False, repetition_penalties=[1.1, 1.0],
                  presence_penalties=[0.0, 0.0], frequency_penalties=[0.0, 0.2])
    ours = h.ours_sampler_output(
        logits, h.sampling_metadata([0.0, 0.7], [[], []], max_num_logprobs=2,
                                    logprobs_mode=mode,
                                    generators={1: torch.Generator().manual_seed(7)},
                                    **common), mode=mode)
    up = h.upstream_sampler_output(
        logits, h.upstream_sampling_metadata([0.0, 0.7], [[], []], max_num_logprobs=2,
                                             generators={1: torch.Generator().manual_seed(7)},
                                             **common), mode=mode)
    same = (ours.sampled_token_ids.tolist() == up.sampled_token_ids.tolist()
            and ours.logprobs_tensors.logprob_token_ids.tolist()
            == up.logprobs_tensors.logprob_token_ids.tolist()
            and torch.allclose(ours.logprobs_tensors.logprobs,
                               up.logprobs_tensors.logprobs, atol=1e-6))
    diff_ok.append(same)
check("B1. 4 种模式与上游 `Sampler.forward` 逐值一致（同一 seed 连 token 都比）",
      all(diff_ok), f"{sum(diff_ok)}/4")

# ---------------------------------------------------------------- C. 投机索引
meta, spec_logits, sampler, rs = logs._spec_setup("raw_logprobs")
target = spec_logits[meta.target_logits_indices]
bonus = logs._bonus_logits_for(sampler, meta, spec_logits, "raw_logprobs")
sampled_all = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
got = rs._get_logprobs_tensors(2, meta, spec_logits, target, bonus, sampled_all)
expected_rows = [logs.hand_log_softmax(spec_logits[row].tolist())[token]
                 for row, token in enumerate([0, 1, 2, 3])]
check("C1. 第 j 个位置读第 j 行（接受候选 / bonus）",
      all(abs(got.logprobs[j, 0].item() - value) < 1e-5
          for j, value in enumerate(expected_rows)),
      f"got={[round(got.logprobs[j,0].item(),4) for j in range(4)]}")

# 拒绝 → 恢复 token 读同一个位置那一行
meta2, logits2, sampler2, rs2 = logs._spec_setup(
    "raw_logprobs",
    drafts=(3, 1, 2),
    rows=[[0.0, 0.0, 0.0, 5.0], [0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0],
          [5.0, 0.0, 0.0, 0.0]])
bonus2 = logs._bonus_logits_for(sampler2, meta2, logits2, "raw_logprobs")
got2 = rs2._get_logprobs_tensors(1, meta2, logits2, logits2[meta2.target_logits_indices],
                                 bonus2, torch.tensor([[3, 1, -1, -1]], dtype=torch.int32))
recovered_expected = logs.hand_log_softmax(logits2[1].tolist())[1]
wrong_row = logs.hand_log_softmax(logits2[3].tolist())[1]
check("C2. 被拒位置（恢复 token）也读它自己那一行，不是 bonus 行",
      abs(got2.logprobs[1, 0].item() - recovered_expected) < 1e-5
      and abs(got2.logprobs[1, 0].item() - wrong_row) > 1.0,
      f"got={got2.logprobs[1,0].item():.4f} want={recovered_expected:.4f}")

up_meta = h.upstream_spec_metadata([[0, 1, 2]], device="cpu")
from vllm.v1.sample.rejection_sampler import RejectionSampler as UpstreamRejection  # noqa: E402

up_rs = UpstreamRejection(sampler=__import__("vllm.v1.sample.sampler",
                                             fromlist=["Sampler"]).Sampler(
                                                 logprobs_mode="raw_logprobs"))
up_got = up_rs._get_logprobs_tensors(2, up_meta, spec_logits, target, bonus, sampled_all)
check("C3. 与上游 `_get_logprobs_tensors` 逐值一致",
      got.logprob_token_ids.tolist() == up_got.logprob_token_ids.tolist()
      and torch.allclose(got.logprobs, up_got.logprobs, atol=1e-6)
      and got.selected_token_ranks.tolist() == up_got.selected_token_ranks.tolist())

ok, detail = _raises(NotImplementedError, rs._get_logprobs_tensors, -1, meta, spec_logits,
                     target, bonus, sampled_all)
check("C4. 本项目当前对投机 + logprobs=-1 提前拒绝（⚠️ 属未对齐，不是上游行为）", ok, detail)
# 2026-10-06 独立复核：-1 在引擎入口（gpu_input_batch.py:435-440）就被归一化成 vocab_size，
# 用户请求到不了下面这条路径；只有"手搓 metadata"才会踩到它
ok, detail = _raises(RuntimeError, torch.topk, torch.randn(1, 4), -1, dim=-1)
check("C5. 手搓 metadata 才会踩到的失败路径（torch.topk(k=-1) 抛错）——非用户可达", ok, detail)

# ---------------------------------------------------------------- D. 截断
rows, lists = rs.parse_output(torch.tensor([[0, 1, -1, -1]], dtype=torch.int32),
                              vocab_size=4, logprobs_tensors=got)
check("D1. parse_output 用同一张 mask 同时裁 token 与 logprobs（尾部不漏出）",
      rows == [[0, 1]] and lists.cu_num_generated_tokens == [0, 2]
      and lists.logprob_token_ids.shape == (2, 3),
      f"rows={rows} cu={lists.cu_num_generated_tokens} shape={lists.logprob_token_ids.shape}")

# ---------------------------------------------------------------- E. 约束
probs = logs.hand_log_softmax([2.0, 1.0, 0.0])
probs = [__import__("math").exp(value) for value in probs]
masked = logs.Sampler.apply_min_p(torch.tensor([[2.0, 1.0, 0.0]]), torch.tensor([0.2]))
check("E1. min_p 手算：低于 max(p)×min_p 的 token 被打成 -inf",
      masked[0, 2].item() == float("-inf") and probs[2] < probs[0] * 0.2)

# 逐行假设历史（frequency penalty 按次数）——需要 Triton 的 expand 内核
if torch.cuda.is_available():
    kv, other = 2, 3
    spec_rows = logs.prob_rows([[2.0, 1.0, 2.0, 1.0], [2.0, 1.0, 2.0, 1.0]], device="cuda")
    sm = h.sampling_metadata([0.0], [[kv, other]], device="cuda", no_penalties=False,
                             output_token_ids=[[kv]], prompt_token_ids=[[]],
                             frequency_penalties=[1.0], presence_penalties=[0.0],
                             repetition_penalties=[1.0])
    processed = rs.apply_logits_processors(spec_rows.clone(), sm,
                                           logs.make_metadata([[kv, other]], device="cuda"))
    import math

    check("E2. 候选行 j 的假设历史 = 已提交 + 草稿前缀 [:j]，且只施加一次",
          abs(processed[0, kv].item() - (math.log(2.0) - 1.0)) < 1e-6
          and abs(processed[1, kv].item() - (math.log(2.0) - 2.0)) < 1e-6,
          f"row0={processed[0,kv].item():.4f} row1={processed[1,kv].item():.4f}")
else:
    check("E2. 候选行假设历史（需要 CUDA 的 expand 内核）", False, "本机没有 CUDA：待验")

# min_p：只作用在 bonus 行（上游 argmax 不变处理器只在 Sampler.sample() 里跑），候选行被跳过
if torch.cuda.is_available():
    from minivllm.sample.rejection_sampler import apply_sampling_constraints

    row = torch.tensor([[2.0, 1.0, 0.0, -1.0]], device="cuda")
    sm_minp = h.sampling_metadata([1.0], [[]], device="cuda", min_p=[1.0],
                                  logprobs_mode="processed_logits", max_num_logprobs=4)
    candidate = apply_sampling_constraints(
        row.clone(), torch.tensor([1], dtype=torch.int32, device="cuda"), sm_minp)
    bonus_out = logs.Sampler("processed_logits").forward(row.clone(), sm_minp)
    cols = bonus_out.logprobs_tensors.logprob_token_ids[0].tolist()
    vals = bonus_out.logprobs_tensors.logprobs[0].tolist()
    check("E3. min_p 只作用在 bonus 行：候选行不被掩码，bonus 行除 argmax 外全 -inf",
          torch.equal(candidate, row)
          and abs(vals[cols.index(0)] - 2.0) < 1e-6
          and all(v == float("-inf") for c, v in enumerate(vals) if cols[c] != 0),
          f"候选行={candidate.tolist()} bonus列={cols} bonus值={[round(v,2) if v != float('-inf') else '-inf' for v in vals]}")
else:
    check("E3. min_p 只作用在 bonus 行（候选行部分需要 CUDA 的 expand 内核）", False, "本机没有 CUDA：待验")

# ---------------------------------------------------------------- F. 未接入项
from minivllm import SamplingParams  # noqa: E402

for field, value in (("allowed_token_ids", [1]), ("bad_words", ["x"]),
                     ("logit_bias", {1: 0.5}), ("thinking_token_budget", 8),
                     ("logprob_token_ids", [1]), ("prompt_logprobs", 1)):
    ok, detail = _raises(NotImplementedError, lambda f=field, v=value: SamplingParams(**{f: v}))
    check(f"F1. 未接入项请求期拒绝：{field}", ok, detail)

ok, detail = _raises(ValueError, SamplingParams, logprobs=0)
check("F2. logprobs 取值校验（0 非法）", ok, detail)
ok, detail = _raises(ValueError, SamplingParams, min_p=2.0)
check("F3. min_p 取值校验（>1 非法）", ok, detail)

print()
print("口径说明（完整版见 docs/step68_alignment.md）：")
print("  · raw/processed 的区别只在**是否施加惩罚/温度/top-k/top-p**：掩码（语法）两份都带，")
print("    因为它是在采样器之前打到 logits 上的（上游同序）。")
print("  · 投机下「第 j 个位置读第 j 行」：接受候选读候选行、恢复 token 读同一个位置的行、")
print("    bonus 读 bonus 行；被拒的候选位多算一份，但由 parse_output 的同一张 mask 滤掉。")
print("  · ⚠️ 2026-10-06 独立复核推翻了两条「上游疑似 bug」（见 vllm_bugs/VERIFICATION_REPORT_20261006.md）：")
print("    processed_logprobs 的 bonus 位虽走了两次 log_softmax，但 log_softmax 幂等 → 数字正确；")
print("    logprobs=-1 在引擎入口就被归一化成 vocab_size → topk(k=-1) 那条路径用户不可达。")
print("    真正待修的是本项目缺这条归一化（非投机交付为空 / 投机提前拒绝）。")
print("  · 新增未复核项：min_p 只作用在 bonus 行，候选验证行不加掩码（E3 与 tests/step68 的钉住用例）。")
print(f"设备={DEVICE}；torch={torch.__version__}")
print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
