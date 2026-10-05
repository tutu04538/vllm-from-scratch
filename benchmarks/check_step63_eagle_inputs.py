"""63 关（第一阶段）验收脚本：EAGLE 的第一遍输入对齐。

分段：
  A. 需求 §3 的两请求例子（逐元素）
  B. 逐请求边界：多请求不同长度 / 单 token 请求 / 全单 token 批
  C. 那个坑：补丁下标错一位 → A 的最后一格留成 B 的 token
  D. 特征与 positions 不动（含扩容行的位置）
  E. 与上游内核 `copy_and_expand_eagle_inputs_kernel(shift_input_ids=True)` 逐值差分（CUDA）
  F. 两条通路的物理布局差异（每条请求的 [shift 行 + 扩容行] 一致）

状态：**本关只有第一阶段完成**（模型适配与端到端待做，见 docs/step63_alignment.md §5）。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step63"))

from test_eagle_inputs import _run_upstream_kernel  # noqa: E402

from minivllm.spec_decode.utils import expand_draft_inputs  # noqa: E402
from minivllm.testing.eagle_inputs_ref import (PADDING_TOKEN_ID, DraftInputRows,  # noqa: E402
                                               eagle_first_pass_input_ids,
                                               expand_eagle_inputs_shifted)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# ================================================ A. 需求例子
print("== A. 需求 §3 的两请求例子 ==")
ids, indices = eagle_first_pass_input_ids([11, 12, 21, 22, 23], [13, 24], [0, 2, 5])
check("A1. [a1,a2,b1,b2,b3] + next [a3,b4] → [a2,a3,b2,b3,b4]",
      ids == [12, 13, 22, 23, 24], f"{ids}")
check("A2. 每条请求的最后一格是采样行", indices == [1, 4], f"{indices}")

# ================================================ B. 逐请求边界
print("\n== B. 逐请求边界 ==")
ids3, idx3 = eagle_first_pass_input_ids([1, 2, 3, 4, 5, 6], [30, 40, 60], [0, 3, 4, 6])
check("B1. 三条请求长度不同：每格只放自己的新 token",
      ids3 == [2, 3, 30, 40, 6, 60] and idx3 == [2, 3, 5], f"{ids3} / {idx3}")
ids1, idx1 = eagle_first_pass_input_ids([7], [8], [0, 1])
check("B2. 单 token 请求：那一格就是自己的新 token", ids1 == [8] and idx1 == [0])
ids_s, idx_s = eagle_first_pass_input_ids([7, 8, 9], [17, 18, 19], [0, 1, 2, 3])
check("B3. 全单 token 批", ids_s == [17, 18, 19] and idx_s == [0, 1, 2])

# ================================================ C. 那个坑
print("\n== C. 补丁下标错一位（需求点名的坑）==")
target = [11, 12, 21, 22, 23]
correct, _ = eagle_first_pass_input_ids(target, [13, 24], [0, 2, 5])
wrong = list(target)
wrong[:len(target) - 1] = target[1:]
for index, token in zip([2, 5], [13, 24]):        # 错：query_start_loc[1:]
    if index < len(wrong):
        wrong[index] = token
check("C1. 正确：A 的最后一格 = A 的新 token（13）", correct[1] == 13, f"{correct}")
check("C2. 错误：A 的最后一格留成了 B 的第一个 token（21）", wrong[1] == 21, f"{wrong}")

# ================================================ D. 特征 / positions 不动
print("\n== D. 特征与 positions 不动 ==")
rows = [DraftInputRows([1, 2, 3], start=10, next_token_id=30, num_rejected=0),
        DraftInputRows([4, 5], start=20, next_token_id=50, num_rejected=0)]
s_ids, s_pos, s_rej, s_idx = expand_eagle_inputs_shifted(rows)
check("D1. shift 后 token 序列 = [2,3,30,5,50]", s_ids == [2, 3, 30, 5, 50], f"{s_ids}")
check("D2. positions 与 target 逐行相同（含扩容行 = 最后一行位置）",
      s_pos == [10, 11, 12, 20, 21], f"{s_pos}")
check("D3. 采样行 = 扩容行", s_idx == [2, 4], f"{s_idx}")
check("D4. 被拒行仍在工作区里（padding + mask）",
      expand_eagle_inputs_shifted(
          [DraftInputRows([1, 2], start=0, next_token_id=20, num_rejected=1)])[0]
      == [2, 20, PADDING_TOKEN_ID])
unshifted = expand_draft_inputs(rows)
check("D5. 与不 shift 的展开相比只少一个 token（第一个被跳过）",
      s_ids[:2] == unshifted[0][1:3], f"shift={s_ids[:2]} unshift={unshifted[0][1:3]}")

# ================================================ E. 与上游内核差分
print("\n== E. 与上游内核差分 ==")
if DEVICE != "cuda":
    check("E1. 内核差分（shift_input_ids=True）", True, "跳过：内核只在 CUDA 上（与上游一致）")
else:
    krows = [DraftInputRows([11, 12], start=0, next_token_id=13, num_rejected=0),
             DraftInputRows([21, 22, 23], start=10, next_token_id=24, num_rejected=0)]
    ours = expand_eagle_inputs_shifted(krows)
    up = _run_upstream_kernel(krows, shift_input_ids=True, num_padding_slots_per_request=1)
    check("E1. 无被拒行：input_ids/positions/is_rejected/采样行 逐值一致",
          list(ours[0]) == up["input_ids"] and list(ours[1]) == up["positions"]
          and list(ours[2]) == up["is_rejected"] and list(ours[3]) == up["new_indices"],
          f"{ours[0]} vs {up['input_ids']}")
    krows2 = [DraftInputRows([1, 2, 3], start=0, next_token_id=40, num_rejected=2)]
    ours2 = expand_eagle_inputs_shifted(krows2)
    up2 = _run_upstream_kernel(krows2, shift_input_ids=True, num_padding_slots_per_request=1)
    check("E2. 带被拒行：逐值一致", list(ours2[0]) == up2["input_ids"]
          and list(ours2[2]) == up2["is_rejected"], f"{ours2[0]} vs {up2['input_ids']}")
    check("E3. 对照：58 关差分过的 False 分支仍是原样",
          list(expand_draft_inputs(krows)[0]) == _run_upstream_kernel(
              krows, shift_input_ids=False, num_padding_slots_per_request=1)["input_ids"])

# ================================================ F. 两条通路的布局差异
print("\n== F. 两条通路的物理布局差异 ==")
mixed = [DraftInputRows([11, 12], start=0, next_token_id=13, num_rejected=1),
         DraftInputRows([21, 22, 23], start=10, next_token_id=24, num_rejected=0)]
plain, _ = eagle_first_pass_input_ids([11, 12, 21, 22, 23], [13, 24], [0, 2, 5])
upmixed = _run_upstream_kernel(mixed, shift_input_ids=True, num_padding_slots_per_request=1) \
    if DEVICE == "cuda" else None
if upmixed is None:
    check("F1. 内核路径为被拒行留位、后续请求整块后移", True, "跳过：需要 CUDA")
else:
    check("F1. 内核路径为被拒行留位、后续请求整块后移",
          upmixed["input_ids"][:3] == [12, 13, PADDING_TOKEN_ID]
          and upmixed["input_ids"][3:] == [22, 23, 24], f"{upmixed['input_ids']}")
    check("F2. 不变量：每条请求的 [shift 行 + 扩容行] 相同",
          plain[:2] == upmixed["input_ids"][:2] and plain[2:] == upmixed["input_ids"][3:],
          f"plain={plain} kernel={upmixed['input_ids']}")

print()
print(f"设备={DEVICE}")
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
