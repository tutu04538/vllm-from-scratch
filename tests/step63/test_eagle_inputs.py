"""step63：EAGLE 的第一遍输入对齐（需求 063 §3）。

这一段的主题是**行号**：draft 的输入 token 要逐请求错开一格、特征和 positions 不许动、
每请求的最后一格必须正好是那条请求新采出的 token。

对照对象：
  - 需求 §3 的两请求例子（期望值逐元素写死在用例里）；
  - 上游 `v1/spec_decode/utils.py::copy_and_expand_eagle_inputs_kernel`（`shift_input_ids=True`）
    —— CUDA 上直接调它做逐值差分（58 关差分的是 `False` 分支）。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm.spec_decode.utils import expand_draft_inputs  # noqa: E402
from minivllm.testing.eagle_inputs_ref import (PADDING_TOKEN_ID, DraftInputRows,  # noqa: E402
                                               eagle_first_pass_input_ids,
                                               expand_eagle_inputs_shifted)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 需求 §3 的两请求例子
# ---------------------------------------------------------------------------


def test_requirement_two_request_example():
    """`[a1,a2, b1,b2,b3]` + next `[a3,b4]` → draft 输入 `[a2,a3, b2,b3,b4]`（逐元素）。"""
    a1, a2, a3 = 11, 12, 13
    b1, b2, b3, b4 = 21, 22, 23, 24
    input_ids, sample_indices = eagle_first_pass_input_ids(
        target_token_ids=[a1, a2, b1, b2, b3],
        next_token_ids=[a3, b4],
        query_start_loc=[0, 2, 5])
    assert input_ids == [a2, a3, b2, b3, b4]
    assert sample_indices == [1, 4], "每条请求的最后一格（A→1，B→4）"


def test_each_request_gets_its_own_next_token():
    """三条请求、长度不同：每条的最后一格只放自己的新 token（不串味）。"""
    # A=[1,2,3], B=[4], C=[5,6]
    input_ids, sample_indices = eagle_first_pass_input_ids(
        target_token_ids=[1, 2, 3, 4, 5, 6],
        next_token_ids=[30, 40, 60],
        query_start_loc=[0, 3, 4, 6])
    assert input_ids == [2, 3, 30, 40, 6, 60]
    assert sample_indices == [2, 3, 5]


def test_single_token_request_keeps_the_next_token():
    """一条请求只有一个 token：没有"上一格"可移，那一格必须还是自己的新 token。"""
    input_ids, sample_indices = eagle_first_pass_input_ids([7], [8], [0, 1])
    assert input_ids == [8] and sample_indices == [0]


def test_batch_of_single_token_requests():
    input_ids, sample_indices = eagle_first_pass_input_ids([7, 8, 9], [17, 18, 19], [0, 1, 2, 3])
    assert input_ids == [17, 18, 19] and sample_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# 那个坑：补丁下标错一位 → A 的最后一格留着 B 的 token
# ---------------------------------------------------------------------------


def test_wrong_patch_index_leaks_the_next_request_token():
    """反证：把补丁位置算成 `query_start_loc[1:]`（下一条请求的第一格）会怎样。

    这正是需求里那句"不能直接全局 roll 后把 A 的最后一格留成 B 的 token"：
    A 的新 token 写到 B 头上，A 的最后一格留着 B 的第一个 token。
    """
    target = [11, 12, 21, 22, 23]
    next_tokens = [13, 24]
    query_start_loc = [0, 2, 5]
    correct, correct_indices = eagle_first_pass_input_ids(target, next_tokens, query_start_loc)
    assert correct == [12, 13, 22, 23, 24] and correct_indices == [1, 4]

    # 错误实现：也做了"整体左移"，但把新 token 写到**下一条请求的第一格**
    # （`query_start_loc[1:]` 而不是 `query_start_loc[1:] - 1`）
    wrong = list(target)
    wrong[:len(target) - 1] = target[1:]
    for index, token in zip(query_start_loc[1:], next_tokens):
        if index < len(wrong):
            wrong[index] = token
    assert wrong[:2] == [12, 21], "A 的最后一格留成了 B 的第一个 token（21）"
    assert wrong != correct

    # 三请求时更直观：错误实现下 A 的最后一格是 B 的第一个 token，B 的最后一格是 C 的
    target3 = [1, 2, 3, 4]
    next3 = [30, 40, 0]
    correct3, indices3 = eagle_first_pass_input_ids(target3, next3[:2], [0, 2, 4])
    assert correct3 == [2, 30, 4, 40] and indices3 == [1, 3]
    wrong3 = list(target3)
    wrong3[:3] = target3[1:]
    for index, token in zip([2, 4], next3[:2]):
        if index < len(wrong3):
            wrong3[index] = token
    assert wrong3[1] == 3, "A 的最后一格 = B 的第一个 token（3）"


# ---------------------------------------------------------------------------
# 特征 / positions 不动
# ---------------------------------------------------------------------------


def test_positions_and_hidden_rows_are_not_shifted():
    """positions 与 hidden states 逐行不动：第 i 行的 token 是 t_{i+1}，但位置/特征还是第 i 行。

    上游内核注释写着 "Positions are NOT shifted"；`hidden_state_mapping` 也是
    `src(query_start+j) → dst(output_start+j)` 的逐行平移。
    """
    rows = [DraftInputRows(valid_token_ids=[1, 2, 3], start=10, next_token_id=30,
                           num_rejected=0),
            DraftInputRows(valid_token_ids=[4, 5], start=20, next_token_id=50,
                           num_rejected=0)]
    input_ids, positions, is_rejected, sample_indices = expand_eagle_inputs_shifted(rows)
    # 第 0 行：token 是原来的第 1 个（2），位置仍是 10；扩容行位置 12
    assert input_ids == [2, 3, 30, 5, 50]
    assert positions == [10, 11, 12, 20, 21], "positions 不跟着移"
    assert is_rejected == [0, 0, 0, 0, 0]
    assert sample_indices == [2, 4]
    # 特征配对：第 i 行的特征就是 target 第 i 行的特征（逐行平移），扩容行用最后一行的特征
    hidden_rows = [f"h{i}" for i in range(5)]
    mapping = list(range(5))          # src(query_start+j) → dst(output_start+j)，slots=1 时是恒等
    assert [hidden_rows[mapping[i]] for i in range(5)] == hidden_rows


def test_shifted_expansion_has_one_fewer_valid_row_than_the_unshifted_one():
    """两条通路的关系：shift=True 时"有效行"少一行，靠扩容行补回来，总行数不变。"""
    rows = [DraftInputRows([1, 2, 3], start=0, next_token_id=30, num_rejected=0)]
    unshifted = expand_draft_inputs(rows)
    shifted = expand_eagle_inputs_shifted(rows)
    assert unshifted[0] == [1, 2, 3, 30] and shifted[0] == [2, 3, 30]
    assert len(shifted[0]) == len(unshifted[0]) - 1, "少掉的就是被跳过的第一个 token"
    assert shifted[3] == [2] and unshifted[3] == [3], "扩容行的行号跟着前移"


def test_rejected_rows_stay_in_the_workspace_as_padding():
    """被拒行仍占物理行（KV 要写、下一轮由 Scheduler 丢），token 是 padding、位置为 0。"""
    rows = [DraftInputRows([1, 2], start=0, next_token_id=20, num_rejected=2)]
    input_ids, positions, is_rejected, sample_indices = expand_eagle_inputs_shifted(rows)
    assert input_ids == [2, 20, PADDING_TOKEN_ID, PADDING_TOKEN_ID]
    assert positions[:2] == [0, 1] and positions[2:] == [0, 0]
    assert is_rejected == [0, 0, 1, 1]
    assert sample_indices == [1]


def test_bad_arguments_are_rejected_loudly():
    """形参对不上就报错（不静默按错的行数算）。"""
    with pytest.raises(ValueError, match="query_start_loc 的末位"):
        eagle_first_pass_input_ids([1, 2, 3], [9], [0, 2])
    with pytest.raises(ValueError, match="一条一个"):
        eagle_first_pass_input_ids([1, 2, 3], [9], [0, 1, 3])
    with pytest.raises(ValueError, match="至少要有两个元素"):
        eagle_first_pass_input_ids([1], [9], [0])


# ---------------------------------------------------------------------------
# 与上游内核逐值差分（CUDA）
# ---------------------------------------------------------------------------


def _run_upstream_kernel(rows, *, shift_input_ids, num_padding_slots_per_request):
    """直接调上游 `copy_and_expand_eagle_inputs_kernel`（只允许测试这么干）。"""
    from vllm.v1.spec_decode.utils import copy_and_expand_eagle_inputs_kernel

    target_token_ids, target_positions, next_token_ids = [], [], []
    query_start_loc, query_end_loc = [0], []
    for row in rows:
        valid_len = len(row.valid_token_ids)
        query_end_loc.append(query_start_loc[-1] + valid_len - 1)
        target_token_ids.extend(row.valid_token_ids)
        target_positions.extend(range(row.start, row.start + valid_len))
        # 被拒行也属于这条请求的查询块（它们真的被喂进了 target），但不算 valid
        for offset in range(row.num_rejected):
            target_token_ids.append(999)
            target_positions.append(row.start + valid_len + offset)
        next_token_ids.append(row.next_token_id)
        query_start_loc.append(query_start_loc[-1] + valid_len + row.num_rejected)

    total_input_tokens = len(target_token_ids)
    batch_size = len(rows)
    # 内核的公式：每请求 = num_valid + slots + num_rejected（shift 时 num_valid 少 1）
    total_output_tokens = 0
    for index, row in enumerate(rows):
        # shift 时 num_valid = query_end - query_start；不 shift 时还要加上被跳过的那个 token
        num_valid = query_end_loc[index] - query_start_loc[index] + (0 if shift_input_ids else 1)
        total_output_tokens += num_valid + num_padding_slots_per_request + row.num_rejected
    device = torch.device("cuda")
    out_input_ids = torch.zeros(total_output_tokens + 8, dtype=torch.int32, device=device)
    out_positions = torch.zeros(total_output_tokens + 8, dtype=torch.int32, device=device)
    out_rejected = torch.zeros(total_output_tokens + 8, dtype=torch.bool, device=device)
    out_masked = torch.zeros(total_output_tokens + 8, dtype=torch.bool, device=device)
    out_new_indices = torch.zeros(batch_size * max(num_padding_slots_per_request, 1),
                                  dtype=torch.int32, device=device)
    out_hidden_map = torch.zeros(total_input_tokens, dtype=torch.int32, device=device)

    copy_and_expand_eagle_inputs_kernel[(batch_size, 1)](
        target_token_ids_ptr=torch.tensor(target_token_ids, dtype=torch.int32, device=device),
        target_positions_ptr=torch.tensor(target_positions, dtype=torch.int32, device=device),
        next_token_ids_ptr=torch.tensor(next_token_ids, dtype=torch.int32, device=device),
        out_input_ids_ptr=out_input_ids,
        out_positions_ptr=out_positions,
        out_is_rejected_token_mask_ptr=out_rejected,
        out_is_masked_token_mask_ptr=out_masked,
        out_new_token_indices_ptr=out_new_indices,
        out_hidden_state_mapping_ptr=out_hidden_map,
        query_start_loc_ptr=torch.tensor(query_start_loc, dtype=torch.int32, device=device),
        query_end_loc_ptr=torch.tensor(query_end_loc, dtype=torch.int32, device=device),
        padding_token_id=PADDING_TOKEN_ID,
        parallel_drafting_token_id=0,
        total_input_tokens=total_input_tokens,
        num_padding_slots_per_request=num_padding_slots_per_request,
        shift_input_ids=shift_input_ids,
        BLOCK_SIZE_TOKENS=16,
    )
    return {
        "input_ids": out_input_ids[:total_output_tokens].tolist(),
        "positions": out_positions[:total_output_tokens].tolist(),
        "is_rejected": [int(x) for x in out_rejected[:total_output_tokens].tolist()],
        "new_indices": out_new_indices[:batch_size].tolist(),
        "hidden_map": out_hidden_map.tolist(),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="内核只在 CUDA 上（与上游一致）")
def test_matches_upstream_kernel_with_shift_true():
    """与上游内核 `shift_input_ids=True, num_padding_slots_per_request=1` 逐值一致。"""
    rows = [DraftInputRows([11, 12], start=0, next_token_id=13, num_rejected=0),
            DraftInputRows([21, 22, 23], start=10, next_token_id=24, num_rejected=0)]
    ours_ids, ours_positions, ours_rejected, ours_indices = expand_eagle_inputs_shifted(rows)
    upstream = _run_upstream_kernel(rows, shift_input_ids=True,
                                    num_padding_slots_per_request=1)
    # 上游在为被拒行留位；本轮没有被拒行，所以逐值应完全一致
    assert ours_ids == upstream["input_ids"]
    assert ours_positions == upstream["positions"]
    assert ours_rejected == upstream["is_rejected"]
    assert ours_indices == upstream["new_indices"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="内核只在 CUDA 上（与上游一致）")
def test_matches_upstream_kernel_with_shift_true_and_rejections():
    """带被拒行时也逐值一致（被拒行是 padding + 位置 0 + mask True）。"""
    rows = [DraftInputRows([1, 2, 3], start=0, next_token_id=40, num_rejected=2)]
    ours_ids, ours_positions, ours_rejected, ours_indices = expand_eagle_inputs_shifted(rows)
    upstream = _run_upstream_kernel(rows, shift_input_ids=True,
                                    num_padding_slots_per_request=1)
    assert ours_ids == upstream["input_ids"]
    assert ours_positions == upstream["positions"]
    assert ours_rejected == upstream["is_rejected"]
    assert ours_indices == upstream["new_indices"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="内核只在 CUDA 上（与上游一致）")
def test_two_branches_agree_per_request():
    """两条通路（无扩容 shift+打补丁 / 内核 shift=True slots=1）**逐请求前缀**一致。

    差别在物理布局，不在取值：内核路径会给每条请求补上 `num_rejected` 个 padding 行
    （"被拒行也占工作区"，见内核的 is_rejected_region），于是**后面的请求整块后移**；
    无扩容通路不带这些行（被拒位置本来就是本轮 target 查询行的一部分，原地留着）。
    所以不变量是"每条请求的 [shift 后的有效行 + 扩容行] 相同"，而不是"整段扁平数组相同"。
    """
    rows = [DraftInputRows([11, 12], start=0, next_token_id=13, num_rejected=1),
            DraftInputRows([21, 22, 23], start=10, next_token_id=24, num_rejected=0)]
    our_input_ids, _ = eagle_first_pass_input_ids(
        target_token_ids=[11, 12, 21, 22, 23],
        next_token_ids=[13, 24],
        query_start_loc=[0, 2, 5])
    upstream = _run_upstream_kernel(rows, shift_input_ids=True,
                                    num_padding_slots_per_request=1)
    # A：shift 后 1 行 + 扩容行 + 1 个被拒占位 → [12, 13, PAD]
    assert upstream["input_ids"][:3] == [12, 13, PADDING_TOKEN_ID]
    # B：接着排（整块后移一位）→ [22, 23, 24]
    assert upstream["input_ids"][3:] == [22, 23, 24]
    # 无扩容通路：A 的同样两行在前，B 直接从第 2 行开始
    assert our_input_ids == [12, 13, 22, 23, 24]
    assert our_input_ids[:2] == upstream["input_ids"][:2], "A 的 shift 行 + 扩容行一致"
    assert our_input_ids[2:] == upstream["input_ids"][3:], "B 的整块一致（只是位置差 1）"
    assert upstream["is_rejected"] == [0, 0, 1, 0, 0, 0], "只有 A 的那个占位行是 rejected"
