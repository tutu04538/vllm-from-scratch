"""step77：DSpark 的两种 query 布局（anchor-as-first 与 1+K fill-in）。

三条线：

    A. **与上游内核逐值差分**：同一个 `_prepare_dflash_inputs_kernel`、同一个
       `SAMPLE_FROM_ANCHOR` constexpr —— 两种布局各比 12 个输出（positions / sample rows /
       slots / seq_lens / 补齐）。
    B. **布局的定义**：anchor-as-first 是 K 个 query（每行预测下一个 token），fill-in 是 1+K
       （只有 mask 行采样、预测自己）。两者**草稿位置相同**，差别只在"谁坐在哪个位置上"。
    C. **提议者/调度侧派生的预算**：`num_query_per_req`、`_anchor_idx`、图的 decode 长度、
       `max_num_new_slots_for_drafting = K−1`、`num_lookahead_tokens = K`。

K=1 是边界（anchor-as-first 只剩一行、fill-in 只剩 mask 那一行）——单独钉。
"""

import pytest
import torch

from minivllm.config import CUDAGraphMode  # noqa: E402

from dspark_helpers import (DEVICE, make_engine, make_speculator,  # noqa: E402
                            make_spec_config)
from spec76_helpers import layout_case, run_our_v2_kernel, run_upstream_v2_kernel  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSpark 的输入组装是 Triton 内核（上游同款边界），只在 CUDA 上可执行")

MASK_TOKEN_ID = 7


def dspark_case(anchor: bool, *, k: int = 3, device: str = DEVICE):
    """两请求的布局事实：请求 0 有 4 行 target（位置 7..10，被拒 2 行），请求 1 有 2 行。"""
    case = layout_case(device)
    case.num_speculative_steps = k
    case.num_query_per_req = k if anchor else 1 + k
    return case


# ---------------- A. 与上游内核逐值差分 ----------------

@requires_cuda
def test_anchor_as_first_layout_matches_the_upstream_kernel():
    """anchor-as-first：K 个 query 行，**每行都被采样**、每行预测下一个 token。"""
    case = dspark_case(anchor=True)
    upstream = run_upstream_v2_kernel(case, sample_from_anchor=True)
    ours = run_our_v2_kernel(case, sample_from_anchor=True)
    for name, want in upstream.items():
        got = ours[name]
        assert got.dtype == want.dtype, name
        assert torch.equal(want, got), name

    # 位置/行/映射的**具体数字**（不是"看起来对"）：last_valid_pos = 8（请求 0）
    assert upstream["query_positions"][:3].tolist() == [9, 10, 11]
    assert upstream["input_ids"][:3].tolist() == [11, MASK_TOKEN_ID, MASK_TOKEN_ID]
    # 三行都采样：行号 = query 行，采样位置 = query 位置 + 1
    assert upstream["sample_indices"][:3].tolist() == [0, 1, 2]
    assert upstream["sample_pos"][:3].tolist() == [10, 11, 12]
    assert upstream["sample_idx_mapping"][:6].tolist() == [0, 0, 0, 5, 5, 5]
    # seq_lens = last_valid_pos + 1 + K（请求 0：8+1+3；请求 1：21+1+3）
    assert upstream["seq_lens"][:2].tolist() == [12, 25]
    assert upstream["query_start_loc"][:3].tolist() == [0, 3, 6]
    # 槽位按块表查：位置 9 → 块 2、偏移 1 → 槽 9
    assert upstream["query_slot_mapping"][:3].tolist() == [9, 10, 11]
    # 被拒的两行是无效上下文：位置 0、槽位 -1（既不读也不写）
    assert upstream["context_positions"][:4].tolist() == [7, 8, 0, 0]
    assert upstream["context_slot_mapping"][:4].tolist() == [7, 8, -1, -1]


@requires_cuda
def test_fill_in_layout_matches_the_upstream_kernel():
    """fill-in（`sample_from_anchor=False`）：1+K 个 query，只有 mask 行采样、预测自己。"""
    case = dspark_case(anchor=False)
    upstream = run_upstream_v2_kernel(case, sample_from_anchor=False)
    ours = run_our_v2_kernel(case, sample_from_anchor=False)
    for name, want in upstream.items():
        got = ours[name]
        assert got.dtype == want.dtype, name
        assert torch.equal(want, got), name

    assert upstream["query_positions"][:4].tolist() == [9, 10, 11, 12]
    assert upstream["input_ids"][:4].tolist() == [11, MASK_TOKEN_ID, MASK_TOKEN_ID,
                                                  MASK_TOKEN_ID]
    # 第 0 行是 bonus：不采样；后 3 行是 mask：采样位置 = 自己的位置
    assert upstream["sample_indices"][:3].tolist() == [1, 2, 3]
    assert upstream["sample_pos"][:3].tolist() == [10, 11, 12]
    assert upstream["seq_lens"][:2].tolist() == [13, 26]
    assert upstream["query_start_loc"][:3].tolist() == [0, 4, 8]


@requires_cuda
def test_both_layouts_draft_the_same_positions():
    """两种布局的**草稿位置必须一致**（10, 11, 12），差别只在 query 行怎么排。

    这条不变量是"布局可切换、草稿语义不变"的定义：anchor-as-first 用 K 行（第 0 行是真实
    bonus）预测 10..12；fill-in 用 1+K 行（第 1..3 行是 mask）预测同样的 10..12。
    """
    anchor_case, fill_case = dspark_case(anchor=True), dspark_case(anchor=False)
    anchor = run_our_v2_kernel(anchor_case, sample_from_anchor=True)
    fill = run_our_v2_kernel(fill_case, sample_from_anchor=False)
    num = 3 * 2   # 2 请求 × K=3
    assert anchor["sample_pos"][:num].tolist() == fill["sample_pos"][:num].tolist()
    assert anchor["sample_idx_mapping"][:num].tolist() == \
        fill["sample_idx_mapping"][:num].tolist()
    # query 行数不同（3 vs 4），于是 seq_lens 差 1
    assert fill["seq_lens"][0] - anchor["seq_lens"][0] == 1
    assert fill["query_start_loc"][1] - anchor["query_start_loc"][1] == 1
    # 采样的 query 行不同：anchor 的是 [0,1,2]，fill-in 的是 [1,2,3]
    assert anchor["sample_indices"][:3].tolist() == [0, 1, 2]
    assert fill["sample_indices"][:3].tolist() == [1, 2, 3]


@requires_cuda
@pytest.mark.parametrize("anchor", [True, False])
def test_k1_boundary(anchor):
    """K=1：anchor-as-first 只剩一行（它就是预测行）；fill-in 只剩 mask 那一行。"""
    case = dspark_case(anchor=anchor, k=1)
    upstream = run_upstream_v2_kernel(case, sample_from_anchor=anchor)
    ours = run_our_v2_kernel(case, sample_from_anchor=anchor)
    for name, want in upstream.items():
        assert torch.equal(want, ours[name]), name

    num_query = 1 if anchor else 2
    assert upstream["query_start_loc"][:3].tolist() == [0, num_query, 2 * num_query]
    # 每请求恰好 1 个采样行，采样位置都是 10（= last_valid_pos + 1 + 1）
    assert upstream["sample_pos"][:2].tolist() == [10, 23]
    assert upstream["sample_indices"][:2].tolist() == ([0, 1] if anchor else [1, 3])
    # seq_lens = last_valid_pos + 1 + num_query_per_req（请求 0：8+1+1 或 8+1+2）
    assert upstream["seq_lens"][:2].tolist() == ([10, 23] if anchor else [11, 24])


# ---------------- B. 提议者从配置派生的布局与预算 ----------------

@requires_cuda
@pytest.mark.parametrize("anchor", [True, False])
def test_speculator_derives_layout_from_config(anchor):
    speculator, _ = make_speculator(k=3, anchor=anchor)
    assert speculator.sample_from_anchor is anchor
    assert speculator.num_query_per_req == (3 if anchor else 4)
    # `_anchor_idx` 是"每请求 query 块第 0 行"的常驻下标：步长 = 每请求的 query 行数
    stride = 3 if anchor else 4
    # `max_num_seqs=2`（夹具默认）→ 这张表只有 2 项，步长就是每请求的 query 行数
    assert speculator._anchor_idx[:2].tolist() == [0, stride]
    speculator.init_cudagraph_manager(CUDAGraphMode.NONE)
    assert speculator.query_cudagraph_manager.decode_query_len == speculator.num_query_per_req


def test_slot_budget_matches_the_upstream_table():
    """`max_num_new_slots_for_drafting`：DSpark = K−1（DFlash 是 K）。"""
    spec = make_spec_config(k=3)
    assert spec.parallel_drafting is True, "上游对 dflash/dspark 都强制打开并行提议"
    assert spec.max_num_new_slots_for_drafting == 2
    assert spec.use_eagle() is True


@requires_cuda
def test_scheduler_budgets_for_dspark():
    """调度侧两份预算：输入工作区 K−1 行、KV lookahead K 个位置（上游同一组口径）。"""
    engine, core, runner = make_engine(k=3)
    try:
        assert core.scheduler.draft_slots == 2
        assert core.scheduler.num_lookahead_tokens == 3
        assert runner.speculator.num_query_per_req == 3
    finally:
        engine.shutdown()


def test_both_layouts_are_accepted_by_config():
    """两种布局都能过配置期校验（上游默认 True，speculators 格式写 False）。"""
    for anchor in (True, False):
        spec = make_spec_config(k=3, anchor=anchor)
        assert spec.draft_model_config.hf_config["sample_from_anchor"] is anchor
