"""73 关：`next_prefill_tokens`——**分块 prefill 请求的"下一枚 token"**（写它 / 读它各一条用例）。

这个字段回答的问题是：**一条还在分块 prefill 的请求，本轮一枚 token 都没产出，草稿模型的
第一遍该拿什么当输入？**

- 本轮算了 prompt 的 [4, 8) 这一段，采样侧 `num_sampled = 0`（`seq_len < prefill_len`，
  这个请求这轮不产出 token）；
- 草稿第一遍的输入块 = target 的输入块**左移一行** + 最后一格补"下一枚 token"；
- 有采样 token 时补 `last_sampled_tokens`；**没有**时补的就是 `next_prefill_tokens`
  （`autoregressive/speculator.py` 里那个 `if num_sampled > 0: ... else: 读 lookahead` 分支）。

它由 target 的输入组装内核顺手写出：`input_batch.py::_prepare_prefill_inputs_kernel` 把本轮
chunk 从 slot 的 `all_token_ids` 抄进 `input_ids` 之后，再取紧随其后的 `num_lookahead` 枚
（`pos >= prefill_len` 越界 → 填 **0**；用 0 而不是 -1 是因为 0 是合法 token id，安全性靠
**调用方的 `num_sampled` 分支**保证：能走到"越界"的那一轮必然有采样 token，于是读的是
`last_sampled_tokens`，那个 0 永远不被使用）。

形状是 `[num_prefill_lookahead, max_num_reqs]`（**lookahead 维在前**，所以"第 i 枚 lookahead token
跨所有 slot"是一段连续内存——正是提议者内核按 `lookahead * stride + slot` 取的布局）。宽度默认 1；
上游只有**多模块 MTP**（80 关）会把它开成 `num_speculative_steps`（每个模块吃一枚）。
"""

from __future__ import annotations

import numpy as np
import torch

from spec73_helpers import DEVICE, requires_cuda  # noqa: E402

from minivllm.worker.gpu.buffer_utils import async_copy_to_gpu  # noqa: E402
from minivllm.worker.gpu.input_batch import (  # noqa: E402
    InputBuffers,
    prepare_prefill_inputs,
)
from minivllm.worker.gpu.spec_decode.autoregressive.speculator import (  # noqa: E402
    _prepare_prefill_inputs_kernel as _draft_prepare_prefill_inputs_kernel,
)

MAX_REQS = 4
MAX_TOKENS = 32
MAX_MODEL_LEN = 32
PROMPT = list(range(11, 21))          # 10 枚 prompt，值 11..20（好认）


def make_all_token_ids(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`all_token_ids [R, L]` + `prefill_len [R]` + `num_computed [R]`（都按 slot）。"""
    all_token_ids = np.zeros((MAX_REQS, MAX_MODEL_LEN), dtype=np.int32)
    prefill_len = np.zeros(MAX_REQS, dtype=np.int32)
    for slot, tokens in enumerate(rows):
        all_token_ids[slot, : len(tokens)] = tokens
        prefill_len[slot] = len(tokens)
    return (async_copy_to_gpu(all_token_ids, device=DEVICE),
            async_copy_to_gpu(prefill_len, device=DEVICE), prefill_len)


@requires_cuda
def test_middle_chunk_stores_the_next_prompt_token():
    """中间块（`num_computed + query_len < prefill_len`）→ lookahead 是**真正的下一枚 prompt token**。

    宽度 2 时才看得出"第 i 枚 lookahead"的排布；宽度 1（本仓库默认）第 0 行就是答案。
    """
    all_token_ids, prefill_len, _ = make_all_token_ids([PROMPT])
    input_ids = torch.zeros(MAX_TOKENS, dtype=torch.int32, device=DEVICE)
    next_prefill_tokens = torch.full((2, MAX_REQS), -7, dtype=torch.int32, device=DEVICE)
    idx_mapping = async_copy_to_gpu(np.array([0], dtype=np.intp), device=DEVICE)

    # 第 1 轮：算 [0, 4) → 本轮不产出 token；下一枚 prompt token 是 all_token_ids[4] = 15
    query_start_loc = async_copy_to_gpu(np.array([0, 4], dtype=np.int32), device=DEVICE)
    prepare_prefill_inputs(input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
                           all_token_ids, prefill_len,
                           torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE))
    torch.cuda.synchronize()
    assert input_ids[:4].tolist() == PROMPT[:4]
    assert next_prefill_tokens[:, 0].tolist() == [PROMPT[4], PROMPT[5]]     # 15, 16

    # 第 2 轮：算 [4, 9) → 只剩 1 枚（PROMPT[9] = 20），第 2 个 lookahead 越界 → 0
    query_start_loc = async_copy_to_gpu(np.array([0, 5], dtype=np.int32), device=DEVICE)
    prepare_prefill_inputs(input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
                           all_token_ids, prefill_len,
                           async_copy_to_gpu(np.array([4], dtype=np.int32), device=DEVICE))
    torch.cuda.synchronize()
    assert input_ids[:5].tolist() == PROMPT[4:9]
    assert next_prefill_tokens[:, 0].tolist() == [PROMPT[9], 0]             # 20, 越界 → 0

    # 第 3 轮：算完 [9, 10)（刚好到 prefill 尾）→ 两个 lookahead 都越界 → 0
    query_start_loc = async_copy_to_gpu(np.array([0, 1], dtype=np.int32), device=DEVICE)
    prepare_prefill_inputs(input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
                           all_token_ids, prefill_len,
                           async_copy_to_gpu(np.array([9], dtype=np.int32), device=DEVICE))
    torch.cuda.synchronize()
    assert input_ids[:1].tolist() == PROMPT[9:]
    assert next_prefill_tokens[:, 0].tolist() == [0, 0]


def run_draft_prep(*, target_input_ids: list[int], positions: list[int], seq_len: int,
                   num_sampled: int, num_rejected: int, last_sampled: int,
                   lookahead: int):
    """调提议者的输入组装内核，返回 `(draft_input_ids, last_token_indices, current_step)`。"""
    num_reqs = 1
    query_start_loc = async_copy_to_gpu(
        np.array([0, len(target_input_ids)], dtype=np.int32), device=DEVICE)
    last_token_indices = torch.full((MAX_REQS,), -1, dtype=torch.int32, device=DEVICE)
    current_step = torch.full((1,), -1, dtype=torch.int32, device=DEVICE)
    draft_input_ids = torch.full((MAX_TOKENS,), -1, dtype=torch.int32, device=DEVICE)
    draft_positions = torch.full((MAX_TOKENS,), -1, dtype=torch.int32, device=DEVICE)
    draft_query_start_loc = torch.full((MAX_REQS + 1,), -1, dtype=torch.int32, device=DEVICE)
    draft_seq_lens = torch.full((MAX_REQS,), -1, dtype=torch.int32, device=DEVICE)
    _draft_prepare_prefill_inputs_kernel[(num_reqs,)](
        last_token_indices,
        current_step,
        draft_input_ids,
        draft_positions,
        draft_query_start_loc,
        draft_seq_lens,
        async_copy_to_gpu(np.array(target_input_ids, dtype=np.int32), device=DEVICE),
        async_copy_to_gpu(np.array(positions, dtype=np.int32), device=DEVICE),
        async_copy_to_gpu(np.array([0], dtype=np.intp), device=DEVICE),
        async_copy_to_gpu(np.array([last_sampled], dtype=np.int32), device=DEVICE),
        async_copy_to_gpu(np.array([[lookahead]], dtype=np.int32), device=DEVICE),
        async_copy_to_gpu(np.array([num_sampled], dtype=np.int32), device=DEVICE),
        async_copy_to_gpu(np.array([num_rejected], dtype=np.int32), device=DEVICE),
        query_start_loc,
        async_copy_to_gpu(np.array([seq_len], dtype=np.int32), device=DEVICE),
        MAX_REQS,
        BLOCK_SIZE=1024,
    )
    torch.cuda.synchronize()
    return draft_input_ids, last_token_indices, current_step


@requires_cuda
def test_speculator_takes_lookahead_only_when_no_token_was_sampled():
    """读它：草稿第一遍的最后一格——有采样 token 用 `last_sampled`，没有才用 lookahead。"""
    # 本轮 target 的行是 [101, 102, 103]（对应 prompt 的 [4,7)），还有一枚 lookahead = 999
    target_input_ids, positions = [101, 102, 103], [4, 5, 6]

    # A) num_sampled = 0（还在分块 prefill）→ 最后一格 = lookahead
    draft_ids, last_idx, step = run_draft_prep(
        target_input_ids=target_input_ids, positions=positions, seq_len=7,
        num_sampled=0, num_rejected=0, last_sampled=777, lookahead=999)
    assert draft_ids[:3].tolist() == [102, 103, 999]     # 左移一行 + 补 lookahead
    assert last_idx[0].item() == 2                       # 锚点 = 块的最后一行
    assert step.item() == 0

    # B) num_sampled = 1 → 最后一格 = 本轮采出的 token（lookahead 不参与）
    draft_ids, last_idx, step = run_draft_prep(
        target_input_ids=target_input_ids, positions=positions, seq_len=7,
        num_sampled=1, num_rejected=0, last_sampled=777, lookahead=999)
    assert draft_ids[:3].tolist() == [102, 103, 777]
    assert last_idx[0].item() == 2

    # C) 有被拒行（num_rejected=1）→ 真实块长 = 3 − 1 = 2，锚点跟着前移（63 关的不变量）
    draft_ids, last_idx, step = run_draft_prep(
        target_input_ids=target_input_ids, positions=positions, seq_len=7,
        num_sampled=1, num_rejected=1, last_sampled=777, lookahead=999)
    assert draft_ids[:2].tolist() == [102, 777]
    assert last_idx[0].item() == 1
