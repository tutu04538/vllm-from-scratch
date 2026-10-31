"""多模块 MTP 的输入布局、跨轮缓存、eager 执行与 CUDA Graph（080A/080B/080C）。

对应冻结上游 `v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py`。本文件分两层：

    输入层（80A）
        prepare_input_buffers / _prepare_input_buffers_kernel
        prepare_input_hidden_states_and_embeddings / 对应 kernel
        cache_inputs / _cache_inputs_kernel
        pad_trailing_draft_slots / _pad_trailing_draft_slots_kernel
        update_draft_inputs / _shift_input_ids_kernel / _shift_input_embeds_kernel

    执行层（80B eager + 80C Graph）
        MultiModuleMTPSpeculator：显式 propose → _prepare_inputs →
        slots/pad/metadata → _generate_drafts（或整段 replay）→ cache_inputs →
        每个 module forward / sample / update_draft_inputs

80B 做**单卡、纯文本、eager**；80C 把整段 K 步循环捕获成一张 FULL 图，混合批仍
走 eager。图模式按冻结 `init_cudagraph_manager()` 收敛：PIECEWISE 不支持，纯
PIECEWISE -> NONE，FULL_AND_PIECEWISE -> FULL_DECODE_ONLY。没有 Gemma4、Step3p5、
Q-only 共享 KV 或稀疏索引。

输入布局一旦错位，后面的模型计算与 KV 修复都会把错误带到更深处，而且不会报错；
所以 80A 的逐值差分、80B 的“逐模块 KV 对照”与 80C 的“replay 后常驻状态对照”
是三套独立证据。

三个索引必须分开（本文件所有函数都按这条写）：

    req_idx                        当前 batch 行（kernel 的 program id）
    idx_mapping[req_idx]           持久 request slot（跨轮不变的 cache 行）
    last_token_indices[req_idx]    当前 packed 输入里真正要采样的行

跨轮缓存按 **持久 slot** 索引，列则保留 **未截断的 window_start** 偏移：
`window_start = last_token_index - (K - 1) + 1` 可以是负数，读的时候只把下界夹到
`query_start`，写 cache 的列仍用 `i - window_start`。因此短窗口会在左侧留下空列，
这不是 bug；把数据向 0 列挤会改变窗口内相对位置，下一轮按列读就会错位。

`K = 1` 时缓存宽度就是 0；**不要**用 `max(1, K - 1)` 把它改成 1——那会改变协议，
而不是修一个空窗口。测试 `tests/step80/test_multi_module_inputs_alignment.py` 会
把 `K=1/2/4`、`query` 长度 `< = > K-1`、`num_rejected=0..K` 逐值钉住。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .....config import CUDAGraphMode, VllmConfig
from .....forward_context import BatchDescriptor, set_forward_context
from .....triton_utils import tl, triton
from ....utils import AttentionGroup, get_uniform_decode_token_count
from ...block_table import PAD_SLOT_ID
from ...cudagraph_utils import BatchExecutionDescriptor
from ...input_batch import InputBatch, InputBuffers
from ..eagle.utils import load_eagle_model
from ..speculator import DraftModelSpeculator
from ..autoregressive.cudagraph_utils import SpeculatorCudaGraphManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 第一遍输入：target 的行 → draft 的 packed 行
# ---------------------------------------------------------------------------


@triton.jit
def _prepare_input_buffers_kernel(
    last_token_indices_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    cached_draft_input_ids_ptr,
    cached_draft_input_ids_stride0,
    draft_input_id_overrides_ptr,
    draft_input_id_overrides_stride0,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    next_prefill_tokens_stride0,
    num_sampled_ptr,
    num_rejected_ptr,
    target_seq_lens_ptr,
    query_start_loc_ptr,
    max_num_reqs,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(target_seq_lens_ptr + req_idx)

    # 目标拒绝的草稿位不能作为 draft 输入；其中最后 num_rejected-1 个位置
    # 在上一轮已经被后续 MTP 模块写过 KV，是“过期槽位”，需要在本轮重算。
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    num_reprefill_tokens = max(0, num_rejected - 1)

    num_input_tokens = query_len - num_rejected
    # 重算行被塞进 query 窗口，不额外加宽；因此有效 KV 长度要减去重算行数。
    seq_len -= num_reprefill_tokens
    tl.store(draft_seq_lens_ptr + req_idx, seq_len)

    # 下一枚输入 token：正常 decode 用 target 刚采出的 token；chunked prefill
    # 还没有采样结果，用已知的下一枚 prompt token。
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # chunked prefill 的后续 draft step 也要吃已知 future prompt token，而不是
    # 采样出的草稿。这里按 **持久 slot** 取，不能用 next_prefill_tokens[:, :B]。
    for i in range(1, num_speculative_steps):
        future_token = tl.load(
            next_prefill_tokens_ptr + i * next_prefill_tokens_stride0 + req_state_idx
        )
        override = tl.where(num_sampled > 0, -1, future_token).to(tl.int64)
        tl.store(
            draft_input_id_overrides_ptr
            + req_idx * draft_input_id_overrides_stride0
            + i
            - 1,
            override,
        )

    # target 的第 0 行是 anchor，不进 draft 输入；把第 1..num_input-1 行
    # 复制到 draft 的 [query_start+num_reprefill, ...)，右侧补 next_token。
    for i in range(1, num_input_tokens, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < num_input_tokens
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(
            draft_input_ids_ptr + query_start + num_reprefill_tokens - 1 + block,
            input_ids,
            mask=mask,
        )
    last_token_index = query_start + num_reprefill_tokens + num_input_tokens - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

    # positions 与 input_ids 不同：target 的第 0 行 position 要保留（它对应
    # draft 窗口的第一个真实位置），所以从 i=0 开始复制。
    for i in range(0, num_input_tokens, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < num_input_tokens
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(
            draft_positions_ptr + query_start + num_reprefill_tokens + block,
            target_pos,
            mask=mask,
        )

    # 重算 gap：从跨轮缓存里读上一轮窗口尾部，位置紧挨在 target 第一行之前。
    first_position = tl.load(target_positions_ptr + query_start)
    for i in range(num_reprefill_tokens):
        cache_read_slot = num_speculative_steps - 1 - num_reprefill_tokens + i
        cached_token_id = tl.load(
            cached_draft_input_ids_ptr
            + req_state_idx * cached_draft_input_ids_stride0
            + cache_read_slot
        )
        tl.store(draft_input_ids_ptr + query_start + i, cached_token_id)
        tl.store(
            draft_positions_ptr + query_start + i,
            first_position - num_reprefill_tokens + i,
        )

    if req_idx == (num_reqs - 1):
        # 为 CUDA Graph 补齐：超出真实请求数的 query_start_loc / seq_lens /
        # last_token_indices 必须写成一致的空值，不能留上一轮的残值。
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, query_end, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(draft_seq_lens_ptr + block, 0, mask=mask)
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


def prepare_input_buffers(
    num_reqs: int,
    input_batch: InputBatch,
    # [max_num_reqs, num_speculative_steps - 1]
    cached_draft_input_ids: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [num_prefill_lookahead, max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs, num_speculative_steps - 1]
    draft_input_id_overrides: torch.Tensor,
    input_buffers: InputBuffers,
    # [max_num_reqs]
    last_token_indices: torch.Tensor,
    max_num_reqs: int,
    num_speculative_steps: int,
) -> None:
    """把 target 本轮输入整理成第一遍 draft 输入（原地写缓冲）。

    前置：`input_buffers.query_start_loc[:num_reqs+1]` 已经等于本批的
    `input_batch.query_start_loc[:num_reqs+1]`（上游在 `propose()` 里先拷贝）。

    本函数只做布局，不跑模型、不写 KV。尾部 padding 行由
    `pad_trailing_draft_slots()` 负责把它们的槽位打成 `PAD_SLOT_ID`。
    """
    if num_reqs == 0:
        return
    assert next_prefill_tokens.shape[0] >= num_speculative_steps, (
        "next_prefill_tokens 至少要有 num_speculative_steps 行："
        f"got {next_prefill_tokens.shape[0]}, K={num_speculative_steps}"
    )
    _prepare_input_buffers_kernel[(num_reqs,)](
        last_token_indices,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.seq_lens,
        input_batch.input_ids,
        input_batch.positions,
        cached_draft_input_ids,
        cached_draft_input_ids.stride(0),
        draft_input_id_overrides,
        draft_input_id_overrides.stride(0),
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        next_prefill_tokens.stride(0),
        num_sampled,
        num_rejected,
        input_batch.seq_lens,
        input_buffers.query_start_loc,
        max_num_reqs,
        num_speculative_steps,
        BLOCK_SIZE=1024,
    )


# ---------------------------------------------------------------------------
# 2. hidden state / embedding：与第一遍输入同一套 gap
# ---------------------------------------------------------------------------


@triton.jit
def _prepare_input_hidden_states_and_embeddings_kernel(
    draft_input_hidden_states_ptr,
    draft_input_hidden_states_stride0,
    target_hidden_states_ptr,
    target_hidden_states_stride0,
    cached_target_hidden_states_ptr,
    cached_target_hidden_states_stride0,
    cached_target_hidden_states_stride1,
    input_embeds_ptr,
    input_embeds_stride0,
    cached_draft_input_embeds_ptr,
    cached_draft_input_embeds_stride0,
    cached_draft_input_embeds_stride1,
    idx_mapping_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    num_speculative_steps,
    hidden_size,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    USE_INPUT_EMBEDS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    query_block_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    dim_block = dim_block_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    dim_mask = dim_block < hidden_size

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    num_reprefill_hidden_states = max(0, num_rejected - 1)
    num_input_hidden_states = query_len - num_rejected

    # target hidden 与 input token 一样跳过 anchor 行，右侧让出 gap；每个
    # program 复制一块 (BLOCK_SIZE_Q, BLOCK_SIZE_H)。
    query_block = query_block_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    query_mask = (query_block < num_input_hidden_states)[:, None] & dim_mask[None, :]
    hidden_state = tl.load(
        target_hidden_states_ptr
        + (query_start + query_block)[:, None] * target_hidden_states_stride0
        + dim_block[None, :],
        mask=query_mask,
    )
    tl.store(
        draft_input_hidden_states_ptr
        + (query_start + num_reprefill_hidden_states + query_block)[:, None]
        * draft_input_hidden_states_stride0
        + dim_block[None, :],
        hidden_state,
        mask=query_mask,
    )

    if query_block_idx == 0:
        # gap 最多 K-1 行，在第一个 query block 里串行补；hidden 与 embedding
        # 必须来自同一缓存列，不能只补 hidden 后让 embedding 走另一条路径。
        for i in range(num_reprefill_hidden_states):
            cache_read_slot = (
                num_speculative_steps - 1 - num_reprefill_hidden_states + i
            )
            cached_hidden_state = tl.load(
                cached_target_hidden_states_ptr
                + req_state_idx * cached_target_hidden_states_stride0
                + cache_read_slot * cached_target_hidden_states_stride1
                + dim_block,
                mask=dim_mask,
            )
            tl.store(
                draft_input_hidden_states_ptr
                + (query_start + i) * draft_input_hidden_states_stride0
                + dim_block,
                cached_hidden_state,
                mask=dim_mask,
            )
            if USE_INPUT_EMBEDS:
                cached_embed = tl.load(
                    cached_draft_input_embeds_ptr
                    + req_state_idx * cached_draft_input_embeds_stride0
                    + cache_read_slot * cached_draft_input_embeds_stride1
                    + dim_block,
                    mask=dim_mask,
                )
                tl.store(
                    input_embeds_ptr
                    + (query_start + i) * input_embeds_stride0
                    + dim_block,
                    cached_embed,
                    mask=dim_mask,
                )


def prepare_input_hidden_states_and_embeddings(
    num_reqs: int,
    # Upper bound on the draft query length of any request in the batch.
    max_query_len: int,
    # [num_tokens, hidden_size]
    hidden_states: torch.Tensor,
    # [num_tokens, hidden_size]
    target_hidden_states: torch.Tensor,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_target_hidden_states: torch.Tensor,
    # [num_tokens, hidden_size]
    input_embeds: torch.Tensor | None,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_draft_input_embeds: torch.Tensor | None,
    input_batch: InputBatch,
    input_buffers: InputBuffers,
    # [num_reqs]
    num_rejected: torch.Tensor,
    num_speculative_steps: int,
) -> None:
    """复制 target hidden 到 draft 窗口，并用同一缓存列补 gap。

    这一层只搬数据；哪些行该被 mask 由 `pad_trailing_draft_slots()` 和下游
    attention 的 slot mapping 决定。`input_embeds` 与 `cached_draft_input_embeds`
    同时为 `None` 时走“无多模态 embedding”分支。
    """
    if num_reqs == 0 or max_query_len <= 0:
        return
    use_input_embeds = input_embeds is not None
    if use_input_embeds:
        assert cached_draft_input_embeds is not None, (
            "启用 input_embeds 时 cached_draft_input_embeds 不能为空："
            "gap 必须从同源缓存补 embedding"
        )
    hidden_size = target_hidden_states.shape[-1]
    query_block_size = 16
    hidden_block_size = 256
    grid = (
        num_reqs,
        triton.cdiv(max_query_len, query_block_size),
        triton.cdiv(hidden_size, hidden_block_size),
    )
    _prepare_input_hidden_states_and_embeddings_kernel[grid](
        hidden_states,
        hidden_states.stride(0),
        target_hidden_states,
        target_hidden_states.stride(0),
        cached_target_hidden_states,
        cached_target_hidden_states.stride(0),
        cached_target_hidden_states.stride(1),
        input_embeds,
        input_embeds.stride(0) if input_embeds is not None else 0,
        cached_draft_input_embeds,
        cached_draft_input_embeds.stride(0)
        if cached_draft_input_embeds is not None
        else 0,
        cached_draft_input_embeds.stride(1)
        if cached_draft_input_embeds is not None
        else 0,
        input_batch.idx_mapping,
        num_rejected,
        input_buffers.query_start_loc,
        num_speculative_steps,
        hidden_size,
        BLOCK_SIZE_Q=query_block_size,
        BLOCK_SIZE_H=hidden_block_size,
        USE_INPUT_EMBEDS=use_input_embeds,
    )


# ---------------------------------------------------------------------------
# 3. 跨轮缓存：按持久 slot 存最后 K-1 列
# ---------------------------------------------------------------------------


@triton.jit
def _cache_inputs_kernel(
    draft_input_ids_ptr,
    draft_input_embeds_ptr,
    draft_input_embeds_stride0,
    draft_input_hidden_states_ptr,
    draft_input_hidden_states_stride0,
    cached_draft_input_ids_ptr,
    cached_draft_input_ids_stride0,
    cached_draft_input_embeds_ptr,
    cached_draft_input_embeds_stride0,
    cached_draft_input_embeds_stride1,
    cached_target_hidden_states_ptr,
    cached_target_hidden_states_stride0,
    cached_target_hidden_states_stride1,
    idx_mapping_ptr,
    last_token_indices_ptr,
    query_start_loc_ptr,
    num_speculative_steps,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    USE_INPUT_EMBEDS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < hidden_size

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    if req_state_idx < 0:
        # CUDA Graph 的补齐请求：不能写到 0 号 slot。
        return

    query_start = tl.load(query_start_loc_ptr + req_idx)
    last_token_index = tl.load(last_token_indices_ptr + req_idx)

    # 只快照最后 K-1 列；列号用**未截断**的 window_start，因此短窗口
    # 会空出左侧列，而不是把 token 挤到 0 列。
    cache_window_size = num_speculative_steps - 1
    window_start = last_token_index - cache_window_size + 1
    for i in range(max(window_start, query_start), last_token_index + 1):
        cache_write_slot = i - window_start
        if block_idx == 0:
            input_id = tl.load(draft_input_ids_ptr + i)
            tl.store(
                cached_draft_input_ids_ptr
                + req_state_idx * cached_draft_input_ids_stride0
                + cache_write_slot,
                input_id,
            )
        if USE_INPUT_EMBEDS:
            input_embeds = tl.load(
                draft_input_embeds_ptr + i * draft_input_embeds_stride0 + block,
                mask=mask,
            )
            tl.store(
                cached_draft_input_embeds_ptr
                + req_state_idx * cached_draft_input_embeds_stride0
                + cache_write_slot * cached_draft_input_embeds_stride1
                + block,
                input_embeds,
                mask=mask,
            )
        hidden_state = tl.load(
            draft_input_hidden_states_ptr
            + i * draft_input_hidden_states_stride0
            + block,
            mask=mask,
        )
        tl.store(
            cached_target_hidden_states_ptr
            + req_state_idx * cached_target_hidden_states_stride0
            + cache_write_slot * cached_target_hidden_states_stride1
            + block,
            hidden_state,
            mask=mask,
        )


def cache_inputs(
    input_buffers: InputBuffers,
    # [num_tokens, hidden_size]
    draft_input_embeds: torch.Tensor | None,
    # [num_tokens, hidden_size]
    draft_input_hidden_states: torch.Tensor,
    # [max_num_reqs, num_speculative_steps - 1]
    cached_draft_input_ids: torch.Tensor,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_draft_input_embeds: torch.Tensor | None,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_target_hidden_states: torch.Tensor,
    # [num_reqs]
    last_token_indices: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    num_reqs: int,
    num_speculative_steps: int,
    use_input_embeds: bool,
) -> None:
    """快照每条请求窗口尾部的 token/hidden/embedding，供下一轮补算。

    `last_token_indices` 是**本轮**的采样行；缓存行是 `idx_mapping[req_idx]`
    指向的持久 slot，不是 `req_idx`。`K=1` 时列宽为 0，循环不执行。
    """
    if num_reqs == 0:
        return
    if use_input_embeds:
        assert draft_input_embeds is not None and cached_draft_input_embeds is not None, (
            "use_input_embeds=True 时 draft_input_embeds / "
            "cached_draft_input_embeds 都必须提供"
        )
    hidden_size = draft_input_hidden_states.shape[-1]
    hidden_block_size = 1024
    _cache_inputs_kernel[(num_reqs, triton.cdiv(hidden_size, hidden_block_size))](
        input_buffers.input_ids,
        draft_input_embeds,
        draft_input_embeds.stride(0) if draft_input_embeds is not None else 0,
        draft_input_hidden_states,
        draft_input_hidden_states.stride(0),
        cached_draft_input_ids,
        cached_draft_input_ids.stride(0),
        cached_draft_input_embeds,
        cached_draft_input_embeds.stride(0)
        if cached_draft_input_embeds is not None
        else 0,
        cached_draft_input_embeds.stride(1)
        if cached_draft_input_embeds is not None
        else 0,
        cached_target_hidden_states,
        cached_target_hidden_states.stride(0),
        cached_target_hidden_states.stride(1),
        idx_mapping,
        last_token_indices,
        input_buffers.query_start_loc,
        num_speculative_steps,
        hidden_size,
        BLOCK_SIZE=hidden_block_size,
        USE_INPUT_EMBEDS=use_input_embeds,
    )


# ---------------------------------------------------------------------------
# 4. padding：尾部无效行的槽位必须真的打成 PAD
# ---------------------------------------------------------------------------


@triton.jit
def _pad_trailing_draft_slots_kernel(
    slot_mappings_ptr,
    slot_mappings_stride0,
    query_start_loc_ptr,
    last_token_indices_ptr,
    PAD_ID,
    BLOCK_SIZE: tl.constexpr,
):
    group_idx = tl.program_id(0)
    req_idx = tl.program_id(1)
    # 从 last_token_index+1 到 query_end 的槽位是按“陈旧 token 位置”算出来的；
    # 不打成 PAD 的话 attention 会把这些行写进合法块。
    start = tl.load(last_token_indices_ptr + req_idx) + 1
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    base = slot_mappings_ptr + group_idx * slot_mappings_stride0
    for i in range(start, end, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        tl.store(base + offs, PAD_ID, mask=mask)


def pad_trailing_draft_slots(
    # [num_groups, num_tokens_padded]
    slot_mappings: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
    # [num_reqs]
    last_token_indices: torch.Tensor,
    num_reqs: int,
) -> None:
    """把每条请求 `last_token_index` 之后的槽位原地打成 `PAD_SLOT_ID`。

    这不是可选的清理：尾部行在 packed 缓冲里物理存在，如果还带着一个合法
    槽位号，attention 会照常读/写 KV。调用方必须传**本轮真正的**
    `last_token_indices`；传 `query_end - 1` 会让循环为空，看起来“通过”，
    但尾部并没有被屏蔽。
    """
    if num_reqs == 0:
        return
    num_groups = slot_mappings.shape[0]
    _pad_trailing_draft_slots_kernel[(num_groups, num_reqs)](
        slot_mappings,
        slot_mappings.stride(0),
        query_start_loc,
        last_token_indices,
        PAD_SLOT_ID,
        BLOCK_SIZE=256,
    )


# ---------------------------------------------------------------------------
# 5. 窗口移动：丢掉最左一行，右边追加新候选
# ---------------------------------------------------------------------------


@triton.jit
def _shift_input_ids_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    last_token_indices_ptr,
    draft_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    if req_state_idx < 0:
        return

    query_start = tl.load(query_start_loc_ptr + req_idx)
    # 用 post-rejection 的 last index：移动与插入必须和采样位置对齐。
    last_token_index = tl.load(last_token_indices_ptr + req_idx)
    query_len = last_token_index - query_start + 1

    # 把窗口内容整体向“更早的行”移动一格（丢掉最左一行），再在右端
    # last_token_index 处写入新候选。
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(input_ids_ptr + query_start + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block - 1, input_ids, mask=mask)
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + last_token_index, draft_token)


@triton.jit
def _shift_input_embeds_kernel(
    input_embeds_ptr,
    input_embeds_stride0,
    draft_embeds_ptr,
    draft_embeds_stride0,
    idx_mapping_ptr,
    query_start_loc_ptr,
    last_token_indices_ptr,
    hidden_size,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    if req_state_idx < 0:
        return

    block_idx = tl.program_id(1)
    dim_block = block_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    dim_mask = dim_block < hidden_size

    query_start = tl.load(query_start_loc_ptr + req_idx)
    last_token_index = tl.load(last_token_indices_ptr + req_idx)
    query_len = last_token_index - query_start + 1

    # 与 token 同一方向：丢最左一行，右端写新候选的 embedding。
    for i in range(1, query_len, BLOCK_SIZE_Q):
        query_block = i + tl.arange(0, BLOCK_SIZE_Q)
        query_mask = query_block < query_len
        mask = query_mask[:, None] & dim_mask[None, :]
        input_embed = tl.load(
            input_embeds_ptr
            + (query_start + query_block)[:, None] * input_embeds_stride0
            + dim_block[None, :],
            mask=mask,
        )
        tl.store(
            input_embeds_ptr
            + (query_start + query_block - 1)[:, None] * input_embeds_stride0
            + dim_block[None, :],
            input_embed,
            mask=mask,
        )
    draft_embed = tl.load(
        draft_embeds_ptr + req_idx * draft_embeds_stride0 + dim_block,
        mask=dim_mask,
    )
    tl.store(
        input_embeds_ptr + last_token_index * input_embeds_stride0 + dim_block,
        draft_embed,
        mask=dim_mask,
    )


def update_draft_inputs(
    draft_tokens: torch.Tensor,
    draft_embeds: torch.Tensor | None,
    input_buffers: InputBuffers,
    input_embeds: torch.Tensor | None,
    last_token_indices: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_reqs: int,
) -> None:
    """把本步采样出的草稿 token/embedding 接到窗口右端。

    “丢最左一行、右边追加新候选”只移动行，不改变绝对 position。多模块 MTP
    的模块循环保持 positions 不变；采样用的位置由调用方单独推进。
    """
    if num_reqs == 0:
        return
    _shift_input_ids_kernel[(num_reqs,)](
        input_buffers.input_ids,
        idx_mapping,
        input_buffers.query_start_loc,
        last_token_indices,
        draft_tokens,
        BLOCK_SIZE=1024,
    )
    if input_embeds is not None:
        assert draft_embeds is not None, (
            "input_embeds 不为空时 draft_embeds 不能为空"
        )
        hidden_size = input_embeds.shape[-1]
        hidden_block_size = 256
        _shift_input_embeds_kernel[
            (num_reqs, triton.cdiv(hidden_size, hidden_block_size))
        ](
            input_embeds,
            input_embeds.stride(0),
            draft_embeds,
            draft_embeds.stride(0),
            idx_mapping,
            input_buffers.query_start_loc,
            last_token_indices,
            hidden_size,
            BLOCK_SIZE_Q=16,
            BLOCK_SIZE_H=hidden_block_size,
        )


# ---------------------------------------------------------------------------
# 6. 执行层（80B eager + 80C CUDA Graph）
# ---------------------------------------------------------------------------


class MultiModuleMTPSpeculator(DraftModelSpeculator):
    """多模块 MTP 的 V2 提议者（对应上游 `MultiModuleMTPSpeculator`）。

    本类实现 **单卡、纯文本** 的冻结调用链，并且和上游一样把**整段 K 步循环**
    `_generate_drafts` 录成图；混合批仍走 eager。与上游的差异都来自本仓库缺少的
    组件（ModelState / kv_cache_config / DP / 多模态），逐条记在
    `docs/step80_alignment.md`：

    - 不接收 `num_tokens_across_dp` / `mm_inputs` / `skip_attn_for_dummy_run`；
    - 一个 `SpeculatorCudaGraphManager` 管理整段循环，不套用普通自回归的
      prefill/decode 两张图；
    - 图模式按冻结 `init_cudagraph_manager()` 收敛：带 PIECEWISE 时降级，
      纯 PIECEWISE → NONE，FULL_AND_PIECEWISE → FULL_DECODE_ONLY；
    - `inputs_embeds` 恒为 None（本关纯文本），只走 80A 的无 embedding 分支。
    """

    #: 多模块 MTP 没有 adaptive verification；runner 用它决定是否建 79 关 manager。
    enable_adaptive_verification = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

        # 本关纯文本：不走 80A 的 embedding 复制分支，但保留字段名，接线时不用改调用点。
        self.inputs_embeds: torch.Tensor | None = None
        self.cached_draft_input_embeds: torch.Tensor | None = None

        # chunked prefill 的 future prompt token override；非 prefill 请求填 -1。
        self.draft_input_id_overrides = torch.full(
            (self.max_num_reqs, self.num_speculative_steps - 1),
            -1,
            dtype=torch.int64,
            device=device,
        )

        # 上一轮窗口尾部的同源快照；80A 的 cache_inputs 写，prepare_input_buffers 读。
        self.cached_draft_input_ids = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps - 1,
            dtype=torch.int64,
            device=device,
        )
        self.cached_target_hidden_states = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps - 1,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )

        self.attn_groups: list[list[AttentionGroup]] = []
        self.block_tables = None
        self.target_input_buffers: InputBuffers | None = None
        self.target_build_attn_metadata = None

        # 80C 的图管理器；一个 manager 管整段 K 步循环。runner 的 capture 统计读
        # `cudagraph_managers`，所以这里保留与自回归路径同形的账本接口。
        self.cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.cudagraph_managers: dict[str, Any] = {}
        self.cudagraph_mode = CUDAGraphMode.NONE
        self.requested_cudagraph_mode = CUDAGraphMode.NONE
        self.use_fused_multi_step_decode = False

    # -------- 加载与注意力设施 --------

    def load_draft_model(self, target_model: torch.nn.Module) -> torch.nn.Module:
        """MTP 权重就在 target checkpoint 里，走与 EAGLE 相同的 draft 加载路径。"""
        return load_eagle_model(target_model, self.vllm_config)

    def set_attn(
        self,
        block_tables,
        target_input_buffers: InputBuffers,
        target_build_attn_metadata,
    ) -> None:
        """接上块表、target 输入缓冲与草稿自己的注意力组。

        与 80A–C 的单组版本不同：Step3p5/多 KV group 的草稿层会按
        `module.kv_cache_group_id` 分组，每个 target group 建一个 `AttentionGroup`。
        这与上游 `init_attn_backend(... active_layer_names=...)` 的职责一致。
        """
        super().set_attn(
            block_tables, target_input_buffers, target_build_attn_metadata
        )
        if not self.draft_attn_layer_names:
            raise ValueError(
                "multi-module MTP 没有草稿注意力层名：runner 必须先 load_model() "
                "再设置 draft_attn_layer_names，最后调 set_attn()"
            )
        from .....attention.layer import Attention

        kv_config = getattr(self, "kv_cache_config", None)
        by_group: dict[int, list[str]] = {}
        self.draft_layer_modules: dict[str, Attention] = {}
        for _, module in self.model.named_modules():
            if not isinstance(module, Attention):
                continue
            if module.layer_name not in self.draft_attn_layer_names:
                continue
            self.draft_layer_modules[module.layer_name] = module
            group_id = (0 if module.kv_cache_group_id is None
                        else int(module.kv_cache_group_id))
            by_group.setdefault(group_id, []).append(module.layer_name)
        if not by_group:
            raise ValueError(
                "草稿模型里没有可建组的 Attention 层：draft_attn_layer_names "
                "与模块层名不一致")
        backend = self.draft_attention_backend
        builder_cls = backend.get_builder_cls()
        groups = []
        for group_id in sorted(by_group):
            spec = None
            if kv_config is not None and group_id < len(kv_config.kv_cache_groups):
                spec = kv_config.kv_cache_groups[group_id].kv_cache_spec
            groups.append(AttentionGroup(
                backend=backend,
                layer_names=by_group[group_id],
                metadata_builders=[
                    builder_cls(self.vllm_config.cache_config.block_size)],
                kv_cache_group_id=group_id,
                kv_cache_spec=spec,
            ))
        self.attn_groups = [groups]

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """建多模块草稿的图管理器（对应冻结 `init_cudagraph_manager()`）。

        冻结源码只对 **PIECEWISE** 做降级，不把 FULL 降成 FULL_DECODE_ONLY：

            PIECEWISE              -> NONE（没有 FULL 那一半）
            FULL_AND_PIECEWISE     -> FULL_DECODE_ONLY（保留 decode 的 FULL）
            FULL / FULL_DECODE_ONLY / NONE -> 原样

        这就是“PIECEWISE 不支持，但不是一个未声明的 eager 假动作”：降级后的
        mode 会写到 `self.cudagraph_mode`，测试与 results 可以查。
        """
        self.requested_cudagraph_mode = cudagraph_mode
        draft_mode = cudagraph_mode
        if draft_mode.has_piecewise_cudagraphs():
            draft_mode = (
                CUDAGraphMode.FULL_DECODE_ONLY
                if draft_mode.has_full_cudagraphs()
                else CUDAGraphMode.NONE
            )
            logger.warning(
                "多模块 MTP 不支持 PIECEWISE CUDA Graph（冻结源码的 TODO）："
                "requested=%s，草稿实际=%s",
                cudagraph_mode.name,
                draft_mode.name,
            )
        self.cudagraph_mode = draft_mode
        self.cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            draft_mode,
            # 整段 K 步循环按“每请求 K+1 行（anchor + K 个 draft 输入）”建图；
            # 不能套普通 decode 的 1 行/请求。
            self.num_speculative_steps + 1,
        )
        # runner 的 capture 统计按名字遍历；与自回归路径的 prefill/decode 账本同形。
        self.cudagraph_managers = {"multi_module": self.cudagraph_manager}

    def capture(self) -> None:
        """捕获整段 `_generate_drafts`（K 个模块步 + 采样 + 输入 shift）。

        捕获前清掉两个跨轮索引的旧值：dummy 批会按它们选采样行/写 cache；上一轮
        留下的 slot 可能越界。上游只清 `last_token_indices`，这里连 `idx_mapping`
        一起清，避免 dummy 请求把 cache 写到非 0 slot（问题一样，只是更早暴露）。
        """
        assert self.cudagraph_manager is not None, "先调 init_cudagraph_manager()"
        self.last_token_indices.zero_()
        self.idx_mapping.zero_()
        self.hidden_states.zero_()
        self.cudagraph_manager.capture(
            self._generate_drafts,
            self._draft_attn_metadata_for_capture,
            self.input_buffers,
            self.block_tables,
            progress_bar_desc="Capturing multi-module MTP CUDA graphs",
        )

    # -------- 主流程 --------

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [num_prefill_lookahead, max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dummy_run: bool = False,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return self.draft_tokens[:0]

        num_tokens = int(input_batch.num_tokens)
        num_tokens_padded = int(input_batch.num_tokens_after_padding)
        max_query_len = int(input_batch.num_scheduled_tokens.max())
        max_seq_len = int(
            input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        )
        # 上游多模块版本不额外 +K：本轮 query 窗口本身已经包含 anchor + 候选。
        self.draft_max_seq_len = min(max_seq_len, self.max_model_len)

        self._copy_request_inputs(
            num_reqs, input_batch.idx_mapping, temperature, seeds
        )
        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(
            input_batch.query_start_loc[: num_reqs + 1]
        )

        self._prepare_inputs(
            last_hidden_states,
            input_batch,
            max_query_len,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
        )

        # 统一 decode 的判定与自回归路径同一条：未补齐的 num_tokens、真实 max query。
        # 多模块整段循环的图键是“每请求 K+1 行”，所以这里给出实际行数，不在外面
        # 用 Python 去猜。
        uniform_token_count = get_uniform_decode_token_count(
            num_reqs, num_tokens, max_query_len, input_batch.has_prefill
        )
        batch_desc = self._dispatch_cudagraph(
            num_reqs,
            num_tokens_padded,
            uniform_token_count,
            need_eager=is_profile,
        )

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # 图在捕获时已经记下了整段循环的 metadata 指针。运行期必须把同一批
            # 常驻缓冲（块表 / slot_mapping / seq_lens / query_start_loc）写到与
            # 捕获桶一致的状态，然后只 replay，不再跑 `_generate_drafts`。
            num_reqs_padded = batch_desc.num_reqs or num_reqs
            num_tokens_graph = batch_desc.num_tokens
            if num_tokens_graph > num_tokens:
                # 图桶比真实批大：补齐行不能读上一轮的 token/position/hidden。
                self.input_buffers.input_ids[num_tokens:num_tokens_graph].zero_()
                self.input_buffers.positions[num_tokens:num_tokens_graph].zero_()
                self.hidden_states[num_tokens:num_tokens_graph].zero_()

            self.block_tables.gather_block_tables(
                self.idx_mapping[:num_reqs], num_reqs_padded
            )
            slot_mappings_tensor = self.block_tables.compute_slot_mappings(
                self.idx_mapping[:num_reqs],
                self.input_buffers.query_start_loc,
                self.input_buffers.positions,
                num_tokens_graph,
            )
            if num_tokens_graph > num_tokens:
                # `compute_slot_mappings` 只按真实请求数启动；图桶多出来的行
                # 必须显式写哨兵，否则 replay 会按上一轮残值写 KV。
                slot_mappings_tensor[:, num_tokens:].fill_(PAD_SLOT_ID)
            pad_trailing_draft_slots(
                slot_mappings_tensor,
                self.input_buffers.query_start_loc,
                self.last_token_indices[:num_reqs],
                num_reqs,
            )
            assert self.cudagraph_manager is not None
            self.cudagraph_manager.run_fullgraph(batch_desc)
            return self.draft_tokens[:num_reqs]

        # eager / 非图回退：槽位按“本轮 packed 窗口的位置”现算；尾部无效行必须打 PAD。
        slot_mappings_tensor = self.block_tables.compute_slot_mappings(
            self.idx_mapping[:num_reqs],
            self.input_buffers.query_start_loc,
            self.input_buffers.positions,
            num_tokens,
        )
        pad_trailing_draft_slots(
            slot_mappings_tensor,
            self.input_buffers.query_start_loc,
            self.last_token_indices[:num_reqs],
            num_reqs,
        )
        draft_attn_metadata = self._build_draft_attn_metadata(
            num_reqs, num_tokens
        )
        slot_mappings_by_layer = {"slot_mapping": slot_mappings_tensor}

        self._generate_drafts(
            num_reqs,
            num_tokens,
            draft_attn_metadata,
            slot_mappings_by_layer,
        )
        return self.draft_tokens[:num_reqs]

    def _dispatch_cudagraph(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        need_eager: bool = False,
    ) -> BatchExecutionDescriptor:
        """图分派（唯一写入 selection 账本的地方；与 74 的薄封装同形）。"""
        manager = self.cudagraph_manager
        if manager is None:
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
            )
        desc = manager.dispatch(
            num_reqs,
            num_tokens,
            uniform_token_count,
            need_eager=need_eager,
        )
        manager.record_selection(desc, num_reqs, num_tokens)
        return desc

    @property
    def cudagraph_selections(self) -> list[dict]:
        """多模块草稿的“真实选中”账本（与自回归路径同形，测试直接读）。"""
        manager = self.cudagraph_manager
        if manager is None:
            return []
        return [dict(entry, graph="multi_module") for entry in manager.selections]

    def _prepare_inputs(
        self,
        target_hidden_states: torch.Tensor,
        input_batch: InputBatch,
        max_query_len: int,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> None:
        """80A 的两个准备函数：第一遍 token/position + 同源的 hidden。"""
        num_reqs = input_batch.num_reqs
        prepare_input_buffers(
            num_reqs,
            input_batch,
            self.cached_draft_input_ids,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.draft_input_id_overrides,
            self.input_buffers,
            self.last_token_indices,
            self.max_num_reqs,
            self.num_speculative_steps,
        )
        prepare_input_hidden_states_and_embeddings(
            num_reqs,
            max_query_len,
            self.hidden_states,
            target_hidden_states,
            self.cached_target_hidden_states,
            None,
            None,
            input_batch,
            self.input_buffers,
            num_rejected,
            self.num_speculative_steps,
        )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_tokens: int,
        *,
        num_reqs_padded: int | None = None,
        num_tokens_padded: int | None = None,
        uniform_query_len: int | None = None,
        max_query_len: int | None = None,
        block_table: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        """为 packed query 建**按 KV group 分组**的 metadata。

        `block_table` / `slot_mapping` 参数只保留给单组调用点/测试做显式覆盖；
        正常路径按 `attn_group.kv_cache_group_id` 从常驻块表和槽位里取对应组。
        """
        if not self.attn_groups:
            return None
        reqs = num_reqs if num_reqs_padded is None else num_reqs_padded
        tokens = num_tokens if num_tokens_padded is None else num_tokens_padded
        if not hasattr(self, "draft_layer_modules"):
            self.draft_layer_modules = {
                module.layer_name: module
                for _, module in self.model.named_modules()
                if hasattr(module, "layer_name")
                and module.layer_name in self.draft_attn_layer_names
            }
        result: dict[str, Any] = {}
        for groups in self.attn_groups:
            for attn_group in groups:
                group_id = (0 if attn_group.kv_cache_group_id is None
                            else int(attn_group.kv_cache_group_id))
                builder = attn_group.get_metadata_builder()
                group_block_table = (
                    block_table if block_table is not None and group_id == 0
                    else self.block_tables.input_block_tables[group_id][:reqs]
                )
                group_slot_mapping = (
                    slot_mapping if slot_mapping is not None and group_id == 0
                    else self.block_tables.slot_mappings[group_id][:tokens]
                )
                spec = attn_group.kv_cache_spec
                sliding = getattr(spec, "sliding_window", None)
                if sliding is None and attn_group.layer_names:
                    module = self.draft_layer_modules.get(attn_group.layer_names[0])
                    if module is not None:
                        sliding = module.per_layer_sliding_window
                metadata = builder.build(
                    query_start_loc=self.input_buffers.query_start_loc[: reqs + 1],
                    seq_lens=self.input_buffers.seq_lens[:reqs],
                    block_table=group_block_table,
                    slot_mapping=group_slot_mapping,
                    num_reqs=reqs,
                    uniform_query_len=uniform_query_len,
                    max_query_len=max_query_len,
                    sliding_window=sliding,
                )
                for name in attn_group.layer_names:
                    result[name] = metadata
        return result

    def _draft_attn_metadata_for_capture(
        self, input_batch: InputBatch
    ) -> dict[str, Any]:
        """捕获期 metadata：dummy 输入已写进常驻缓冲，按每个 group 分别建。"""
        uniform = input_batch.uniform_query_len
        max_query = None if uniform is not None else int(input_batch.max_query_len)
        metadata = self._build_draft_attn_metadata(
            input_batch.num_reqs,
            input_batch.num_tokens,
            uniform_query_len=uniform,
            max_query_len=max_query,
        )
        assert metadata is not None
        return metadata

    def _draft_layer_names(self) -> list[str]:
        if self.attn_groups:
            return [name for groups in self.attn_groups
                    for group in groups for name in group.layer_names]
        return list(self.draft_attn_layer_names)

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        spec_module_idx: int,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """跑一个真实 MTP 模块；`spec_module_idx` 由模型自己的协议解释。

        `cudagraph_runtime_mode` 只在捕获时传 FULL；重放时整个 Python 循环不会执行，
        由 `torch.cuda.CUDAGraph` 按捕获时的 kernels 顺序跑完 K 步。
        """
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            slot_mapping=slot_mappings,
        ):
            ret_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                spec_step_idx=spec_module_idx,
            )
        if isinstance(ret_hidden_states, tuple):
            last_hidden_states, hidden_states = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def _generate_drafts(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """整段 packed query 的模块循环（对应上游同名方法）。

        关键时点（顺序不能改）：

            1. `cache_inputs()` 在**第一个模块之前**快照窗口尾部；
            2. 每个 step 用 `spec_module_idx=step` 跑模型；
            3. 用 `last_token_indices` 选采样行；
            4. 非最后一步：把反馈 hidden 写回，按 override 选下一枚输入 token，
               `update_draft_inputs()` 丢最左一行、右边追加候选。
        """
        last_token_indices = self.last_token_indices[:num_reqs]
        sample_positions = self.input_buffers.positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]

        cache_inputs(
            self.input_buffers,
            None,
            self.hidden_states,
            self.cached_draft_input_ids,
            None,
            self.cached_target_hidden_states,
            last_token_indices,
            idx_mapping,
            num_reqs,
            self.num_speculative_steps,
            use_input_embeds=False,
        )

        for step in range(self.num_speculative_steps):
            self.current_draft_step.fill_(step)

            # eager 路径保持 4 参数调用（80B 的观测/替换点不变）；只有捕获时把
            # FULL 传进 forward context，让注意力走图内固定形状路径。
            if cudagraph_runtime_mode == CUDAGraphMode.NONE:
                last_hidden_states, hidden_states = self._run_model(
                    num_tokens,
                    attn_metadata,
                    slot_mappings,
                    spec_module_idx=step,
                )
            else:
                last_hidden_states, hidden_states = self._run_model(
                    num_tokens,
                    attn_metadata,
                    slot_mappings,
                    spec_module_idx=step,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                )
            sample_hidden_states = last_hidden_states[last_token_indices]
            draft_tokens = self.sample_draft(
                sample_hidden_states,
                sample_positions,
                idx_mapping,
                self.temperature,
                self.seeds,
                self.current_draft_step,
                self.draft_logits,
                spec_step_idx=step,
            )
            self.draft_tokens[:num_reqs, step] = draft_tokens

            if step < self.num_speculative_steps - 1:
                self.hidden_states[:num_tokens] = hidden_states
                overrides = self.draft_input_id_overrides[:num_reqs, step]
                next_input_tokens = torch.where(
                    overrides >= 0, overrides, draft_tokens
                )
                update_draft_inputs(
                    next_input_tokens,
                    None,
                    self.input_buffers,
                    None,
                    last_token_indices,
                    idx_mapping,
                    num_reqs,
                )
                sample_positions += 1
