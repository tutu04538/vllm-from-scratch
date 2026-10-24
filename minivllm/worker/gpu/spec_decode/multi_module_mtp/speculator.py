"""多模块 MTP 的输入准备与跨轮缓存（需求 080A）。

对应冻结上游 `v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` 中
**输入布局**这一层，逐函数移植：

    prepare_input_buffers / _prepare_input_buffers_kernel
    prepare_input_hidden_states_and_embeddings / _prepare_input_hidden_states_and_embeddings_kernel
    cache_inputs / _cache_inputs_kernel
    pad_trailing_draft_slots / _pad_trailing_draft_slots_kernel
    update_draft_inputs / _shift_input_ids_kernel / _shift_input_embeds_kernel

本关**不实现** `MultiModuleMTPSpeculator` 类，也不接生产调用链：80B 才把这里
的函数接进 `propose()`，并做“目标拒绝后重算后续模块 KV”的修复。这样安排是因为
输入布局一旦错位，后面的模型计算与 KV 修复都会把错误带到更深处，而且不会报错。

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

import torch

from .....triton_utils import tl, triton
from ...block_table import PAD_SLOT_ID
from ...input_batch import InputBatch, InputBuffers


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

    “丢最左一行、右边追加新候选”只移动行，不改变绝对 position；positions
    的推进由调用方按模型语义另行处理（80B 接线）。
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
