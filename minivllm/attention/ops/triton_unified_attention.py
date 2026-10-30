"""Triton unified paged attention（086 的生产入口）。

本文件是上游 `v1/attention/ops/triton_unified_attention.py` 的**本地子集**，
只保留本仓库真实需要的路径：

    * 2D：整个 batch 的扁平 query 行，一个 program 负责一个 query block/head；
    * 3D：单 token decode 时把一条长 KV 切成多个 segment，再 `reduce_segments`；
    * paged KV、GQA、causal/非 causal、sliding window；
    * Q-only 层只读 cache，不写。

不做的分支（量化、ALiBi、sinks、softcap、cascade、MM prefix、RSWA、tensor
descriptor）在 backend 的入口检查里明确拒绝，不在这里静默走另一条路径。

布局约定（与 `minivllm` 的 KV cache 一致）：

    q/out       [num_tokens, num_heads, head_size]
    k/v         [num_blocks, block_size, num_kv_heads, head_size]
    block_table [num_reqs, max_blocks_per_req]
    seqused_k   [num_reqs]，值是**本轮之后**的上下文长度
    cu_seqlens_q[num_reqs + 1]，query 行的前缀和

`window_size` 沿用上游的二元组：`(-1, -1)` 表示全注意力；`(w-1, 0)` 表示
causal sliding window，窗口大小为 `w`。kernel 内部把 `SLIDING_WINDOW` 直接
当作“可见 key 数”，条件为 `q_pos - key_pos < w`。
"""

from __future__ import annotations

import torch

from ...triton_utils import tl, triton


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


@triton.jit
def _cdiv(x, y):
    return (x + y - 1) // y


@triton.jit
def _find_seq_idx(query_start_loc_ptr, target_idx, num_seqs,
                  BLOCK_Q: tl.constexpr, use_q_block: tl.constexpr):
    """在 query 前缀和里二分，找到 target 属于哪条请求。"""
    left = tl.full([], 0, tl.int32)
    right = tl.full([], num_seqs, tl.int32)
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_loc_ptr + mid)
        if use_q_block:
            val = val // BLOCK_Q + mid
        if val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _resolve_seq(query_start_loc_ptr, seq_lens_ptr, q_block_global_idx,
                 num_seqs, BLOCK_Q: tl.constexpr):
    seq_idx = _find_seq_idx(
        query_start_loc_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)
    q_start = tl.load(query_start_loc_ptr + seq_idx)
    q_end = tl.load(query_start_loc_ptr + seq_idx + 1)
    q_block_start = q_start // BLOCK_Q + seq_idx
    q_block_local = q_block_global_idx - q_block_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    return seq_idx, q_block_local, q_start, q_end - q_start, seq_len


@triton.jit
def _softmax_step(S, M, L):
    m_new = tl.maximum(M, tl.max(S, axis=1))
    # 整行都被 mask 时（padding / 空请求）避免 max 保持 -inf。
    m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
    P = tl.exp(S - m_new[:, None])
    l_new = L * tl.exp(M - m_new) + tl.sum(P, axis=1)
    alpha = tl.exp(M - m_new)
    return m_new, l_new, P, alpha


@triton.jit
def _tile_loop_bounds(context_len, seq_len, q_len, q_block_local,
                      segm_idx, tiles_per_segment, TILE_SIZE: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_Q: tl.constexpr,
                      GROUP: tl.constexpr, SLIDING_WINDOW: tl.constexpr,
                      IS_3D: tl.constexpr, USE_CAUSAL: tl.constexpr):
    # 一个 query block 能看到的最远 key：最后一个 query 位置 + 1。
    max_prefix = context_len + q_block_local * BLOCK_Q + (
        (BLOCK_M - 1) // GROUP) + 1
    if USE_CAUSAL:
        max_prefix = tl.minimum(max_prefix, seq_len)
    else:
        # 非 causal 要看完整有效范围；越界槽位由 seq_mask 挡掉。
        max_prefix = seq_len

    num_tiles = _cdiv(max_prefix, TILE_SIZE)
    tile_start = 0
    tile_end = num_tiles

    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // GROUP, q_len - 1)
        q_abs = context_len + qpos_lo
        first_key = tl.maximum(0, q_abs - SLIDING_WINDOW + 1)
        if USE_CAUSAL:
            last_key = context_len + qpos_hi
        else:
            last_key = tl.minimum(
                context_len + qpos_hi + SLIDING_WINDOW - 1, seq_len - 1)
        tile_start = first_key // TILE_SIZE
        tile_end = tl.minimum(last_key // TILE_SIZE + 1, num_tiles)

    if IS_3D:
        loop_lo = tl.maximum(segm_idx * tiles_per_segment, tile_start)
        loop_hi = tl.minimum((segm_idx + 1) * tiles_per_segment, tile_end)
    else:
        loop_lo = tile_start
        loop_hi = tile_end
    return loop_lo, loop_hi, max_prefix


@triton.jit
def _kv_seq_mask(query_abs_pos, seq_offset, seq_len,
                 SLIDING_WINDOW: tl.constexpr, USE_CAUSAL: tl.constexpr):
    if USE_CAUSAL:
        seq_mask = seq_offset[None, :] <= query_abs_pos
    else:
        seq_mask = seq_offset[None, :] < seq_len

    if SLIDING_WINDOW > 0:
        dist = query_abs_pos - seq_offset[None, :]
        seq_mask = seq_mask & (dist < SLIDING_WINDOW)
        if not USE_CAUSAL:
            seq_mask = seq_mask & (
                (seq_offset[None, :] - query_abs_pos) < SLIDING_WINDOW)
    return seq_mask


# ---------------------------------------------------------------------------
# 主 kernel：2D 与 3D 共用
# ---------------------------------------------------------------------------


@triton.jit
def kernel_unified_attention(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_CAUSAL: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    IS_3D: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    Q_IS_FP32: tl.constexpr,
    segm_output_ptr=None,
    segm_max_ptr=None,
    segm_expsum_ptr=None,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (seq_idx, q_block_local, q_start, q_len,
     seq_len) = _resolve_seq(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx,
        num_seqs, BLOCK_Q)

    # 网格是按上界算的；超出这条请求真实 q_len 的 block 直接退出。
    if q_block_local * BLOCK_Q >= q_len:
        return

    if IS_3D:
        tiles_per_segment = _cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    dim_mask = offs_d < HEAD_SIZE

    # (query 位置, GQA 组内 head) 的笛卡尔积，展开成 BLOCK_M 行。
    query_pos = q_block_local * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = q_start + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + (
        offs_m % num_queries_per_kv)
    row_mask = (
        (offs_m < BLOCK_Q * num_queries_per_kv)
        & (query_pos < q_len)
        & (query_offset_1 < num_query_heads)
    )

    q_ptrs = (
        query_ptr
        + query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )
    Q = tl.load(
        q_ptrs,
        mask=row_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    context_len = seq_len - q_len
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    loop_lo, loop_hi, max_prefix = _tile_loop_bounds(
        context_len, seq_len, q_len, q_block_local, segm_idx,
        tiles_per_segment, TILE_SIZE, BLOCK_M, BLOCK_Q,
        num_queries_per_kv, SLIDING_WINDOW, IS_3D, USE_CAUSAL)

    block_table_offset = seq_idx * block_table_stride
    query_abs_pos = context_len + query_pos[:, None]

    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_prefix

        physical_block = tl.load(
            block_tables_ptr + block_table_offset
            + seq_offset // BLOCK_SIZE,
            mask=tile_mask,
            other=0,
        ).to(tl.int64)
        slot_in_block = seq_offset % BLOCK_SIZE

        k_ptrs = (
            key_cache_ptr
            + physical_block[None, :] * stride_k_cache_0
            + slot_in_block[None, :] * stride_k_cache_1
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
        )
        v_ptrs = (
            value_cache_ptr
            + physical_block[:, None] * stride_v_cache_0
            + slot_in_block[:, None] * stride_v_cache_1
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
        )
        K = tl.load(
            k_ptrs,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        ).to(Q.dtype)
        V = tl.load(
            v_ptrs,
            mask=tile_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(Q.dtype)

        if Q_IS_FP32:
            S = tl.dot(Q, K, input_precision="ieee") * scale
        else:
            S = tl.dot(Q, K) * scale
        seq_mask = _kv_seq_mask(
            query_abs_pos, seq_offset, seq_len,
            SLIDING_WINDOW, USE_CAUSAL)
        keep = row_mask[:, None] & tile_mask[None, :] & seq_mask
        S = tl.where(keep, S, float("-inf"))

        M, L, P, alpha = _softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        # 被窗口/tail mask 掉的 key 可能落在未写过的槽位；P=0 乘 NaN 仍是
        # NaN，所以先把这些 V 清零，再进 dot。
        v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0
        V = tl.where(v_keep[:, None], V, 0.0)
        if Q_IS_FP32:
            acc += tl.dot(P.to(V.dtype), V, input_precision="ieee")
        else:
            acc += tl.dot(P.to(V.dtype), V)

    if IS_3D:
        # 3D 保存的是未归一化的部分和；M/L 供 reduce_segments 合并。
        segm_out_off = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None].to(tl.int64)
            * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
            + offs_d[None, :]
        )
        tl.store(
            segm_output_ptr + segm_out_off,
            acc,
            mask=row_mask[:, None] & dim_mask[None, :],
        )
        scalars_off = (
            query_offset_0.to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
            + query_offset_1 * NUM_SEGMENTS_PER_SEQ
            + segm_idx
        )
        tl.store(segm_max_ptr + scalars_off, M, mask=row_mask)
        tl.store(segm_expsum_ptr + scalars_off, L, mask=row_mask)
    else:
        out = tl.where(L[:, None] > 0.0, acc / L[:, None], 0.0)
        out_off = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d[None, :]
        )
        tl.store(
            output_ptr + out_off,
            out.to(output_ptr.dtype.element_ty),
            mask=row_mask[:, None] & dim_mask[None, :],
        )


@triton.jit
def reduce_segments(
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    num_seqs,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False)
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    tiles_per_segment = _cdiv(
        seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    tiles_per_segment = tl.maximum(tiles_per_segment, 1)
    act_segments = _cdiv(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < act_segments
    dim_mask = tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE

    segm_off = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(
        segm_max_ptr + segm_off, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)
    overall_max = tl.where(
        overall_max > float("-inf"), overall_max, 0.0)

    segm_expsum = tl.load(
        segm_expsum_ptr + segm_off, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    segm_out_off = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_out = tl.load(
        segm_output_ptr + segm_out_off,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_out = segm_out * tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_out, axis=0)
    acc = tl.where(
        overall_expsum > 0.0, acc_sum / overall_expsum, 0.0)

    out_off = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(
        output_ptr + out_off,
        acc.to(output_ptr.dtype.element_ty),
        mask=dim_mask,
    )


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def make_3d_workspace(device, num_heads: int, head_size: int,
                      seq_threshold_3D: int,
                      num_par_softmax_segments: int):
    """3D 分段的常驻 FP32 工作区（backend 缓存，不每层临时分配）。"""
    head_padded = max(16, _next_power_of_2(head_size))
    return (
        torch.empty(
            (seq_threshold_3D, num_heads, num_par_softmax_segments,
             head_padded),
            dtype=torch.float32,
            device=device,
        ),
        torch.empty(
            (seq_threshold_3D, num_heads, num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        ),
        torch.empty(
            (seq_threshold_3D, num_heads, num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        ),
    )


def can_use_3d(*, max_seqlen_q: int | None,
               num_seqs: int,
               seq_threshold_3D: int | None,
               num_par_softmax_segments: int | None) -> bool:
    """3D 只在“单 token decode + 小 batch”时启用。

    K=2 target 有 K+1 行 query，`max_seqlen_q > 1`，必须留在 2D。这个判断
    放在 Python 侧且只读 Python int，不读 GPU 张量，因此捕获安全。
    """
    return (
        max_seqlen_q == 1
        and num_seqs > 0
        and seq_threshold_3D is not None
        and num_par_softmax_segments is not None
        and num_seqs <= seq_threshold_3D
    )


def use_3d_path(*, max_seqlen_q: int | None,
                num_seqs: int,
                seq_threshold_3D: int | None,
                num_par_softmax_segments: int | None,
                workspace) -> bool:
    return (
        workspace is not None
        and can_use_3d(
            max_seqlen_q=max_seqlen_q,
            num_seqs=num_seqs,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
        )
    )


def unified_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int] = (-1, -1),
    max_seqlen_q: int | None = None,
    seq_threshold_3D: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
) -> torch.Tensor:
    """统一分页 attention；`out` 由调用方提供并直接写入。"""
    if triton is None:
        raise RuntimeError("Triton 不可用，不能使用 TRITON_ATTN 后端")
    if q.ndim != 3 or out.ndim != 3:
        raise ValueError(f"q/out 必须是 3D，收到 {tuple(q.shape)} / {tuple(out.shape)}")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"k/v 必须是 4D，收到 {tuple(k.shape)} / {tuple(v.shape)}")
    if q.shape[2] != k.shape[3] or q.shape[2] != v.shape[3]:
        raise ValueError("q/k/v 的 head_size 不一致")
    if q.shape[1] != out.shape[1] or q.shape[2] != out.shape[2]:
        raise ValueError("q/out 形状不一致")

    num_tokens, num_heads, head_size = q.shape
    num_kv_heads = k.shape[2]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"GQA 要求 query head 数整除 KV head 数，收到 {num_heads}/{num_kv_heads}")
    num_seqs = int(seqused_k.numel())
    if num_tokens == 0 or num_seqs == 0:
        return out

    group = num_heads // num_kv_heads
    block_size = int(k.shape[1])
    if block_size <= 0:
        raise ValueError(f"block_size 必须为正，收到 {block_size}")

    # BLOCK_M 是“query 位置 × GQA 组内 head”的总行数；尽量让 K+1 行共享
    # 一次 KV 扫描。组大于 16 时退化为一个 query 位置。
    block_m = 16 if group <= 16 else _next_power_of_2(group)
    block_q = block_m // group
    if block_q < 1:
        raise ValueError(f"GQA group={group} 无法装入 BLOCK_M={block_m}")
    head_padded = max(16, _next_power_of_2(head_size))

    workspace = None
    if (softmax_segm_output is not None
            and softmax_segm_max is not None
            and softmax_segm_expsum is not None):
        workspace = (softmax_segm_output, softmax_segm_max,
                     softmax_segm_expsum)
    is_3d = use_3d_path(
        max_seqlen_q=max_seqlen_q,
        num_seqs=num_seqs,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        workspace=workspace,
    )

    sliding_window = 0
    if window_size[0] is not None and window_size[0] >= 0:
        sliding_window = int(window_size[0]) + 1
    tile_size = 16 if is_3d else 32

    # 上界：sum ceil(q_len / BLOCK_Q) <= floor(N / BLOCK_Q) + num_seqs。
    total_q_blocks = num_tokens // block_q + num_seqs
    grid = ((total_q_blocks, num_kv_heads, num_par_softmax_segments)
            if is_3d else (total_q_blocks, num_kv_heads))

    segm_out = workspace[0] if is_3d else None
    segm_max = workspace[1] if is_3d else None
    segm_expsum = workspace[2] if is_3d else None

    kernel_unified_attention[grid](
        out,
        q,
        k,
        v,
        block_table,
        seqused_k,
        cu_seqlens_q,
        softmax_scale,
        num_query_heads=num_heads,
        num_queries_per_kv=group,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_padded,
        SLIDING_WINDOW=sliding_window,
        USE_CAUSAL=bool(causal),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        BLOCK_Q=block_q,
        num_seqs=num_seqs,
        BLOCK_M=block_m,
        IS_3D=is_3d,
        NUM_SEGMENTS_PER_SEQ=(num_par_softmax_segments if is_3d else 1),
        Q_IS_FP32=(q.dtype == torch.float32),
        segm_output_ptr=segm_out,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
    )

    if is_3d:
        reduce_segments[(num_tokens, num_heads)](
            out,
            segm_out,
            segm_max,
            segm_expsum,
            seqused_k,
            cu_seqlens_q,
            num_seqs,
            num_query_heads=num_heads,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            TILE_SIZE=tile_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_padded,
            BLOCK_Q=block_q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
        )
    return out
