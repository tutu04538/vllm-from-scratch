"""固定场景的 Triton paged attention。

固定条件（不再支持的场景由调用方拒绝）：

    block_size = 16
    head_dim   = 16 / 32 / 64 / 128（>=16，2 的幂）
    causal
    GQA
    paged KV
    target 的 prefill / decode / 混合批（q_len 任意）
    fp16 / bf16 / fp32
    无量化 / 无 sliding / 无 cascade / 无 sinks / 无 ALiBi / 无 softcap

一个 program 负责一条请求的一个 query head：先按 BLOCK_M 循环 query 行，再按
BLOCK_N 循环 KV 块。decode 时 q_len=1..K+1，通常只走一个 query block；prefill 时
按 query block 继续循环（同一请求的 K/V 会被多个 query block 重读，这是固定场景的
取舍：decode 优先）。

所有动态长度来自 GPU 张量（seq_lens/query_start_loc），没有 `.item()`、没有 Python
请求循环；grid 只由 Python int（num_reqs、num_heads）决定，可进 CUDA Graph。
"""

from __future__ import annotations

import torch

from ...triton_utils import tl, triton


@triton.jit
def _fixed_paged_attention_kernel(
    q_ptr, out_ptr, k_ptr, v_ptr,
    block_table_ptr, seq_lens_ptr, query_start_loc_ptr,
    stride_qn, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_btn, stride_btb,
    stride_on, stride_oh, stride_od,
    SM_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCKS_PER_REQ: tl.constexpr,
    NUM_REQS: tl.constexpr,
    USE_BLOCK_MAP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # program = (global query block, kv head)
    # 一个 program 把 GROUP_SIZE 个 query head 打包进 BLOCK_M 行：
    #   BLOCK_Q = BLOCK_M // GROUP_SIZE 个 query 位置 × GROUP_SIZE 个 head
    # 这样 K/V 在同一 GQA 组内只读一遍，grid 第二维是 num_kv_heads，不是 num_heads。
    pid = tl.program_id(0)
    kv_head = tl.program_id(1)
    BLOCK_Q: tl.constexpr = BLOCK_M // GROUP_SIZE

    if USE_BLOCK_MAP:
        # 上游同款：grid 第 0 维是整个 batch 的 block 上界
        #   floor(N / BLOCK_M) + num_reqs
        # 用 GPU 上的 query_start_loc 做二分，把 pid 映射到 (req, q_blk)。
        # 这样不需要“每请求固定 block 数”，prefill 不会因为某条请求长而
        # 给所有请求都分配同样多的空 block。
        left = tl.full([], 0, tl.int32)
        right = tl.full([], NUM_REQS, tl.int32)
        while left < right:
            mid = (left + right) // 2
            mid_val = tl.load(query_start_loc_ptr + mid) // BLOCK_Q + mid
            if mid_val <= pid:
                left = mid + 1
            else:
                right = mid
        req = left - 1
        q_start = tl.load(query_start_loc_ptr + req)
        q_block_start = q_start // BLOCK_Q + req
        q_blk = pid - q_block_start
    else:
        # decode 已知 max_query_len（Python int），用它得到紧上界：
        #   pid0 = req * BLOCKS_PER_REQ + q_blk
        req = pid // BLOCKS_PER_REQ
        q_blk = pid % BLOCKS_PER_REQ
        q_start = tl.load(query_start_loc_ptr + req)

    q_end = tl.load(query_start_loc_ptr + req + 1)
    q_len = q_end - q_start
    seq_len = tl.load(seq_lens_ptr + req)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # 这个 query block 可能超过该请求的真实 q_len（grid 是上界对齐的）。
    if q_blk * BLOCK_Q < q_len:
        # 16 行 = (query 位置, 组内 head) 的笛卡尔积。
        local_q = offs_m // GROUP_SIZE
        q_head = kv_head * GROUP_SIZE + (offs_m % GROUP_SIZE)
        row = q_start + q_blk * BLOCK_Q + local_q
        valid_m = local_q < q_len - q_blk * BLOCK_Q

        q = tl.load(
            q_ptr + row[:, None] * stride_qn + q_head[:, None] * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=valid_m[:, None] & mask_d[None, :], other=0.0,
        ).to(tl.float32)

        # query 的绝对位置：seq_len - q_len + 行在请求内的位置。
        q_pos = seq_len - q_len + q_blk * BLOCK_Q + local_q

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        z_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for blk in range(0, tl.cdiv(seq_len, BLOCK_N)):
            log_n = blk * BLOCK_N + offs_n
            valid_n = log_n < seq_len

            # 每个逻辑 key 单独查物理块；计算块可以跨多个物理块。
            phys = tl.load(
                block_table_ptr + req * stride_btn
                + (log_n // BLOCK_SIZE) * stride_btb,
                mask=valid_n, other=0,
            )
            blk_off = log_n % BLOCK_SIZE
            kv_offsets = (
                phys[:, None] * stride_kb + blk_off[:, None] * stride_ks
                + kv_head * stride_kh + offs_d[None, :] * stride_kd
            )
            k = tl.load(
                k_ptr + kv_offsets,
                mask=valid_n[:, None] & mask_d[None, :], other=0.0,
            ).to(tl.float32)

            # [BLOCK_M, D] @ [D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            scores = tl.dot(q, tl.trans(k)) * SM_SCALE
            # causal + padding：无效 key 和未来 key 都 mask。
            keep = valid_n[None, :] & (log_n[None, :] <= q_pos[:, None])
            scores = tl.where(keep, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            z_i = z_i * alpha + tl.sum(p, axis=1)

            v_offsets = (
                phys[:, None] * stride_vb + blk_off[:, None] * stride_vs
                + kv_head * stride_vh + offs_d[None, :] * stride_vd
            )
            v = tl.load(
                v_ptr + v_offsets,
                mask=valid_n[:, None] & mask_d[None, :], other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # seq_len=0 / 全无效行：0/0 会产生 NaN；真实行不会走到这里。
        out = tl.where(z_i[:, None] > 0.0, acc / z_i[:, None], 0.0)
        tl.store(
            out_ptr + row[:, None] * stride_on + q_head[:, None] * stride_oh
            + offs_d[None, :] * stride_od,
            out.to(out_ptr.dtype.element_ty),
            mask=valid_m[:, None] & mask_d[None, :],
        )


def fixed_paged_attention(
    q: torch.Tensor,
    out: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    block_size: int,
    scale: float,
    max_query_len: int | None = None,
) -> torch.Tensor:
    """固定场景 launcher；q/out [N,H,D]，k/v [blocks, block_size, Hk, D]。"""
    # q 是扁平行 [N,H,D]；num_reqs 来自 metadata，不是 N。
    num_heads, head_dim = q.shape[1], q.shape[2]
    num_reqs = seq_lens.numel()
    group_size = num_heads // k.shape[2]
    block_m = 16
    block_n = 64
    block_d = max(16, triton.next_power_of_2(head_dim))
    if group_size > block_m or block_m % group_size != 0:
        raise ValueError(
            f"fixed kernel 要求 GQA group_size 整除 {block_m}，收到 {group_size}")
    block_q = block_m // group_size
    # 每请求的 block 数必须是 Python int（图内不能 .item()）：
    #   - uniform/varlen decode 有每请求上界，用它得到紧 grid；
    #   - 混合 prefill 没有单请求上界，用“总 block 上界 + kernel 内二分映射”，
    #     避免给每条请求都分配 ceil(N/BLOCK_M) 个 block。
    if max_query_len is not None:
        blocks_per_req = max(1, triton.cdiv(max_query_len, block_q))
        num_blocks = num_reqs * blocks_per_req
        use_block_map = False
    else:
        blocks_per_req = 1
        num_blocks = q.shape[0] // block_q + num_reqs
        use_block_map = True
    num_kv_heads = k.shape[2]
    _fixed_paged_attention_kernel[(num_blocks, num_kv_heads)](
        q, out, k, v,
        block_table, seq_lens, query_start_loc,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        SM_SCALE=scale,
        BLOCK_SIZE=block_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCKS_PER_REQ=blocks_per_req,
        NUM_REQS=num_reqs,
        USE_BLOCK_MAP=use_block_map,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        GROUP_SIZE=group_size,
    )
    return out
