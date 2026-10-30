"""按 slot 写 paged KV 的 Triton kernel（086）。

对齐上游 `v1/attention/ops/triton_reshape_and_cache_flash.py` 的核心语义：

    * slot_mapping 在 kernel 内读；`slot < 0` 直接返回，不写“0 号垃圾桶”；
    * 接收 K/V cache 的实际 stride（packed slab 的 block 维可以不是连续页）；
    * 源 K/V 支持 packed QKV 的非连续 row stride（按 token/head 寻址）。

不做 FP8/INT4/per-token-head 量化分支；这些组合在 backend 入口明确拒绝。
"""

from __future__ import annotations

import torch

from ...triton_utils import tl, triton


@triton.jit
def reshape_and_cache_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    key_stride_tok: tl.int64,
    key_stride_head: tl.int64,
    value_stride_tok: tl.int64,
    value_stride_head: tl.int64,
    kc_stride_block: tl.int64,
    kc_stride_slot: tl.int64,
    kc_stride_head: tl.int64,
    kc_stride_dim: tl.int64,
    vc_stride_block: tl.int64,
    vc_stride_slot: tl.int64,
    vc_stride_head: tl.int64,
    vc_stride_dim: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot < 0:
        return

    block_idx = slot // BLOCK_SIZE
    block_offset = slot % BLOCK_SIZE

    tile_idx = tl.program_id(1)
    pos = tile_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    mask = pos < NUM_HEADS * HEAD_SIZE
    head = pos // HEAD_SIZE
    dim = pos % HEAD_SIZE

    key = tl.load(
        key_ptr + token_idx * key_stride_tok
        + head * key_stride_head + dim,
        mask=mask,
        other=0.0,
    )
    value = tl.load(
        value_ptr + token_idx * value_stride_tok
        + head * value_stride_head + dim,
        mask=mask,
        other=0.0,
    )

    k_target = (
        key_cache_ptr
        + block_idx * kc_stride_block
        + block_offset * kc_stride_slot
        + head * kc_stride_head
        + dim * kc_stride_dim
    )
    v_target = (
        value_cache_ptr
        + block_idx * vc_stride_block
        + block_offset * vc_stride_slot
        + head * vc_stride_head
        + dim * vc_stride_dim
    )
    tl.store(k_target, key, mask=mask)
    tl.store(v_target, value, mask=mask)


def triton_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """把 `key/value [tokens, kv_heads, head_size]` 写到分页 cache。

    cache 形状 `[num_blocks, block_size, kv_heads, head_size]`，但允许 block
    维是非连续 stride（packed shared slab）。
    """
    if triton is None:
        raise RuntimeError("Triton 不可用，不能使用 TRITON_ATTN 后端")
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            f"key/value 必须是 [tokens, kv_heads, head_size]，收到 "
            f"{tuple(key.shape)} / {tuple(value.shape)}")
    if key.shape != value.shape:
        raise ValueError(f"key/value 形状不一致：{tuple(key.shape)} vs {tuple(value.shape)}")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("key_cache/value_cache 必须是 4D")
    if key_cache.shape[2:] != key.shape[1:]:
        raise ValueError(
            f"cache 的 kv_heads/head_size 与 key 不符：cache="
            f"{tuple(key_cache.shape[2:])}, key={tuple(key.shape[1:])}")

    num_tokens = key.shape[0]
    if num_tokens == 0:
        return
    if slot_mapping.numel() < num_tokens:
        raise ValueError(
            f"slot_mapping 只有 {slot_mapping.numel()} 个，少于 {num_tokens} 个 token")
    if key_cache.dtype != key.dtype or value_cache.dtype != value.dtype:
        raise ValueError(
            f"K/V cache dtype 必须与源张量一致："
            f"k={key_cache.dtype}/{key.dtype}, v={value_cache.dtype}/{value.dtype}")

    num_heads, head_size = key.shape[1], key.shape[2]
    block_size = key_cache.shape[1]
    tile_size = 1024
    grid = (num_tokens, triton.cdiv(num_heads * head_size, tile_size))
    reshape_and_cache_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
    )
