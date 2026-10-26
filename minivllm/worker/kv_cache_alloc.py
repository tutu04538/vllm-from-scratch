"""KV cache 的物理分配：旧 per-layer 布局与 packed shared slab（81/E2）。

控制面只决定“块 id 怎么分”，这里决定“块 id 对应哪段显存”。两种布局：

    per-layer          每层一个完整 [2, B, block, kvh, head] tensor
    packed shared slab 一个 int8 backing，每个 block id 一行；
                       每层是 `[block_stride]` 行里 offset 处的一段 view

packed 的关键不变量与上游相同：**一个 block id 同一时刻只归一个 KV group**。
不同 group 的 offset 可以重叠，但不会同时读同一 id，所以不会串 K/V。
"""

from __future__ import annotations

from typing import Any

import torch

from ..attention.layer import Attention
from ..core.kv_cache_interface import (KVCacheConfig, KVCacheSpec,
                                       make_packed_kv_cache_view)


def allocate_packed_target_caches(
    kv_cache_config: KVCacheConfig,
    target_layers: dict[str, Attention],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """为所有 owner 层分配一份 packed backing 并返回逐层 view。

    返回 `(layer_name -> view, backing, block_stride)`；`backing` 必须被 Runner 持有，
    虽然 view 也会持有 storage，但显式保存便于排查“谁还活着”。
    """
    if kv_cache_config.num_kv_cache_groups <= 1:
        raise ValueError("packed 布局只用于多 group；单 group 走 per-layer 分配")
    block_stride, tensors = kv_cache_config.packed_tensors()
    total_size = block_stride * kv_cache_config.num_gpu_blocks
    backing = torch.zeros(total_size, dtype=torch.int8, device=device)

    caches: dict[str, torch.Tensor] = {}
    for tensor in tensors:
        for layer_name in tensor.shared_by:
            layer = target_layers.get(layer_name)
            if layer is None:
                raise KeyError(f"packed 布局里的层 {layer_name!r} 不在 target 模型中")
            if layer.q_only:
                raise ValueError(
                    f"packed 布局的拥有层 {layer_name!r} 是 Q-only；Q-only 不拥有 cache")
            group_id = kv_cache_config.group_id_for_layer(layer_name)
            spec: KVCacheSpec = kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            cache = make_packed_kv_cache_view(
                backing, spec,
                offset=tensor.offset,
                block_stride=block_stride,
                num_blocks=kv_cache_config.num_gpu_blocks,
            )
            layer.kv_cache = cache
            layer.kv_cache_group_id = group_id
            caches[layer_name] = cache
    if len(caches) != sum(len(g.layer_names) for g in kv_cache_config.kv_cache_groups):
        raise RuntimeError("packed 布局没有覆盖所有 KV group 的拥有层")
    return caches, backing, block_stride


def allocate_per_layer_target_caches(
    kv_cache_config: KVCacheConfig,
    target_layers: dict[str, Attention],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """旧布局：每层一份完整 cache（单 group 或兼容路径）。"""
    caches: dict[str, torch.Tensor] = {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            layer = target_layers[layer_name]
            if layer.q_only:
                raise ValueError(f"拥有层 {layer_name!r} 是 Q-only")
            cache = torch.zeros(
                2, kv_cache_config.num_gpu_blocks, spec.block_size,
                spec.num_kv_heads, spec.head_size,
                dtype=spec.dtype, device=device)
            layer.kv_cache = cache
            layer.kv_cache_group_id = group_id
            caches[layer_name] = cache
    return caches
