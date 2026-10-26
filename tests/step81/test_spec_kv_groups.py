"""81 关子集：多 KV group 的块分配 / 前缀命中 / 组间共同边界。

**本文件只覆盖已经接进生产的控制面**：FullAttention + SlidingWindow 两组、不同
block_size、组 id 分开的 cache 索引。Mamba 混合状态、抢占三时刻、KV transfer 不在本文件；
它们仍属 81 未完成项（见 `docs/step81_alignment.md` 的状态表）。
"""

from __future__ import annotations

import pytest
import torch

from minivllm import CacheConfig, SamplingParams
from minivllm.core.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                              KVCacheGroup, SlidingWindowSpec)
from minivllm.core.kv_cache_manager import KVCacheManager
from minivllm.core.kv_cache_utils import (BlockHasher,
                                          make_block_hash_with_group_id)
from minivllm.request import Request


def _manager() -> KVCacheManager:
    cfg = KVCacheConfig(
        num_gpu_blocks=16,
        kv_cache_groups=[
            KVCacheGroup(
                FullAttentionSpec(block_size=4, num_kv_heads=1,
                                  head_size=8, dtype=torch.float32),
                ["model.layers.0.self_attn.attn"]),
            KVCacheGroup(
                SlidingWindowSpec(block_size=8, num_kv_heads=1,
                                  head_size=16, dtype=torch.float32,
                                  sliding_window=8),
                ["model.layers.1.self_attn.attn"]),
        ],
        enable_prefix_caching=True,
    )
    return KVCacheManager(cfg, max_model_len=64)


def _request(req_id: str, length: int = 16) -> Request:
    return Request(
        req_id, list(range(length)),
        SamplingParams(max_tokens=4, eos_token_id=999),
        block_hasher=BlockHasher(4),   # min(4, 8)
    )


def test_group_block_sizes_and_shared_hit_boundary():
    manager = _manager()
    assert manager.block_size == 4          # scheduler/hash 粒度 = min(4, 8)
    assert [m.block_size for m in manager.coordinator.managers] == [4, 8]

    first = _request("a")
    blocks = manager.allocate_slots(first, 16)
    assert blocks is not None
    group_ids = blocks.get_block_ids()
    assert [len(ids) for ids in group_ids] == [4, 2]
    manager.cache_blocks(first, 16)

    second = _request("b")
    hit, hit_length = manager.get_computed_blocks(second)
    # 留一个 token 给本轮计算，所以最多命中 15；两组合并后是 8（sliding 的块大小）。
    assert hit_length == 8
    assert [len(ids) for ids in hit.get_block_ids()] == [2, 1]


def test_partial_group_hit_does_not_false_hit():
    manager = _manager()
    first = _request("a")
    manager.allocate_slots(first, 16)
    manager.cache_blocks(first, 16)
    assert manager.get_computed_blocks(_request("b"))[1] == 8

    # 只删 sliding 组第一块的索引：full 组仍有同一段，但共同长度必须是 0。
    key = make_block_hash_with_group_id(first.block_hashes[1], 1)
    sliding_block = manager.coordinator.managers[1].get_blocks("a")[0]
    manager.block_pool.cached_block_hash_to_block.pop(key, sliding_block.block_id)
    second = _request("b")
    _blocks, hit_length = manager.get_computed_blocks(second)
    assert hit_length == 0, "一个 group 缺块时不能按另一个 group 的前缀误命中"


def test_different_block_size_allocation_is_atomic_on_failure():
    cfg = KVCacheConfig(
        num_gpu_blocks=2,          # 8 块组一请求就要 1 块；两组共需 3 块
        kv_cache_groups=[
            KVCacheGroup(
                FullAttentionSpec(4, 1, 8, torch.float32),
                ["a"]),
            KVCacheGroup(
                SlidingWindowSpec(8, 1, 16, torch.float32, sliding_window=8),
                ["b"]),
        ],
        enable_prefix_caching=False,
    )
    manager = KVCacheManager(cfg, max_model_len=64)
    request = _request("r", 16)
    assert manager.allocate_slots(request, 16) is None
    # 失败后不能留下半张块表
    assert manager.get_blocks("r").get_block_ids() == ([], [])


def test_sliding_spec_rejects_missing_window():
    with pytest.raises(ValueError, match="sliding_window"):
        SlidingWindowSpec(4, 1, 8, torch.float32, sliding_window=0)
