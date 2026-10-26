"""81 子集：packed shared slab 的布局与视图。

这个文件只验证**物理布局**，不重复 Gemma4 端到端（后者在 step80）。固定例子
G0=2 层、G1=1 层，两组 page size 不同，block_stride 必须取 max 而不是 sum。
"""

from __future__ import annotations

import torch

from minivllm.core.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                              KVCacheGroup,
                                              make_packed_kv_cache_view)


def _config() -> KVCacheConfig:
    g0 = KVCacheGroup(
        FullAttentionSpec(block_size=4, num_kv_heads=1,
                          head_size=8, dtype=torch.float32),
        ["g0.l0", "g0.l1"],
    )
    g1 = KVCacheGroup(
        FullAttentionSpec(block_size=4, num_kv_heads=1,
                          head_size=16, dtype=torch.float32),
        ["g1.l0"],
    )
    return KVCacheConfig(num_gpu_blocks=3, kv_cache_groups=[g0, g1])


def test_packed_layout_uses_max_group_stack_and_offsets():
    cfg = _config()
    block_stride, layers = cfg.packed_layout()
    # G0: 2 * (2*4*1*8*4) = 2*256 = 512 B；G1: 1 * (2*4*1*16*4) = 512 B
    assert block_stride == 512
    assert layers[0] == ["g0.l0", "g1.l0"]
    assert layers[256] == ["g0.l1"]
    assert cfg.layer_offsets() == (512, {"g0.l0": 0, "g1.l0": 0, "g0.l1": 256})

    stride, tensors = cfg.packed_tensors()
    assert stride == 512
    assert [t.offset for t in tensors] == [0, 256]
    assert all(t.size == 512 * 3 and t.block_stride == 512 for t in tensors)


def test_packed_views_alias_one_backing_and_have_block_stride():
    cfg = _config()
    stride, offsets = cfg.layer_offsets()
    backing = torch.zeros(stride * cfg.num_gpu_blocks, dtype=torch.int8)
    caches = {
        name: make_packed_kv_cache_view(
            backing, cfg.kv_cache_groups[cfg.group_id_for_layer(name)].kv_cache_spec,
            offset=offset, block_stride=stride,
            num_blocks=cfg.num_gpu_blocks)
        for name, offset in offsets.items()
    }
    assert len({cache.untyped_storage().data_ptr() for cache in caches.values()}) == 1
    assert all(cache.untyped_storage().data_ptr()
               == backing.untyped_storage().data_ptr() for cache in caches.values())
    for name, cache in caches.items():
        assert cache.stride(1) == stride // 4
        assert cache.storage_offset() == offsets[name] // 4
        assert cache.shape == (2, 3, 4, 1,
                               8 if name.startswith("g0") else 16)

    # 直接写 view，检查它确实落在 backing 的对应 offset，而不是另开一份。
    g0 = caches["g0.l0"]
    g1 = caches["g1.l0"]
    g0[0, 1, 2, 0, :] = torch.arange(8, dtype=torch.float32)
    g1[0, 2, 3, 0, :] = torch.arange(16, dtype=torch.float32) + 100
    flat = backing.view(torch.float32)
    assert flat[512 // 4 + 2 * 8: 512 // 4 + 2 * 8 + 8].tolist() == list(range(8))
    assert flat[2 * 512 // 4 + 3 * 16: 2 * 512 // 4 + 3 * 16 + 16].tolist() == [
        float(100 + i) for i in range(16)]


def test_packed_layout_rejects_page_overflow():
    cfg = _config()
    backing = torch.zeros(cfg.packed_layout()[0] * cfg.num_gpu_blocks,
                          dtype=torch.int8)
    spec = cfg.kv_cache_groups[0].kv_cache_spec
    try:
        make_packed_kv_cache_view(
            backing, spec, offset=256, block_stride=256,
            num_blocks=cfg.num_gpu_blocks)
    except ValueError as exc:
        assert "超出" in str(exc) or "page" in str(exc)
    else:
        raise AssertionError("越界 page 必须报错")
