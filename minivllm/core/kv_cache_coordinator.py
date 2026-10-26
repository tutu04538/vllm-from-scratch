"""KV 多组协调者（对应 vLLM `v1/core/kv_cache_coordinator.py` 的子集）。

**这个文件回答的问题**：一条请求的同一段 token，在 sliding / full / draft 多个 KV
group 里分别能命中多少？只有一个共同前缀长度是安全的——某个 group 少一块，整段就不能
算命中，否则执行侧会读到一段"这一组根本没算过"的 KV。

    UnitaryKVCacheCoordinator   单组（历史路径，行为与原实现一致）
    HybridKVCacheCoordinator    多组：逐组查命中，取共同长度，再按共同长度裁块

**共同长度用不动点迭代算**：每组在自己的 block 粒度上返回命中长度；取 min 后再让所有
组按新长度重查，直到不再下降。这样不会出现"组 A 返回 32、组 B 返回 16，却只把 A 裁到
16 个 token、块数没对齐"的半块状态。

**本文件的边界**：滑窗组只做掩码，不做窗口外块回收（见 `SlidingWindowManager`）；
Mamba 等非注意力状态尚未实现，配置期/构造期不会假装支持。
"""

from __future__ import annotations

import math

from .kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheSpec,
                                 SlidingWindowSpec, full_attention_config)
from .single_type_kv_cache_manager import (FullAttentionManager,
                                           SlidingWindowManager)


class HybridKVCacheCoordinator:
    """按 `KVCacheConfig` 建每组一个 manager，并把命中长度对齐到所有组。"""

    def __init__(self, block_pool, enable_caching: bool,
                 kv_cache_config: KVCacheConfig) -> None:
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self.kv_cache_config = kv_cache_config
        self.hash_block_size = kv_cache_config.hash_block_size
        self.managers = []
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if isinstance(spec, SlidingWindowSpec):
                manager = SlidingWindowManager(
                    block_pool, enable_caching, spec.block_size,
                    kv_cache_group_id=group_id,
                    sliding_window=spec.sliding_window,
                    hash_block_size=self.hash_block_size,
                )
            elif isinstance(spec, FullAttentionSpec):
                manager = FullAttentionManager(
                    block_pool, enable_caching, spec.block_size,
                    kv_cache_group_id=group_id,
                    hash_block_size=self.hash_block_size,
                )
            else:
                raise NotImplementedError(
                    f"KV group {group_id} 的 spec 类型 {type(spec).__name__} 尚未实现："
                    f"本仓库只有 FullAttention/SlidingWindow 两类注意力 cache，"
                    f"Mamba 等混合状态不能按普通 KV 处理")
            manager.layer_names = list(group.layer_names)
            self.managers.append(manager)

    # -------- 观察 --------

    @property
    def block_size(self) -> int:
        return self.hash_block_size

    @property
    def num_reqs(self) -> int:
        return sum(manager.num_reqs() for manager in self.managers)

    def get_blocks(self, request_id: str):
        return tuple(manager.get_blocks(request_id) for manager in self.managers)

    def get_blocks_by_group(self, request_id: str) -> tuple[list, ...]:
        return self.get_blocks(request_id)

    # -------- 命中 --------

    def find_longest_cache_hit(self, block_hashes, max_length: int):
        """返回 `([每组命中块], 共同命中 token 数)`。

        `max_length` 是 scheduler 的 hash 粒度上界；每个 manager 自己换算成本组块粒度。
        """
        if max_length <= 0 or not block_hashes:
            return tuple([] for _ in self.managers), 0
        hit_length = max_length
        blocks_by_group: list[list] = [[] for _ in self.managers]
        # 长度单调下降，最多下降 len(managers)+1 轮就会稳定；用固定上限防御实现错误。
        for _ in range(len(self.managers) + 2):
            new_length = hit_length
            for group_id, manager in enumerate(self.managers):
                blocks, length = manager.find_longest_cache_hit(
                    block_hashes, min(hit_length, max_length))
                blocks_by_group[group_id] = blocks
                new_length = min(new_length, length)
            if new_length == hit_length:
                break
            hit_length = new_length
            if hit_length == 0:
                break
        # 按共同长度裁块：manager 可能返回了比共同长度多的块（特别是 full group）。
        for group_id, manager in enumerate(self.managers):
            num_blocks = hit_length // manager.block_size
            blocks_by_group[group_id] = blocks_by_group[group_id][:num_blocks]
        hit_length = min(
            [len(blocks) * manager.block_size
             for blocks, manager in zip(blocks_by_group, self.managers)]
            or [0])
        return tuple(blocks_by_group), hit_length

    # -------- 容量与分配 --------

    def get_num_blocks_to_allocate(self, request_id: str, num_tokens: int,
                                   new_computed_blocks=()) -> int:
        total = 0
        for group_id, manager in enumerate(self.managers):
            blocks = (new_computed_blocks[group_id]
                      if group_id < len(new_computed_blocks) else ())
            total += manager.get_num_blocks_to_allocate(
                request_id, num_tokens, blocks)
        return total

    def allocate_new_computed_blocks(self, request_id: str, new_computed_blocks,
                                     num_local_computed_tokens: int) -> None:
        """先 touch 所有组的命中块，再分配；失败时调用方已经做过容量检查。"""
        if len(new_computed_blocks) != len(self.managers):
            raise ValueError(
                f"命中块有 {len(new_computed_blocks)} 组，但 KV cache 有 "
                f"{len(self.managers)} 组：组数与层映射不一致")
        for manager, blocks in zip(self.managers, new_computed_blocks):
            manager.add_local_computed_blocks(
                request_id, blocks, num_local_computed_tokens)

    def allocate_new_blocks(self, request_id: str, num_tokens: int):
        return tuple(manager.allocate_new_blocks(request_id, num_tokens)
                     for manager in self.managers)

    def cache_blocks(self, request, num_tokens: int) -> None:
        for manager in self.managers:
            manager.cache_blocks(request, num_tokens)

    def free(self, request_id: str) -> None:
        blocks = []
        for manager in self.managers:
            blocks.extend(manager.free(request_id))
        self.block_pool.free_blocks(blocks)

    def get_num_common_prefix_blocks(self) -> list[int]:
        """每组所有活请求共同的前缀块数（trace/调试用）。"""
        req_ids = set()
        for manager in self.managers:
            req_ids.update(manager.req_to_blocks)
        result = []
        for manager in self.managers:
            rows = [manager.req_to_blocks.get(req_id, []) for req_id in req_ids]
            common = 0
            for blocks in zip(*rows):
                if len({block.block_id for block in blocks}) != 1:
                    break
                common += 1
            result.append(common)
        return result


class UnitaryKVCacheCoordinator(HybridKVCacheCoordinator):
    """单组兼容层：旧调用点仍然可以 `coordinator.manager` 取到唯一 manager。"""

    def __init__(self, block_pool, enable_caching: bool, block_size: int,
                 kv_cache_config: KVCacheConfig | None = None) -> None:
        if kv_cache_config is None:
            # 只关心块分配的旧路径：用一个占位 spec 包成单组。物理缓存由 Runner 按真实
            # Attention 形状另建，这里只做 block 账本。
            class _CacheConfig:
                pass
            cache = _CacheConfig()
            cache.block_size = block_size
            cache.num_gpu_blocks = block_pool.num_gpu_blocks
            kv_cache_config = full_attention_config(cache)
        if kv_cache_config.num_kv_cache_groups != 1:
            raise ValueError(
                f"UnitaryKVCacheCoordinator 只接受 1 个 KV group，收到 "
                f"{kv_cache_config.num_kv_cache_groups} 个；多组请用 HybridKVCacheCoordinator")
        super().__init__(block_pool, enable_caching, kv_cache_config)

    @property
    def manager(self) -> FullAttentionManager:
        return self.managers[0]
