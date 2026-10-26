"""KV cache 的规格与分组（对应 vLLM `v1/kv_cache_interface.py` 的最小子集）。

**为什么不能再用一个 `block_size` 标量代表整张模型**：Gemma4 这类模型同时有

    sliding_attention 层   每层 1 个 KV head、head_dim=256、window=512
    full_attention 层      每层 1 个 KV head、head_dim=512、没有窗口

两类层的 KV 形状不同、逻辑块表也不同。scheduler/prefix cache 必须按 group 记账，
执行侧的每层 attention metadata 也必须拿到**自己那一组**的块表。把它们混成一组会
"不报错地读错层"：full 层的块表和 sliding 层长度不一样，shape 检查也拦不住。

本文件的层次与上游一一对应：

    KVCacheSpec        一类 KV 的形状 + block_size + 是否滑窗
    KVCacheGroup       一组必须一起命中的层（同一 spec）
    KVCacheConfig      一次运行的全部分组 + 总物理块数

**block_size 的口径**：组内 `block_size` 是实际物理块大小；scheduler/块 hash 用所有组
block_size 的 **min**（`KVCacheConfig.hash_block_size`）。每个组的块大小必须是它的整数倍；
组块比 hash 块大时用 `stride` 取对应节点的 hash。不能反过来（4 块组不能用 8 块的 hash
发布自己的第一块），否则前缀缓存会静默退化。跨组共同命中长度由协调者取 min 并要求每组
在自己的块边界上，不会把半块当成完整命中。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class KVCacheSpec:
    """一类 KV cache 的规格（对应上游 `KVCacheSpec` 的必需字段）。

    `page_size_bytes` 只用于容量核算与测试；真正的执行形状由 Runner 按
    `num_kv_heads/head_size` 建。`dtype` 必须是 torch dtype，不能用字符串——执行侧要拿它
    直接建 tensor，字符串会在第一次分配时才炸。
    """

    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size 必须为正，收到 {self.block_size}")
        if self.num_kv_heads <= 0 or self.head_size <= 0:
            raise ValueError(
                "num_kv_heads/head_size 必须为正，收到 "
                f"{self.num_kv_heads}/{self.head_size}")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(f"KVCacheSpec.dtype 必须是 torch.dtype，收到 {self.dtype!r}")

    @property
    def page_size_bytes(self) -> int:
        """一个物理块里 K+V 的字节数（不含 allocator 对齐）。"""
        return (2 * self.block_size * self.num_kv_heads * self.head_size
                * self.dtype.itemsize)


@dataclass(frozen=True)
class FullAttentionSpec(KVCacheSpec):
    """全注意力组：每个位置都可能在之后被读到（不能释放旧块）。"""


@dataclass(frozen=True)
class SlidingWindowSpec(KVCacheSpec):
    """滑窗注意力组（对应上游 `SlidingWindowSpec`）。

    `sliding_window` 的语义与 HF/vLLM 一致：位置 q 的 query 能看到
    `q - sliding_window + 1 .. q`（包含自己）。本项目的 Torch 后端按这条掩码执行；
    **不释放窗口外的块**（保留全历史），因此前缀命中仍是全前缀命中，正确性不受影响，
    代价只是显存没有按窗口回收。这一条是明确边界，不是"滑窗已完全对齐"。
    """

    sliding_window: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sliding_window <= 0:
            raise ValueError(f"sliding_window 必须为正，收到 {self.sliding_window}")


@dataclass
class KVCacheGroup:
    """一组 layer_names + 它们共用的 spec（对应上游 `KVCacheGroup`）。"""

    kv_cache_spec: KVCacheSpec
    layer_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layer_names:
            raise ValueError("KVCacheGroup 至少要有一层")
        if len(set(self.layer_names)) != len(self.layer_names):
            raise ValueError(f"KVCacheGroup 里有重复层名：{self.layer_names}")


@dataclass(frozen=True)
class KVCacheTensor:
    """packed 布局里的一段物理 backing（对应上游 `KVCacheTensor`）。

    `size` 是整个 backing 的字节数；`offset` 是这一层在一个 block 行内的字节偏移；
    `block_stride` 是一行的字节数（0 = 旧的非 packed 布局）。
    """

    size: int
    shared_by: list[str]
    offset: int = 0
    block_stride: int = 0

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"KVCacheTensor.size 必须为正，收到 {self.size}")
        if not self.shared_by:
            raise ValueError("KVCacheTensor.shared_by 不能为空")
        if self.offset < 0 or self.block_stride < 0:
            raise ValueError("KVCacheTensor 的 offset/block_stride 不能为负")


@dataclass
class KVCacheConfig:
    """一次运行的 KV 布局（对应上游 `KVCacheConfig` 的子集）。

    `num_gpu_blocks` 是**全局共享 block id 池**的块数；所有 group manager 从同一个
    `BlockPool` 领块。packed 布局会把各层视图放进同一个 backing，但 block id 仍是全局
    唯一、同一时刻只归一个 group。
    """

    num_gpu_blocks: int
    kv_cache_groups: list[KVCacheGroup] = field(default_factory=list)
    enable_prefix_caching: bool = False

    def __post_init__(self) -> None:
        if self.num_gpu_blocks <= 0:
            raise ValueError(f"num_gpu_blocks 必须为正，收到 {self.num_gpu_blocks}")
        if not self.kv_cache_groups:
            raise ValueError("KVCacheConfig 至少要有一个 KV cache group")
        seen: set[str] = set()
        for group in self.kv_cache_groups:
            overlap = seen.intersection(group.layer_names)
            if overlap:
                raise ValueError(
                    f"同一个注意力层不能属于两个 KV group：{sorted(overlap)}")
            seen.update(group.layer_names)

    @property
    def num_kv_cache_groups(self) -> int:
        return len(self.kv_cache_groups)

    @property
    def block_sizes(self) -> list[int]:
        return [group.kv_cache_spec.block_size for group in self.kv_cache_groups]

    @property
    def hash_block_size(self) -> int:
        """scheduler/块 hash 的粒度 = 所有组 block_size 的**最小值**。

        每组 block_size 必须是它的整数倍（组块可以比 hash 块大，反之不行）；这样
        哈希链的每个节点都落在最小的物理块边界上，组块用 `stride` 取对应节点的
        hash。用 lcm 会让 4+8 这种组合得到 8，导致 4 块组**无法单独发布/命中**
        自己的第一块（块 hash 不存在），前缀缓存会静默退化。
        """
        return min(self.block_sizes)

    @property
    def block_size(self) -> int:
        """兼容旧调用点的别名：scheduler 的 hash 粒度。"""
        return self.hash_block_size

    def group_id_for_layer(self, layer_name: str) -> int:
        for group_id, group in enumerate(self.kv_cache_groups):
            if layer_name in group.layer_names:
                return group_id
        raise KeyError(f"层 {layer_name!r} 不在任何 KV group 里")

    def packed_layout(self) -> tuple[int, dict[int, list[str]]]:
        """返回 `(block_stride_bytes, offset -> layer_names)`（对应上游
        `_get_packed_kv_cache_layout`）。

        每组的层按顺序累加 `page_size_bytes`；不同组的 offset 可以重叠。
        `block_stride` 取所有组 stack 的最大值，而不是各组之和。
        """
        layers_by_offset: dict[int, list[str]] = defaultdict(list)
        block_stride = 0
        for group in self.kv_cache_groups:
            page_size = group.kv_cache_spec.page_size_bytes
            byte_offset = 0
            for layer_name in group.layer_names:
                layers_by_offset[byte_offset].append(layer_name)
                byte_offset += page_size
            block_stride = max(block_stride, byte_offset)
        if block_stride <= 0:
            raise RuntimeError("packed layout 的 block_stride 必须为正")
        return block_stride, dict(layers_by_offset)

    def packed_tensors(self) -> tuple[int, list[KVCacheTensor]]:
        """生成 packed 布局的 `KVCacheTensor` 描述（全部 aliasing 同一个 backing）。"""
        block_stride, layers_by_offset = self.packed_layout()
        total_size = block_stride * self.num_gpu_blocks
        tensors = [
            KVCacheTensor(
                size=total_size,
                shared_by=list(layers),
                offset=offset,
                block_stride=block_stride,
            )
            for offset, layers in sorted(layers_by_offset.items())
        ]
        return block_stride, tensors

    def layer_offsets(self) -> tuple[int, dict[str, int]]:
        """layer_name -> 它在 packed block 行内的字节 offset。"""
        block_stride, layers_by_offset = self.packed_layout()
        offsets = {}
        for offset, layers in layers_by_offset.items():
            for layer_name in layers:
                offsets[layer_name] = offset
        return block_stride, offsets

    def describe(self) -> list[dict]:
        """给 trace/测试用的稳定摘要。"""
        return [
            {
                "group_id": group_id,
                "layer_names": list(group.layer_names),
                "kind": type(group.kv_cache_spec).__name__,
                "block_size": group.kv_cache_spec.block_size,
                "num_kv_heads": group.kv_cache_spec.num_kv_heads,
                "head_size": group.kv_cache_spec.head_size,
                "dtype": str(group.kv_cache_spec.dtype),
                "sliding_window": getattr(
                    group.kv_cache_spec, "sliding_window", None),
            }
            for group_id, group in enumerate(self.kv_cache_groups)
        ]


def full_attention_config(cache_config) -> KVCacheConfig:
    """把一个旧的单组 `CacheConfig` 包成 `KVCacheConfig`（测试/旧路径用）。

    这个 helper **不知道模型的 head 形状**，只用于"只关心块分配"的路径；生产 Runner 必须
    自己按 Attention 层构造真实 spec，不能拿 `head_size=1` 去建物理缓存。
    """
    spec = FullAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    return KVCacheConfig(
        num_gpu_blocks=cache_config.num_gpu_blocks,
        kv_cache_groups=[KVCacheGroup(spec, ["__default__"])],
    )


def make_packed_kv_cache_view(backing: torch.Tensor,
                              kv_cache_spec: KVCacheSpec,
                              *,
                              offset: int,
                              block_stride: int,
                              num_blocks: int) -> torch.Tensor:
    """把 packed backing 的一段切成层 cache 的 `[2, B, block, kvh, head]` view。

    布局与上游 `_reshape_attention_kv_cache` 一致：一个 block 行里 K 页和 V 页相邻，
    下一块从 `block_stride` 开始；`offset` 是这一层在该行内的起始字节。
    """
    if backing.dtype != torch.int8:
        raise TypeError(f"packed backing 必须是 int8，收到 {backing.dtype}")
    if not backing.is_contiguous() or backing.ndim != 1:
        raise ValueError("packed backing 必须是连续的一维 int8 tensor")
    if offset < 0 or block_stride <= 0:
        raise ValueError("offset/block_stride 非法")
    itemsize = kv_cache_spec.dtype.itemsize
    if offset % itemsize or block_stride % itemsize:
        raise ValueError(
            f"packed offset={offset}/block_stride={block_stride} 必须是 dtype "
            f"itemsize={itemsize} 的整数倍")
    page_elems = (2 * kv_cache_spec.block_size
                  * kv_cache_spec.num_kv_heads * kv_cache_spec.head_size)
    if offset + page_elems * itemsize > block_stride:
        raise ValueError(
            f"层 page {page_elems * itemsize} B 超出 block_stride={block_stride} B")
    if block_stride * num_blocks > backing.numel():
        raise ValueError(
            f"packed backing 只有 {backing.numel()} B，需要 "
            f"{block_stride * num_blocks} B")
    shape = (2, num_blocks, kv_cache_spec.block_size,
             kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
    strides = (
        page_elems // 2,                 # K -> V 在同一 block 内相邻
        block_stride // itemsize,        # 下一块
        kv_cache_spec.num_kv_heads * kv_cache_spec.head_size,
        kv_cache_spec.head_size,
        1,
    )
    return torch.as_strided(
        backing.view(kv_cache_spec.dtype), shape, strides,
        storage_offset=offset // itemsize)
