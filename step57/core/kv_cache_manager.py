"""KV 控制面：**只管"哪些块归谁"**，不碰模型计算，也不碰 GPU 写入。

对应 vLLM `v1/core/kv_cache_manager.py`（那里还有块 hash、前缀命中、引用计数、事件发布，
本关只留 57A 需要的部分）：

    allocate_slots(request, num_new_tokens) -> KVCacheBlocks | None
        为"本轮要算的 num_new_tokens"申请块；**不够就返回 None**，不做任何半成品改动。
    get_blocks(req_id) -> KVCacheBlocks        当前块表（快照用）
    free(request)                              请求结束/抢占时归还

**57A 的边界**（写清楚，免得被当成已完成）：

- 没有前缀缓存（`enable_prefix_caching` 在 CacheConfig 里被明确拒绝）——块 hash、命中查询、
  引用计数都在 57C；
- 没有抢占的受害者选择——那是 Scheduler 的事，57C 接；
- 分配器是**简单空闲表**：够就发、不够就 None，块按需增长，不回收半块。
  这不是"vLLM 的等价实现"，是本关允许的简化（194 §"57C 以前允许简单测试分配器"）。

一个不变量：**请求的块表长度恒等于 `ceil(num_computed_tokens / block_size)` 的"已分配"版本**
——分配永远按"本轮要算到哪"补齐，绝不预分配超过需要的块。
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheBlocks:
    """一次分配的**结果块**，按 KV group 分组。57A 只有一个 group，所以长度恒为 1。

    注意语义是"新增的块"（vLLM 同名类型也是这个意思）：`allocate_slots()` 返回的是**本次新
    申请到的**块；执行侧对非恢复请求要把它们**追加**到自己的块表后面。没有新增时是空元组，
    `get_block_ids(allow_none=True)` 回 None —— 包里"没有新块"就是这样表达的。

    查询整张块表用 `KVCacheManager.get_blocks(req_id)`，那是另一件事（只用于新请求的首次下
    发）。`get_block_ids()` 返回**新 list**：快照不能与控制面共享可变对象。
    """

    blocks: tuple[list[int], ...]

    def get_block_ids(self, allow_none: bool = False):
        if allow_none and not self.blocks:
            return None
        return tuple(list(group) for group in self.blocks)

    def __len__(self) -> int:
        return sum(len(group) for group in self.blocks)


class KVCacheManager:
    def __init__(self, cache_config) -> None:
        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        # 空闲块用 list 当栈：分配取尾巴、释放放回去。顺序只影响可复现性，不影响正确性；
        # 用栈是为了让"同样的轨迹得到同样的块号"，测试里好断言。
        self._free_blocks: list[int] = list(range(self.num_gpu_blocks))
        self._req_to_blocks: dict[str, list[int]] = {}
        self.num_allocated_blocks = 0

    # -------- 查询 --------

    def get_blocks(self, req_id: str) -> KVCacheBlocks:
        return KVCacheBlocks((list(self._req_to_blocks.get(req_id, ())),))

    def blocks_for_tokens(self, num_tokens: int) -> int:
        """`num_tokens` 需要几个块（向上取整）。"""
        return math.ceil(num_tokens / self.block_size)

    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    # -------- 分配 --------

    def allocate_slots(self, request, num_new_tokens: int) -> KVCacheBlocks | None:
        """给"算到 `num_computed_tokens + num_new_tokens`"申请块。

        失败（块不够）时**什么都不改**——调用方（Scheduler）按"这一轮不调度它"处理。这一点是
        刻意的：留下"分配了一半"的状态，会让调用方很难撤销（vLLM 在抢占路径里也必须撤销干净）。
        """
        if num_new_tokens <= 0:
            raise ValueError(f"num_new_tokens 必须为正，收到 {num_new_tokens}")
        already = self._req_to_blocks.get(request.request_id, [])
        need_total = self.blocks_for_tokens(request.num_computed_tokens + num_new_tokens)
        missing = need_total - len(already)
        if missing <= 0:
            return KVCacheBlocks(())                      # 没有新增块：包里就是 None
        if missing > len(self._free_blocks):
            return None                                   # 不够：保持原样
        new_blocks = [self._free_blocks.pop() for _ in range(missing)]
        self._req_to_blocks[request.request_id] = already + new_blocks
        self.num_allocated_blocks += missing
        return KVCacheBlocks((list(new_blocks),))         # **只返回新增的**

    # -------- 释放 --------

    def free(self, request) -> None:
        """归还这条请求占的块。重复释放是空操作（结束路径可能被调用两次）。"""
        blocks = self._req_to_blocks.pop(request.request_id, None)
        if not blocks:
            return
        self._free_blocks.extend(blocks)
        self.num_allocated_blocks -= len(blocks)
