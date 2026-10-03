"""`FullAttentionManager`：**请求 → 逻辑块**的账本（对应 vLLM
`v1/core/single_type_kv_cache_manager.py` 的单类型子集）。

它站在 `BlockPool` 之上回答"这条请求该拿几个块、拿到哪几个"：

    req_to_blocks      req_id → [KVCacheBlock]（按逻辑顺序，第 i 个 = 该请求第 i 个逻辑块）
    num_cached_block   req_id → 已经"发布进缓存"的完整块数（避免重复登记）

块数怎么算（`get_num_blocks_to_allocate`）——**这是"容量检查"的全部**：

    running 请求（已经在账本里）  max(需要 - 已有, 0)
    首次接纳/恢复                  max(需要 - (命中块 + 已有), 0) + **命中里的零引用块**

最后那一项最容易漏：命中到的块如果 `ref_cnt == 0`（还在空闲队列里等着被淘汰），
`touch()` 会把它从队列里摘出来——**它同时也就不再是"空闲可用"的了**。所以容量检查必须
把"即将被摘出来的块"算进需求里，否则会出现"检查时说够、分配时不够"，也就是 197 §6 说的
"把同一空闲块算两次"。

**本关只实现全注意力**（没有滑窗的 `remove_skipped_blocks`、没有 CoW、没有 fine-grained
hash）。196/197 明确说不要为不存在的混合模型先写抽象：所以这里没有 `SingleTypeKVCacheManager`
基类，只有一个具体类；将来真要加滑窗，把它提成基类即可。
"""

import math

from .kv_cache_utils import KVCacheBlock


class FullAttentionManager:
    def __init__(self, block_pool, enable_caching: bool, block_size: int,
                 kv_cache_group_id: int = 0) -> None:
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self.block_size = block_size
        self.kv_cache_group_id = kv_cache_group_id
        self.req_to_blocks: dict[str, list[KVCacheBlock]] = {}
        self.num_cached_block: dict[str, int] = {}

    # -------- 观察 --------

    def get_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return self.req_to_blocks.get(request_id, [])

    def num_blocks(self, request_id: str) -> int:
        return len(self.req_to_blocks.get(request_id, ()))

    def num_reqs(self) -> int:
        return len(self.req_to_blocks)

    # -------- 容量 --------

    def get_num_blocks_to_allocate(self, request_id: str, num_tokens: int,
                                   new_computed_blocks=()) -> int:
        """为了让这条请求**拥有 `num_tokens` 个 token 的槽位**，还需要几个块。"""
        num_required_blocks = math.ceil(num_tokens / self.block_size)
        num_req_blocks = self.num_blocks(request_id)

        if request_id in self.num_cached_block:
            # 快路径：已经在跑的请求不会再命中新前缀（它的历史就在自己的块里）
            assert not new_computed_blocks, "running 请求不该带命中块"
            return max(num_required_blocks - num_req_blocks, 0)

        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        num_new_blocks = max(num_required_blocks - num_local_computed_blocks, 0)
        # 命中里 ref_cnt == 0 的块会被 touch 摘出空闲队列，必须在检查时算进需求
        num_evictable_blocks = sum(1 for block in new_computed_blocks if block.ref_cnt == 0)
        return num_new_blocks + num_evictable_blocks

    # -------- 分配 --------

    def add_local_computed_blocks(self, request_id: str, new_computed_blocks,
                                  num_local_computed_tokens: int) -> None:
        """把命中的块挂到这条请求上：先 `touch`（引用计数 +1、摘出空闲队列）再登记。

        顺序反过来的话，"引用计数已经加了但块还在空闲队列里"这个瞬间就可能被别的请求分走。
        这些块已经是完整块、也已经在索引里了，所以 `num_cached_block` 直接从命中长度起算，
        `cache_blocks()` 不会重复登记它们。
        """
        req_blocks = self.req_to_blocks.setdefault(request_id, [])
        if req_blocks:
            raise RuntimeError(f"{request_id!r} 已经有块了，命中块只能加在首次接纳/恢复时")
        if not new_computed_blocks:
            return
        if not self.enable_caching:
            raise RuntimeError("没开前缀缓存却收到了命中块")
        self.block_pool.touch(new_computed_blocks)
        req_blocks.extend(new_computed_blocks)
        self.num_cached_block[request_id] = len(req_blocks)

    def allocate_new_blocks(self, request_id: str, num_tokens: int) -> list[KVCacheBlock]:
        """补齐到 `num_tokens` 个槽位。**这一步不该失败**——容量已经在
        `get_num_blocks_to_allocate` 里核算过了（"先核算、再执行不可失败的更新"）。"""
        req_blocks = self.req_to_blocks.setdefault(request_id, [])
        num_required_blocks = math.ceil(num_tokens / self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        if num_new_blocks > self.block_pool.get_num_free_blocks():
            raise RuntimeError(
                f"{request_id!r} 要 {num_new_blocks} 个新块，但池子里只剩 "
                f"{self.block_pool.get_num_free_blocks()} 个：容量检查没有先做（控制面 bug）")
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        return new_blocks

    # -------- 发布 --------

    def cache_blocks(self, request, num_tokens: int) -> None:
        """把已经算完的**完整块**发布进缓存索引（`num_tokens` 由调用方裁剪）。

        只有完整块能发布：半个块的内容随时会被后续 token 覆盖（197 §4）。
        """
        num_full_blocks = num_tokens // self.block_size
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        if num_cached_blocks >= num_full_blocks:
            return
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id)
        self.num_cached_block[request.request_id] = num_full_blocks

    # -------- 释放 --------

    def free(self, request_id: str) -> list[KVCacheBlock]:
        """摘掉这条请求的账本并返回它的块（**逆序**，让尾部先被淘汰）。

        调用方负责把返回值交给 `BlockPool.free_blocks()`。分两步是为了让"撤销计划"
        这类路径能先拿到块再决定要不要还（本关直接还，见 Scheduler 的抢占）。
        """
        req_blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)
        return list(reversed(req_blocks))

    # -------- 命中查询 --------

    def find_longest_cache_hit(self, block_hashes, max_length: int):
        """从第 0 个块开始逐个查缓存，**遇到第一个 miss 就停**。

        为什么可以直接停：块 hash 是链式的（`H_i` 含 `H_{i-1}`），第 i 个块 miss 意味着
        第 i+1 个块的 hash 不可能匹配上任何已有块。这也是"前缀缓存"这个名字的由来——
        命中的一定是一段**前缀**，中间不会断。

        返回 `(命中的块, 命中 token 数)`；本关不返回 vLLM 的第三项 `shared_prefix_boundary`
        （那是稀疏保留的滑窗/混合模型才需要的交界点，差异账本里记着）。
        """
        hit_blocks: list[KVCacheBlock] = []
        for block_hash in block_hashes[:max_length // self.block_size]:
            block = self.block_pool.get_cached_block(block_hash, self.kv_cache_group_id)
            if block is None:
                break
            hit_blocks.append(block)
        return hit_blocks, len(hit_blocks) * self.block_size
