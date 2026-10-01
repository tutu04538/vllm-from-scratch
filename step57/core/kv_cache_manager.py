"""`KVCacheManager`：**面向请求**的 KV 控制面（对应 vLLM `v1/core/kv_cache_manager.py` 的子集）。

它是 Scheduler 唯一打交道的 KV 接口，做的判断都是"这条请求该不该现在跑"：

    get_computed_blocks(request)      前缀命中：能白拿多少 token 的 KV
    allocate_slots(...)               给"本轮要算的 token"申请槽位；不够就返回 None（不做半成品）
    cache_blocks(request, n)          发布已经算完的完整块（供别的请求命中）
    free(request)                     释放这条请求的块（内容可能仍留在缓存里）
    get_blocks(request_id)            当前块表（打包快照用）

层次（197 §2）：

    KVCacheManager
      └─ UnitaryKVCacheCoordinator
           └─ FullAttentionManager
                └─ BlockPool

**两个不变量**（都被测试盯着）：

1. **容量先核算、再执行不可失败的更新**：`allocate_slots` 先用
   `get_num_blocks_to_allocate` 算出"还差几个块"（含命中块里要被 `touch` 摘出空闲队列的
   那些），不够就直接 `return None`，**一个字节都不改**。失败之后调用方看到的还是干净状态。
2. **物理容量 ≥ 逻辑有效范围**：申请的是"本轮要算到哪"，不是"请求总长"——
   `ceil(num_computed + num_new / block_size)`。尾部多出来的槽位是给下一轮的，
   不影响正确性（attention 只读 `seq_lens` 范围内的 KV）。

**本关的显式教学差异**（197 §4 允许，差异账本里保留）：**发布发生在结果处理之后**，由
Scheduler 在 `update_from_output` 里调 `cache_blocks(request, num_computed_tokens)`，
而不是像 vLLM 那样在 `allocate_slots` 里顺手发布。理由是本关不实现"尚在本轮执行中的块被
复用"——那种提前发布要求"发布的块一定已经写完 KV"，需要与执行顺序严格配合。
代价是放弃了一些更早的命中机会（性能差异，不是正确性差异）。
"""

import math

from .block_pool import BlockPool
from .kv_cache_coordinator import UnitaryKVCacheCoordinator


class KVCacheBlocks:
    """一组（本关：一个 group 的）块。**语义是"这次新增的块"**（与 vLLM 同名类型一致）：

    `allocate_slots()` 返回的是本次**新申请**的块，执行侧对普通续跑要把它们**追加**到自己的
    块表后面；恢复（resumed）请求拿到的是整张新表（走 `get_blocks()`）。

    `get_block_ids()` 产出**新 list of int**——跨执行边界只传编号，且快照不能与控制面共享
    可变对象（把 `KVCacheBlock` 对象直接发出去就等于把池子的内部状态交出去了）。
    """

    def __init__(self, blocks: tuple[list, ...] = ()) -> None:
        self.blocks = blocks

    def get_block_ids(self, allow_none: bool = False):
        """产出 `((块号, ...),)`。`allow_none=True` 时"**一个块都没有**"返回 `None` ——
        协议里"这轮没有新增块"必须是 None 而不是空列表（执行侧据此区分"没动"与"清空"）。

        判空看的是**总块数**而不是"组的个数"：`([],)`（有一个组但组里没块）也算"没有新增块"。
        """
        if allow_none and len(self) == 0:
            return None
        return tuple([block.block_id for block in group] for group in self.blocks)

    def __len__(self) -> int:
        return sum(len(group) for group in self.blocks)

    def __repr__(self) -> str:
        return f"KVCacheBlocks({[len(group) for group in self.blocks]})"


class KVCacheManager:
    def __init__(self, cache_config, max_model_len: int | None = None) -> None:
        """`max_model_len=None` 表示**不做上下文夹取**：只有"越界已由上层保证不会发生"的
        纯分配用例（只测块池行为）才该这么用；生产路径由 EngineCore 传真实值。"""
        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.max_model_len = max_model_len
        self.enable_caching = bool(cache_config.enable_prefix_caching)
        self.block_pool = BlockPool(self.num_gpu_blocks, enable_caching=self.enable_caching)
        self.coordinator = UnitaryKVCacheCoordinator(self.block_pool, self.enable_caching,
                                                     self.block_size)

    # -------- 观察 --------

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """这条请求**当前**的整张块表（打包 `NewRequestData` 用）。"""
        return KVCacheBlocks((list(self.coordinator.get_blocks(request_id)),))

    def num_free_blocks(self) -> int:
        return self.block_pool.get_num_free_blocks()

    def num_cached_blocks(self) -> int:
        return self.block_pool.num_cached_blocks()

    @property
    def num_allocated_blocks(self) -> int:
        """被活请求持有的块数（= 总块数 - 空闲块数）。空闲块里包含"带缓存但没人用"的块。"""
        return self.num_gpu_blocks - self.num_free_blocks()

    def num_common_prefix_blocks(self) -> int:
        """所有活请求共同的前缀块数（trace/调试用；vLLM 用它做 cascade attention 与 P/D 传输）。"""
        groups = [self.coordinator.get_blocks(request_id)
                  for request_id in self.coordinator.manager.req_to_blocks]
        if not groups:
            return 0
        common = 0
        for blocks in zip(*groups):
            if len({block.block_id for block in blocks}) != 1:
                break
            common += 1
        return common

    # -------- 命中 --------

    def get_computed_blocks(self, request):
        """前缀命中查询。返回 `(命中的块, 命中 token 数)`。

        **要在快照之前调用**：命中的 token 会算进这轮的起点（`num_computed_tokens`），
        执行侧因此不会重算它们，而是直接读共享块的 KV。

        命中上限有意压到 `num_tokens - 1`（vLLM 同款）：全部命中就意味着"这段 token 一个都不用
        算"，而**最后一个 token 的 logits 恰恰是下一个 token 的来源**——凭空拿不到。所以至少
        留一个 token 在本轮算。按块粒度，实际会保留一整个块（可能多于一个 token）。
        """
        if not self.enable_caching:
            return KVCacheBlocks(()), 0
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, hit_length = self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length)
        return KVCacheBlocks(tuple(computed_blocks)), hit_length

    # -------- 分配 --------

    def allocate_slots(self, request, num_new_tokens: int, num_new_computed_tokens: int = 0,
                       new_computed_blocks: KVCacheBlocks | None = None) -> KVCacheBlocks | None:
        """给"算到 `num_computed_tokens + num_new_computed_tokens + num_new_tokens`"申请槽位。

        失败（块不够）时**什么都不改**：不 touch、不分配、不改账本。调用方（Scheduler）
        按"这轮排不了它"处理，或者去抢占别的请求再来一遍。
        """
        if num_new_tokens <= 0:
            raise ValueError(f"num_new_tokens 必须为正，收到 {num_new_tokens}")
        computed_blocks = new_computed_blocks.blocks[0] if new_computed_blocks else ()
        num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        # 申请范围**夹到 max_model_len**：多出来的 token 本来就跑不完（Scheduler 裁剪过），
        # 不夹的话会因为"申请超过上下文长度的槽位"而误判容量不足
        num_tokens_need_slot = num_local_computed_tokens + num_new_tokens
        if self.max_model_len is not None:
            num_tokens_need_slot = min(num_tokens_need_slot, self.max_model_len)

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request.request_id, num_tokens_need_slot, computed_blocks)
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None                                     # 原子失败：什么都没改

        if computed_blocks:
            self.coordinator.allocate_new_computed_blocks(
                request.request_id, computed_blocks, num_local_computed_tokens)
        new_blocks = self.coordinator.allocate_new_blocks(request.request_id,
                                                          num_tokens_need_slot)
        return KVCacheBlocks((list(new_blocks),))

    # -------- 发布与释放 --------

    def cache_blocks(self, request, num_computed_tokens: int) -> None:
        """发布"已经确定且算完"的完整块。

        上限是 `floor(min(num_computed_tokens, num_tokens) / block_size)`：
        - `num_computed_tokens` 可能比 `num_tokens` 大（本轮算完了、采样结果还没提交回来），
          夹住就不会把"还不存在的 token"算进块里；
        - 只有完整块会进索引（197 §4）。
        """
        if not self.enable_caching:
            return
        num_tokens = min(num_computed_tokens, request.num_tokens)
        self.coordinator.cache_blocks(request, num_tokens)

    def free(self, request) -> None:
        """释放这条请求占的块。重复释放是空操作（结束路径可能被调用两次）。

        释放**不等于**内容作废：带 hash 的块会回到空闲队列**队尾**，仍可被别人命中，
        直到被分配出去为止。这就是"留下零引用缓存块不算泄漏"（197 §7）。
        """
        self.coordinator.free(request.request_id)

    # -------- 调试 --------

    def block_stats(self) -> dict:
        """给 trace 用的一行摘要。"""
        return {
            "free": self.num_free_blocks(),
            "allocated": self.num_allocated_blocks,
            "cached": self.num_cached_blocks(),
            "reqs": self.coordinator.manager.num_reqs(),
        }
