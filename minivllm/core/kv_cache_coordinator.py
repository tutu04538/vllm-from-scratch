"""`UnitaryKVCacheCoordinator`：**只有一个 KV group** 时的协调者（对应 vLLM
`v1/core/kv_cache_coordinator.py` 的 `UnitaryKVCacheCoordinator`）。

它这一关看着"薄得没必要"——毕竟只有一个 group，每个方法都只是转发。但它不是空壳，它回答的是
一个真实存在的设计问题：**"KV group" 这一层的语义**。

    一个 group = 一组**必须一起命中**的 KV 块。全注意力模型只有一组（每层的 K/V 都是同一套块），
    混合模型（滑窗 + 全注意力、或 Mamba + 注意力）有多组，且"命中长度"要在组之间取**共同**边界。

`UnitaryKVCacheCoordinator` 就是"只有一组"时的退化实现：`self.managers` 长度为 1，
所有方法只跟它打交道。vLLM 里同样的类名、同样的职责（它 `assert len(kv_cache_groups) == 1`）。

**不做的事**（明确写出来，避免"看起来支持混合 KV"）：不构造空的 `HybridKVCacheCoordinator`，
不做组间共同边界计算，不做多层（PP）的额外块（那要求同一个 group 内所有层的块一起分配，
本关所有层共用一张块表，本来就是一致的）。
"""

from .single_type_kv_cache_manager import FullAttentionManager


class UnitaryKVCacheCoordinator:
    def __init__(self, block_pool, enable_caching: bool, block_size: int) -> None:
        self.block_pool = block_pool
        self.block_size = block_size
        self.enable_caching = enable_caching
        # 单 group：列表长度为 1。保留"组"这一层，是为了让调用点的形状与 vLLM 一致
        # （`find_longest_cache_hit` 返回的是"每组一份块"），将来加组时改的是本文件。
        self.kv_cache_group_id = 0
        self.manager = FullAttentionManager(block_pool, enable_caching, block_size,
                                           kv_cache_group_id=self.kv_cache_group_id)

    # -------- 对上层（KVCacheManager）暴露的接口 --------

    def find_longest_cache_hit(self, block_hashes, max_length: int):
        """返回 `([命中块], 命中 token 数)`。单 group，所以外层列表长度为 1。"""
        hit_blocks, hit_length = self.manager.find_longest_cache_hit(block_hashes, max_length)
        return [hit_blocks], hit_length

    def get_num_blocks_to_allocate(self, request_id: str, num_tokens: int,
                                   new_computed_blocks=()) -> int:
        return self.manager.get_num_blocks_to_allocate(request_id, num_tokens,
                                                      new_computed_blocks)

    def allocate_new_computed_blocks(self, request_id: str, new_computed_blocks,
                                     num_local_computed_tokens: int) -> None:
        self.manager.add_local_computed_blocks(request_id, new_computed_blocks,
                                               num_local_computed_tokens)

    def allocate_new_blocks(self, request_id: str, num_tokens: int):
        return self.manager.allocate_new_blocks(request_id, num_tokens)

    def cache_blocks(self, request, num_tokens: int) -> None:
        self.manager.cache_blocks(request, num_tokens)

    def free(self, request_id: str) -> None:
        """释放这条请求的块：`BlockPool.free_blocks` 收的是**逆序**（尾部先淘汰）。"""
        self.block_pool.free_blocks(self.manager.free(request_id))

    def get_blocks(self, request_id: str):
        return self.manager.get_blocks(request_id)
