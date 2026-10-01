"""`BlockPool`：物理块的**池子**——谁空闲、谁被引用、谁的 hash 能复用（对应 vLLM
`v1/core/block_pool.py` 的子集）。

它不认识请求、不认识采样、不认识优先级，只维护三样东西：

    blocks                        所有物理块（编号 0..num_gpu_blocks-1）
    free_block_queue              `ref_cnt == 0` 的块，按淘汰顺序排好
    cached_block_hash_to_block    块 hash → 能复用的物理块

四个动作构成闭环：

    分配 get_new_blocks(n)      从队头取，**顺手清掉它身上的 hash 登记**（它马上会被覆写）
    命中 touch(blocks)          ref_cnt += 1；0→1 时要从空闲队列里**摘出来**（不能再被分出去）
    释放 free_blocks(ordered)   ref_cnt -= 1；到 0 才放回队列（非缓存块进队头、缓存块进队尾）
    发布 cache_full_blocks(...) 给完整块**打上 hash 并登记**，此后它就能被别人命中

**顺序不能反**：先"清登记"再分配、先"摘出队列"再引用、先"减引用"再放回。任何一步写反，
都会出现"同一块同时被两条请求持有"或"命中到已经被覆写的块"。

**与 vLLM 的两处差异**：

1. 没有 `null_block`（vLLM 从空闲队列拿走一个块当占位，因此它实际可用 `num_gpu_blocks - 1`
   个）。本关没有滑窗/CoW，`num_gpu_blocks` 个块全部可用——**测试不能默认 vLLM 的容量口径**。
2. 没有 KV 事件（`BlockStored`/`BlockRemoved`）与 metrics 收集器，那是给外部网关/监控用的。
"""

from .kv_cache_utils import (BlockHashToBlockMap, FreeKVCacheBlockQueue, KVCacheBlock,
                             make_block_hash_with_group_id)


class BlockPool:
    def __init__(self, num_gpu_blocks: int, enable_caching: bool) -> None:
        if num_gpu_blocks <= 0:
            raise ValueError(f"num_gpu_blocks 必须为正，收到 {num_gpu_blocks}")
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.blocks = [KVCacheBlock(block_id) for block_id in range(num_gpu_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block = BlockHashToBlockMap()

    # -------- 观察 --------

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def num_cached_blocks(self) -> int:
        """索引里登记着的物理块数（可能大于"能命中的前缀长度"，因为同 hash 可以有多块）。"""
        return self.cached_block_hash_to_block.num_entries()

    def get_cached_block(self, block_hash: bytes, kv_cache_group_id: int = 0):
        """按 hash 找一个可复用的块；没有就 None。**本关只有一个 KV group**，所以 group id
        由调用方固定传 0（接口保留 group 参数，是为了键的形状与 vLLM 一致）。"""
        return self.cached_block_hash_to_block.get_one_block(
            make_block_hash_with_group_id(block_hash, kv_cache_group_id))

    # -------- 分配 --------

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """从队头取 `num_blocks` 个块。取不到就直接抛（容量检查是调用方的事，见
        `FullAttentionManager.get_num_blocks_to_allocate`）。"""
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"要 {num_blocks} 个块，池子里只有 {self.get_num_free_blocks()} 个")
        new_blocks = self.free_block_queue.popleft_n(num_blocks)
        for block in new_blocks:
            # 这一块马上要被新内容覆写：先把它从 hash 索引里摘掉，否则别人会命中一段
            # 已经不成立的 KV（"逐出不误删同 hash 的其他块"就落在这里）
            self._maybe_evict_cached_block(block)
            if block.ref_cnt != 0:
                raise RuntimeError(f"从空闲队列里取到了 ref_cnt={block.ref_cnt} 的块 {block!r}")
            block.ref_cnt += 1
        return new_blocks

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """清掉这个块自己的 hash 登记（不动同 hash 下别的块）。"""
        block_hash = block.block_hash
        if block_hash is None:
            return False
        popped = self.cached_block_hash_to_block.pop(block_hash, block.block_id)
        if popped is not None:
            block.reset_hash()
        return popped is not None

    # -------- 命中 --------

    def touch(self, blocks) -> None:
        """被别的请求命中：引用计数 +1。0→1 时必须从空闲队列摘出来——否则它既在"可分配"
        名单里又被这条请求用着，下一轮就会被分给别人。"""
        for block in blocks:
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    # -------- 释放 --------

    def free_blocks(self, ordered_blocks) -> None:
        """释放一组块。**调用方要按淘汰优先级传**（队尾块先传，见 `pop_blocks_for_free`）。

        到 0 之后放回队列的位置决定了下次被淘汰的早晚：

            没有 hash（或没开缓存）→ 队头（下次分配优先复用这一块，GPU 局部性好）
            带 hash                → 队尾（越晚淘汰，越可能被命中）
        """
        to_reuse_first: list[KVCacheBlock] = []
        to_evict_last: list[KVCacheBlock] = []
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt < 0:
                raise RuntimeError(f"释放了没被引用的块 {block!r}：引用计数记账错了")
            if block.ref_cnt > 0:
                continue
            if block.block_hash is None or not self.enable_caching:
                to_reuse_first.append(block)
            else:
                to_evict_last.append(block)
        self.free_block_queue.prepend_n(to_reuse_first)
        self.free_block_queue.append_n(to_evict_last)

    # -------- 发布 --------

    def cache_full_blocks(self, request, blocks, num_cached_blocks: int, num_full_blocks: int,
                          block_size: int, kv_cache_group_id: int = 0) -> None:
        """把 `blocks[num_cached_blocks:num_full_blocks]` 这批**完整块**打上 hash 并登记。

        只处理"新变成完整"的那些块：`num_cached_blocks` 之前的已经登记过（可能是自己发布的，
        也可能是命中别人的），重复登记会让 `set_block_hash` 直接报错——那道断言就是在守
        "一个块只有一个 hash"这条不变量。
        """
        if not self.enable_caching:
            return
        if num_cached_blocks >= num_full_blocks:
            return
        hashes = request.block_hashes
        if len(hashes) < num_full_blocks:
            raise RuntimeError(
                f"{request.request_id!r} 只有 {len(hashes)} 个块 hash，却要发布前 "
                f"{num_full_blocks} 个完整块：hash 链没跟上 token 历史")
        for index in range(num_cached_blocks, num_full_blocks):
            block = blocks[index]
            key = make_block_hash_with_group_id(hashes[index], kv_cache_group_id)
            # 命中的块自己已经带 hash（且已在索引里），不改也不重复登记
            if block.block_hash is None:
                block.set_block_hash(key)
            self.cached_block_hash_to_block.insert(key, block)

    # -------- 管理 --------

    def reset_prefix_cache(self) -> bool:
        """清空整个前缀缓存（所有块回到"没有 hash"的状态）。返回是否有块被清。

        vLLM 用它做"切换模型/权重更新后必须让缓存失效"的操作；本关只在测试与 trace 里用。
        """
        removed = 0
        for block in self.free_block_queue.get_all_free_blocks():
            if self._maybe_evict_cached_block(block):
                removed += 1
        for block in self.blocks:
            if block.ref_cnt > 0 and block.block_hash is not None:
                popped = self.cached_block_hash_to_block.pop(block.block_hash, block.block_id)
                if popped is not None:
                    block.reset_hash()
                    removed += 1
        return removed > 0
