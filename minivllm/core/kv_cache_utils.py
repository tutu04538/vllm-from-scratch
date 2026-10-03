"""KV 块的**元数据**与空闲队列、块 hash（对应 vLLM `v1/core/kv_cache_utils.py` 的子集）。

这一层完全不认识 torch、不认识请求调度，只回答三件事：

    KVCacheBlock            一个物理块的元数据：编号、引用计数、块 hash、空闲链表的指针
    FreeKVCacheBlockQueue   `ref_cnt == 0` 的块的**双向链表**，支持 O(1) 摘除/头取/尾插
    BlockHashToBlockMap     hash → 块（同 hash 允许多个物理块，不能覆盖登记）

**为什么空闲队列要自己写双向链表**（vLLM 的注释也这么解释）：`deque` 取头/尾插是 O(1)，
但"按块号摘除中间某个块"是 O(n)——而 `touch()`（命中共享块时把它从空闲队列里摘出来）
正好就是这个操作。双向链表让摘除 O(1)，代价是自己维护 `prev/next`。

**引用计数的含义**：`ref_cnt` 数的是"有多少条**活请求**在用这个块"。它到 0 不代表内容没用
——如果这个块带着块 hash（完整块、已登记），它仍然是**可复用的缓存**，只是回到了空闲队列
等着被淘汰。这条区别是 prefix cache 的全部：0 引用 + 有 hash = 缓存候选。

**本关与 vLLM 的差异**（写清楚，不让测试默认"vLLM 的所有块都可用"）：

- 不保留 `null_block`。vLLM 会从空闲队列拿走一个块当占位（滑窗/CoW 用），于是它实际可用
  的块是 `num_gpu_blocks - 1`；本关没有滑窗也没有 CoW，`num_gpu_blocks` 个块**全部可用**。
- 不实现 partial-tail / COW 元数据（`block_hash_num_tokens` 之外的局部 hash、fine-grained
  hash），本关只做**完整块**路径：一个块要么没有 hash，要么是"整块算完且已发布"。
"""

import hashlib
import os
from dataclasses import dataclass

# 块 hash 是**字节串**（不是 Python 的 int/str）：它要能当持久化键用，不能依赖
# `hash()` 那种每进程随机、跨进程不稳定的值。
BlockHash = bytes

# 每个 hash 键都会带上 group id（本关恒为 0）。带上是为形状与 vLLM 一致：
# 多 KV group 时"同一段 token 在不同 group 的块"必须是不同的键，否则会互相命中。
_SUFFIX_BYTES = 4


def make_block_hash_with_group_id(block_hash: BlockHash, group_id: int) -> bytes:
    return block_hash + group_id.to_bytes(_SUFFIX_BYTES, "big")


def get_group_id(key: bytes) -> int:
    return int.from_bytes(key[-_SUFFIX_BYTES:], "big")


def get_block_hash(key: bytes) -> BlockHash:
    return key[:-_SUFFIX_BYTES]


# 链式 hash 的起点。vLLM 用一个**随机**值，这样不同进程算出的 hash 天然不互通
# （除非显式共享种子）——避免"两个不同模型/不同配置的进程碰巧互认对方的缓存"。
# 本关照做，同时提供 `init_none_hash()` 让测试把种子钉死（否则同一个进程内两次运行
# 也可能不同，断言就没法写）。
NONE_HASH: BlockHash = os.urandom(32)


def init_none_hash(seed: int | str | None = None) -> BlockHash:
    """把链式 hash 的起点设成确定值（测试用）；`None` 表示重新随机。"""
    global NONE_HASH
    NONE_HASH = (os.urandom(32) if seed is None
                 else hashlib.sha256(str(seed).encode()).digest())
    return NONE_HASH


def hash_block_tokens(parent_block_hash: BlockHash | None, token_ids, extra_keys=()) -> BlockHash:
    """`H_i = sha256(H_{i-1} || tokens[i*B:(i+1)*B] || extra_keys)`。

    三个要点：

    1. **链式**：带上父 hash，所以"前缀相同但更早的块不同"不会被误判成命中。反过来，只要
       第一个块不同，后面所有块 hash 都不同——这也是"缺一个块就意味着后面全部 miss"的原因。
    2. **编码必须稳定**：这里自己拼字节（长度前缀 + 逗号分隔），不用 `pickle`/`hash()`。
       `hash()` 每进程随机，`pickle` 依赖类型定义，都不能当持久键。
    3. `extra_keys` 至少包含 `cache_salt`（多租户隔离用）：同一段 token、不同 salt 必须
       算出不同 hash，否则 A 租户能命中 B 租户的缓存。
    """
    parent = parent_block_hash if parent_block_hash else NONE_HASH
    parts = [b"v1", parent]
    for token in token_ids:
        parts.append(str(int(token)).encode())
        parts.append(b",")
    parts.append(b"|")
    for key in extra_keys:
        parts.append(str(key).encode())
        parts.append(b",")
    return hashlib.sha256(b"".join(parts)).digest()


class BlockHasher:
    """给 `Request` 用的**增量**块 hash 计算器。

    `Request` 每次追加 token 都会调用它（`update_block_hashes()`），它只补算**新凑满的完整块**：
    第 i 个块的 hash 依赖第 i-1 个，所以顺序不能乱，但也不必每次从头算。

    "没凑满的尾巴不算 hash"这条很重要：半个块的 KV 随时可能被后续 token 覆盖，
    它进了索引就会让别人命中一段**还不成立**的前缀（197 §4 第二步）。
    """

    def __init__(self, block_size: int, extra_keys=()) -> None:
        self.block_size = block_size
        self.extra_keys = tuple(extra_keys)

    def __call__(self, request) -> list[BlockHash]:
        # 链式依赖必须用**完整的**链来找父 hash：本次调用里算出来的前几个块也是后面块的父，
        # 只读 `request.block_hashes` 会拿到还没写回去的旧列表（这里踩过一次）
        chain = list(request.block_hashes)
        num_full_blocks = len(request.all_token_ids) // self.block_size
        hashes = []
        for index in range(len(chain), num_full_blocks):
            start = index * self.block_size
            chain.append(hash_block_tokens(chain[index - 1] if index else None,
                                           request.all_token_ids[start:start + self.block_size],
                                           self.extra_keys))
            hashes.append(chain[-1])
        return hashes

    def __repr__(self) -> str:
        return f"BlockHasher(block_size={self.block_size}, extra_keys={self.extra_keys})"


@dataclass(eq=False)
class KVCacheBlock:
    """一个物理块的元数据。

    `eq=False`：判等用**身份**。块之间有 `prev/next` 指针，按字段递归比较会顺着空闲链表
    走一遍（还容易在断言里写出"看起来相等其实不是同一个块"的误判）。要比内容就比 `block_id`。
    """

    block_id: int
    ref_cnt: int = 0
    # 只有"完整块且已发布"才有 hash；分配出去时会被清掉（见 BlockPool._maybe_evict_cached_block）
    block_hash: bytes | None = None
    # 由 FreeKVCacheBlockQueue 维护，别的地方不要动
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    def set_block_hash(self, block_hash: bytes) -> None:
        if self.block_hash is not None:
            raise RuntimeError(f"块 {self.block_id} 已经有 hash 了，不该被覆盖登记")
        self.block_hash = block_hash

    def reset_hash(self) -> None:
        self.block_hash = None

    @property
    def is_free(self) -> bool:
        return self.ref_cnt == 0 and (self.prev_free_block is not None
                                      or self.next_free_block is not None)

    def __repr__(self) -> str:
        # 邻居只打编号：直接 repr 块对象会顺着链表递归下去
        prev_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_id = self.next_free_block.block_id if self.next_free_block else None
        return (f"KVCacheBlock(id={self.block_id}, ref={self.ref_cnt}, "
                f"hash={'yes' if self.block_hash else 'no'}, prev={prev_id}, next={next_id})")


class FreeKVCacheBlockQueue:
    """`ref_cnt == 0` 的块组成的双向链表（用假头/假尾省掉边界分支）。

    队头 = 最先被淘汰的块。`BlockPool` 在释放时决定把块放到队头还是队尾：

        非缓存块（没有 hash）→ **队头**（下次分配优先复用同一块，GPU 局部性好）
        带 hash 的块         → **队尾**（越晚被淘汰，命中机会越多）

    这样"释放"这个动作就同时完成了 LRU 排序，不需要额外的访问计数。
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)
        for index in range(self.num_free_blocks):
            if index > 0:
                blocks[index].prev_free_block = blocks[index - 1]
            if index < self.num_free_blocks - 1:
                blocks[index].next_free_block = blocks[index + 1]
        self.fake_head = KVCacheBlock(block_id=-1)
        self.fake_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks:
            self.fake_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_head
            self.fake_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_tail
        else:
            self.fake_head.next_free_block = self.fake_tail
            self.fake_tail.prev_free_block = self.fake_head

    # -------- 取 --------

    def popleft(self) -> KVCacheBlock:
        if self.num_free_blocks == 0:
            raise ValueError("没有空闲块了（调用方应该先查 get_num_free_blocks）")
        return self.popleft_n(1)[0]

    def popleft_n(self, count: int) -> list[KVCacheBlock]:
        if count > self.num_free_blocks:
            raise ValueError(f"要 {count} 个空闲块，只有 {self.num_free_blocks} 个")
        popped: list[KVCacheBlock] = []
        cursor = self.fake_head.next_free_block
        for _ in range(count):
            popped.append(cursor)
            cursor = cursor.next_free_block
        # 重新接上假头与剩下的第一个块
        self.fake_head.next_free_block = cursor
        cursor.prev_free_block = self.fake_head
        for block in popped:
            block.prev_free_block = None
            block.next_free_block = None
        self.num_free_blocks -= count
        return popped

    # -------- 摘除（O(1)，这是自己写链表的唯一理由）--------

    def remove(self, block: KVCacheBlock) -> None:
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"{block!r} 不在空闲队列里，不能摘除")
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

    # -------- 放回 --------

    def append(self, block: KVCacheBlock) -> None:
        """放到队尾（最后被淘汰）。"""
        self.append_n([block])

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        """放到队头（最先被淘汰）——非缓存块的复用路径。"""
        if not blocks:
            return
        first = self.fake_head.next_free_block
        prev_block = self.fake_head
        for block in blocks:
            block.prev_free_block = prev_block
            prev_block.next_free_block = block
            prev_block = block
        prev_block.next_free_block = first
        first.prev_free_block = prev_block
        self.num_free_blocks += len(blocks)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """放到队尾——带 hash 的块走这里，形成 LRU 淘汰顺序。"""
        if not blocks:
            return
        last = self.fake_tail.prev_free_block
        for block in blocks:
            block.prev_free_block = last
            last.next_free_block = block
            last = block
        last.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = last
        self.num_free_blocks += len(blocks)

    # -------- 观察 --------

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """从队头到队尾列出所有空闲块（测试与 trace 用）。"""
        blocks = []
        cursor = self.fake_head.next_free_block
        while cursor is not self.fake_tail:
            blocks.append(cursor)
            cursor = cursor.next_free_block
        if len(blocks) != self.num_free_blocks:
            raise RuntimeError(
                f"空闲队列自相矛盾：链表里有 {len(blocks)} 个，计数器说 {self.num_free_blocks} 个"
                f"（有块被摘除/插入却没维护计数器）")
        return blocks

    def __len__(self) -> int:
        return self.num_free_blocks


class BlockHashToBlockMap:
    """`hash → 块` 的索引。**同一个 hash 允许多个物理块**（vLLM 的注释解释了为什么不去重）：

    两个请求可能各自算出了内容相同的块（比如同一个 prompt 分两批到达且都没命中），它们
    各自持有自己的物理块、各自有相同的 hash。此时索引里就该有两个条目；分配块时也**只能
    移除自己要淘汰的那一个**，不能把同 hash 的另一个一起删掉（197 §5 明确要求）。

    所以内部值是"块或 `{block_id: 块}`"，与 vLLM 同形（用一个 dict 省掉装箱开销）。
    """

    def __init__(self) -> None:
        self._cache: dict[bytes, KVCacheBlock | dict[int, KVCacheBlock]] = {}

    def get_one_block(self, key: bytes) -> KVCacheBlock | None:
        blocks = self._cache.get(key)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            return blocks
        return next(iter(blocks.values()))

    def contain(self, key: bytes, block_id: int) -> bool:
        blocks = self._cache.get(key)
        if blocks is None:
            return False
        if isinstance(blocks, KVCacheBlock):
            return blocks.block_id == block_id
        return block_id in blocks

    def insert(self, key: bytes, block: KVCacheBlock) -> None:
        blocks = self._cache.get(key)
        if blocks is None:
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        else:
            blocks[block.block_id] = block

    def pop(self, key: bytes, block_id: int) -> KVCacheBlock | None:
        """只弹出**指定的那个物理块**；同 hash 下别的块原样留在索引里。"""
        blocks = self._cache.pop(key, None)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            self._cache[key] = blocks            # 不是它：放回去
            return None
        block = blocks.pop(block_id, None)
        if blocks:
            self._cache[key] = blocks
        return block

    def keys_for_block(self, block: KVCacheBlock) -> list[bytes]:
        """这个块登记在哪些 hash 下（本关一个块只有一个 hash，但接口按"可能多个"写）。"""
        if block.block_hash is None:
            return []
        return [block.block_hash] if self.contain(block.block_hash, block.block_id) else []

    def __len__(self) -> int:
        return len(self._cache)

    def num_entries(self) -> int:
        """索引里的**块**总数（同 hash 的多个块分别计数）。"""
        total = 0
        for blocks in self._cache.values():
            total += 1 if isinstance(blocks, KVCacheBlock) else len(blocks)
        return total
