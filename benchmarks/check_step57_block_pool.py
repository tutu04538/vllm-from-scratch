"""57C 验收（对应需求里的 `test_block_pool.py`）：物理块池的不变量。

这一层不认识 torch，也不认识 Scheduler——**纯 CPU 的块账本**，所以可以逐条把不变量钉死：

  1. 空闲队列：`ref_cnt > 0` 的块**不在**队列里；`ref_cnt == 0` 的可用块在队列里**只出现一次**；
     链表与计数器自洽（`get_all_free_blocks()` 自己会查）；
  2. 分配/释放/引用：取队头、`ref_cnt` 加一；`touch` 把 0 引用的块从队列摘出；释放到 0 才回队列；
     非缓存块进**队头**（复用优先）、带 hash 的块进**队尾**（LRU 淘汰）；
  3. 同 hash 多块：索引**不能覆盖**登记；分配其中一个只清它自己，另一个照旧可命中；
  4. 发布：只发布新增的完整块；重复发布无害；块已有 hash 时覆盖登记要报错；
  5. 失败原子性：容量不足时报错且**状态一个字节不变**（前后快照对比）；
  6. 容量口径：本关没有 null block，`num_gpu_blocks` 个块**全部可用**（与 vLLM 差一个）。
"""

import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step57.core.block_pool import BlockPool
from step57.core.kv_cache_utils import (BlockHashToBlockMap, FreeKVCacheBlockQueue,
                                        KVCacheBlock, get_block_hash, hash_block_tokens)

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def first_line(error):
    """报错详情只打第一行。**注意别在 detail 里直接写 `error.splitlines()[0]`**：
    Python 会先求值 detail，`error is None`（本该 FAIL 的那次）会先炸在 IndexError 上。"""
    return error.splitlines()[0] if error else "没有报错"


def snapshot(pool):
    """池子的完整状态指纹：每个块的引用计数、空闲队列顺序、索引里登记了什么。"""
    return {
        "ref": tuple(block.ref_cnt for block in pool.blocks),
        "free_order": tuple(block.block_id for block in pool.free_block_queue.get_all_free_blocks()),
        "index": sorted((key, block.block_id)
                        for key, blocks in pool.cached_block_hash_to_block._cache.items()
                        for block in ([blocks] if isinstance(blocks, KVCacheBlock)
                                      else blocks.values())),
    }


def hashes(count, salt=None):
    """造一串链式 hash（第 i 个依赖第 i-1 个），模拟一段 token 的块链。"""
    chain, parent = [], None
    for index in range(count):
        parent = hash_block_tokens(parent, [index] * 4, (salt,))
        chain.append(parent)
    return chain


class FakeRequest:
    """`cache_full_blocks` 只用到 request 的两个字段，不必造一个真 Request。"""

    def __init__(self, request_id, block_hashes):
        self.request_id = request_id
        self.block_hashes = block_hashes


# ------------------------------------------------ 1. 空闲队列的不变量

pool = BlockPool(num_gpu_blocks=6, enable_caching=True)
check("1. 初始：全部 6 个块都在空闲队列、引用为 0、没有 hash（本关没有 null block）",
      pool.get_num_free_blocks() == 6 and pool.num_cached_blocks() == 0
      and all(block.ref_cnt == 0 and block.block_hash is None for block in pool.blocks),
      f"空闲 {pool.get_num_free_blocks()} / 总 {pool.num_gpu_blocks}")

check("1. 队列自洽：链表的块数 == 计数器（get_all_free_blocks 内部会查）",
      [block.block_id for block in pool.free_block_queue.get_all_free_blocks()] == list(range(6)),
      f"队列顺序 {[b.block_id for b in pool.free_block_queue.get_all_free_blocks()]}")

try:
    BlockPool(num_gpu_blocks=1, enable_caching=True).free_block_queue.remove(KVCacheBlock(99))
    error = None
except RuntimeError as exc:
    error = str(exc)
check("1. 摘除一个不在队列里的块 → 报错（不是静默改坏链表）", error is not None and "不在空闲队列" in error,
      first_line(error))

queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(4)])
check("1. 队列初始顺序是块号顺序，popleft 取队头",
      [queue.popleft().block_id for _ in range(2)] == [0, 1] and len(queue) == 2)
queue.prepend_n([KVCacheBlock(7)])
check("1. prepend 之后新块在队头（非缓存块的复用优先路径）",
      queue.popleft().block_id == 7)

# ------------------------------------------------ 2. 分配、引用与释放

pool = BlockPool(num_gpu_blocks=4, enable_caching=True)
chain = hashes(4)
blocks = pool.get_new_blocks(2)
check("2. 分配：取队头、引用计数变 1、从空闲队列消失",
      [block.block_id for block in blocks] == [0, 1]
      and all(block.ref_cnt == 1 for block in blocks)
      and pool.get_num_free_blocks() == 2,
      f"空闲 {pool.get_num_free_blocks()}")

pool.touch(blocks)                                   # 模拟"被另一个请求命中"
check("2. touch：引用计数 +1（同一个块被两条请求共享）",
      [block.ref_cnt for block in blocks] == [2, 2])

pool.free_blocks(reversed(blocks))                   # 一个请求结束
check("2. 释放一次：引用计数减 1，**还没到 0** 的块不回到空闲队列",
      [block.ref_cnt for block in blocks] == [1, 1] and pool.get_num_free_blocks() == 2,
      f"空闲 {pool.get_num_free_blocks()}")

pool.free_blocks(reversed(blocks))                   # 另一个请求也结束
check("2. 释放到 0：回到空闲队列（没有 hash → 进**队头**，下次分配优先复用）",
      [block.ref_cnt for block in blocks] == [0, 0] and pool.get_num_free_blocks() == 4
      and [block.block_id for block in pool.free_block_queue.get_all_free_blocks()][:2] == [1, 0],
      f"队列头 {[b.block_id for b in pool.free_block_queue.get_all_free_blocks()][:2]}")

pool = BlockPool(num_gpu_blocks=2, enable_caching=True)
held = pool.get_new_blocks(2)                       # 先分配（引用计数 1），发布，再释放
pool.cache_full_blocks(FakeRequest("r1", chain), held, 0, 2, block_size=4)
cached_block = held[0]
pool.free_blocks(list(reversed(held)))              # 逆序释放（尾部先淘汰）
order = [block.block_id for block in pool.free_block_queue.get_all_free_blocks()]
check("2. 带 hash 的块释放后进**队尾**（LRU：越晚淘汰），队头留给非缓存块复用",
      order == [1, 0], f"队列顺序 {order}（0 号带 hash，排在最后）")
check("2. 释放不销毁内容：带 hash 的块仍然可以被命中",
      pool.get_cached_block(chain[0]) is cached_block)

try:
    pool.get_new_blocks(3)
    error = None
except ValueError as exc:
    error = str(exc)
check("2. 分配超过空闲数 → 报错（容量检查是调用方的事，池子只拒绝）",
      error is not None and "池子里只有" in error, first_line(error))

# ------------------------------------------------ 3. 同 hash 多个物理块

pool = BlockPool(num_gpu_blocks=4, enable_caching=True)
first, second = pool.get_new_blocks(2)
pool.cache_full_blocks(FakeRequest("r1", chain), [first, second], 0, 1, block_size=4)
pool.cache_full_blocks(FakeRequest("r2", chain), [second, first], 0, 1, block_size=4)
# 上面第二行把 second 也登记到同一个 hash 下（模拟"两条请求各自算出了同样的内容"）
check("3. 同 hash 允许多个物理块：索引里是两个条目（不能覆盖登记）",
      pool.cached_block_hash_to_block.num_entries() == 2
      and isinstance(pool.cached_block_hash_to_block._cache[first.block_hash], dict),
      f"条目数 {pool.cached_block_hash_to_block.num_entries()}")

# 释放这两块（带 hash → 进空闲队列**队尾**），再一直分配，直到队尾那个被取走
pool.free_blocks(list(reversed([first, second])))
evicted = pool.get_new_blocks(3)   # 2、3 号无 hash，第三个才是队尾那个带 hash 的
cached_evicted = [block for block in evicted if block.block_hash is None and block.ref_cnt == 1]
check("3. 分配一个块只清**它自己**的登记：同 hash 下别的块还能命中",
      pool.cached_block_hash_to_block.num_entries() == 1
      and pool.get_cached_block(chain[0]) is not None
      and [block.block_id for block in evicted][-1] in (first.block_id, second.block_id),
      f"队列头取走 {[b.block_id for b in evicted]}、剩余条目 "
      f"{pool.cached_block_hash_to_block.num_entries()}")

index = BlockHashToBlockMap()
a, b = KVCacheBlock(0), KVCacheBlock(1)
a.set_block_hash(b"k")
b.set_block_hash(b"k")
index.insert(b"k", a)
index.insert(b"k", b)
check("3. pop 指定块：拿对了才删，不是同 hash 的一起删",
      index.pop(b"k", 99) is None and index.num_entries() == 2
      and index.pop(b"k", b.block_id) is b and index.num_entries() == 1
      and index.get_one_block(b"k") is a)

try:
    a.set_block_hash(b"k2")
    error = None
except RuntimeError as exc:
    error = str(exc)
check("3. 给已有 hash 的块再打一个 hash → 报错（一个块只有一个身份）",
      error is not None and "已经有 hash" in error, first_line(error))

# ------------------------------------------------ 4. 发布

pool = BlockPool(num_gpu_blocks=4, enable_caching=True)
blocks = pool.get_new_blocks(3)
request = FakeRequest("r1", hashes(3))
pool.cache_full_blocks(request, blocks, 0, 2, block_size=4)
check("4. 发布只覆盖 [num_cached, num_full) 这一段（前两个块）",
      get_block_hash(blocks[0].block_hash) == request.block_hashes[0]
      and get_block_hash(blocks[1].block_hash) == request.block_hashes[1]
      and blocks[2].block_hash is None
      and pool.num_cached_blocks() == 2)

pool.cache_full_blocks(request, blocks, 0, 2, block_size=4)
check("4. 重复发布同一段：不报错、也不重复登记（幂等）", pool.num_cached_blocks() == 2)

pool.cache_full_blocks(request, blocks, 2, 3, block_size=4)
check("4. 继续发布第三个块（增量路径）",
      get_block_hash(blocks[2].block_hash) == request.block_hashes[2]
      and pool.num_cached_blocks() == 3)

pool = BlockPool(num_gpu_blocks=2, enable_caching=False)
blocks = pool.get_new_blocks(2)
pool.cache_full_blocks(FakeRequest("r1", hashes(2)), blocks, 0, 2, block_size=4)
check("4. 关掉缓存时发布是空操作（同一套代码的另一条分支）",
      pool.num_cached_blocks() == 0 and all(b.block_hash is None for b in blocks))

pool = BlockPool(num_gpu_blocks=3, enable_caching=True)
held = pool.get_new_blocks(2)
pool.cache_full_blocks(FakeRequest("r1", hashes(1)), held, 0, 1, block_size=4)
try:
    pool.cache_full_blocks(FakeRequest("r1", hashes(1)), held, 1, 2, block_size=4)
    error = None
except RuntimeError as exc:
    error = str(exc)
check("4. hash 链跟不上 token 历史 → 报错（发布比 hash 还多）",
      error is not None and "hash 链没跟上" in error, first_line(error))

# ------------------------------------------------ 5. 失败原子性

pool = BlockPool(num_gpu_blocks=3, enable_caching=True)
held = pool.get_new_blocks(2)
pool.cache_full_blocks(FakeRequest("r1", hashes(2)), held, 0, 2, block_size=4)
before = snapshot(pool)
try:
    pool.get_new_blocks(2)                          # 只剩 1 个空闲块
    error = None
except ValueError as exc:
    error = str(exc)
check("5. 容量不足：报错，且**索引、队列、引用计数一个字节都没变**",
      error is not None and snapshot(pool) == before,
      f"error={bool(error)}、状态一致={snapshot(pool) == before}")

pool = BlockPool(num_gpu_blocks=3, enable_caching=True)
held = pool.get_new_blocks(2)
pool.cache_full_blocks(FakeRequest("r1", hashes(1)), held, 0, 1, block_size=4)
before = snapshot(pool)
try:
    pool.cache_full_blocks(FakeRequest("r1", hashes(1)), held, 1, 2, block_size=4)
    error = None
except RuntimeError as exc:
    error = str(exc)
check("5. 发布时 hash 链长度不够 → 报错，且没有留下半个登记",
      error is not None and snapshot(pool) == before,
      f"error={bool(error)}、状态一致={snapshot(pool) == before}")

# ------------------------------------------------ 6. 容量口径与引用对账

pool = BlockPool(num_gpu_blocks=5, enable_caching=True)
check("6. 本关没有 null block：5 个块全部可用（vLLM 会拿走一个，实际只有 4 个）",
      pool.get_num_free_blocks() == 5 and len(pool.blocks) == 5)

held = pool.get_new_blocks(3)
pool.touch(held)
check("6. 引用对账：活引用总数 == sum(ref_cnt)，空闲数 == 队列长度",
      sum(block.ref_cnt for block in pool.blocks) == 6
      and pool.get_num_free_blocks() == pool.free_block_queue.num_free_blocks == 2,
      f"sum(ref)={sum(b.ref_cnt for b in pool.blocks)}")

pool.free_blocks(reversed(held))
pool.free_blocks(reversed(held))
check("6. 全部释放后：引用归零、全部块回到队列（活引用为 0）",
      sum(block.ref_cnt for block in pool.blocks) == 0 and pool.get_num_free_blocks() == 5)

cached = pool.get_new_blocks(3)
pool.cache_full_blocks(FakeRequest("r1", hashes(3)), cached, 0, 3, block_size=4)
pool.free_blocks(list(reversed(cached)))
check("6. 全员空闲但仍有 3 个块带 hash：这不是泄漏，是**可复用的缓存**",
      pool.num_cached_blocks() == 3 and pool.get_num_free_blocks() == 5
      and sum(block.ref_cnt for block in pool.blocks) == 0)

cleared = pool.reset_prefix_cache()
check("6. reset_prefix_cache：清掉所有登记（块还在，只是不再可复用）",
      cleared and pool.num_cached_blocks() == 0
      and all(block.block_hash is None for block in pool.blocks))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
