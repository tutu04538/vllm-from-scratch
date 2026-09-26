"""KV 缓存池：分块存储、slot 寻址、前缀缓存与 LRU 淘汰。

这一层管的是「物理块放在哪、谁还在引用它」，不管 attention 怎么算。

**池子不做容量准入**：准入只判「这条请求单独跑装不装得下」（装不下就抛
`InfeasibleRequest`，谁也帮不了它），块在真正排到 token 时才按需补。池子不够时
`ensure_blocks()` 返回 False，由 `Scheduler` 选犧牲者——那是调度决策，池子不认识
优先级，也不该替调度器做决定。
"""

import hashlib
import itertools
import json
import math
from dataclasses import dataclass

import torch

# 请求状态住在 request.py；这里重导出，让 `from step48.cache import SequenceConfig`
# 这类既有导入继续可用，搬文件本身不破坏任何旧使用者。
from .request import CacheConfig, SequenceConfig  # noqa: F401


@dataclass
class AdmissionPlan:
    """准入的只读结论：命中哪些完整块、还需要多少新块。

    改状态的只有 `_commit_admission()`；计划阶段失败时一个字节都不动。
    """
    matched_block_ids: list
    matched_hashes: list
    required_new_blocks: int


@dataclass
class BlockGrowthPlan:
    """本轮补块的只读结论：从可分配链队首取走哪几个物理块。

    不能在计划阶段就取走，否则 `_plan_block_growth()` 失败时会留下副作用。
    """
    new_block_ids: list


class InfeasibleRequest(Exception):
    """请求在任何情况下都不可能完成（例如它自己要的块数超过整个池子）。

    和「暂时不够」不是一回事：暂时不够会返回 False 让它继续排队，这个必须让调用方看见。
    """


def _stable_hash(previous_hash: bytes, block: tuple[int, ...]) -> bytes:
    # token ID 不限于 0~255，先编码成文本，再交给 sha256
    data = json.dumps((previous_hash.hex(), block)).encode("utf-8")
    return hashlib.sha256(data).digest()


class KVCachePool:

    def __init__(self, block_size, num_kv_blocks, num_kv_heads, head_dim, device,
                 enable_prefix_caching=True, num_layers=1, dtype=torch.float32):
        self.block_size = block_size
        self.num_kv_blocks = num_kv_blocks
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        # 每层一份 KV：同一个 token 在每层的 K/V 不同，不能写进同一片缓存。
        # 池用运行精度，否则显存并没有按预期减少
        self.k_cache = torch.zeros(num_layers, num_kv_blocks, block_size, num_kv_heads, head_dim,
                                   device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, num_kv_blocks, block_size, num_kv_heads, head_dim,
                                   device=device, dtype=dtype)
        # 每层的 [块, 块内偏移] 看成一排 token 槽位，与底层存储共享，不是副本
        self.k_flat = self.k_cache.view(num_layers, -1, num_kv_heads, head_dim)
        self.v_flat = self.v_cache.view(num_layers, -1, num_kv_heads, head_dim)
        self.block_usage = [0] * self.num_kv_blocks  # 引用该块的活动请求数
        # 可分配块（block_usage[b] == 0 的全部块）放在**一条**按淘汰优先级排序的
        # 双向链表里，仿本机 vLLM 的 FreeKVCacheBlockQueue：
        #
        #   队首  下一个被取走 / 淘汰的块
        #   无 hash 的「真正空闲」块**前插**（前插 -> 优先复用）
        #   带 hash 的「闲置缓存」块**后插**（后插 -> 最后才淘汰）
        #
        # 分配从队首取；取到带 hash 的块就地清掉 hash 再用（vLLM 的
        # `_maybe_evict_cached_block`）。删除靠块自带的前后指针，O(1)，
        # 不需要额外的位置索引，也不用堆。
        #
        # 链表用两个**哨兵槽位**（下标 num_kv_blocks / +1）避免首尾分支；
        # block_next/prev[b] == -1 表示「不在链表里」。
        #
        # 不变量：set(链表) == {b | block_usage[b] == 0}，且无重复。
        # 见 _allocatable_block_indices() 的说明。
        # 哨兵占用 num_kv_blocks 与 num_kv_blocks+1 两个槽位
        self._SENTINEL_HEAD = self.num_kv_blocks
        self._SENTINEL_TAIL = self.num_kv_blocks + 1
        self.block_next = [-1] * (self.num_kv_blocks + 2)
        self.block_prev = [-1] * (self.num_kv_blocks + 2)
        self.block_next[self._SENTINEL_HEAD] = self._SENTINEL_TAIL
        self.block_prev[self._SENTINEL_TAIL] = self._SENTINEL_HEAD
        self.num_allocatable = 0
        for block_idx in range(self.num_kv_blocks - 1, -1, -1):
            self._list_prepend(block_idx)      # 逆序前插 -> 最终按编号升序
        self.enable_prefix_caching = enable_prefix_caching
        self.hash_to_block = {}  # 前缀 hash -> 该块物理块编号
        self.block_to_hash = {}  # 物理块编号 -> 仍保留它的缓存条目 hash

    # ---- 可分配块的双向链表 ----
    #
    # 在**状态转换点**增量维护，不在热路径重建：
    #
    #   初始化                全部块按编号升序前插成一条链
    #   _commit_block_growth  从队首取走这次真正用掉的块（带 hash 的就地清 hash）
    #   deallocate_block      引用降到 0：无 hash 前插、带 hash 后插
    #   _commit_admission     借用命中的闲置块时从链上摘掉（O(1)）
    #
    # 不变量：set(链表) == {b | block_usage[b] == 0}，且无重复。

    def _list_link(self, prev_idx, block_idx, next_idx):
        self.block_next[prev_idx] = block_idx
        self.block_prev[block_idx] = prev_idx
        self.block_next[block_idx] = next_idx
        self.block_prev[next_idx] = block_idx
        self.num_allocatable += 1

    def _list_prepend(self, block_idx):
        """插到队首：下一次分配/淘汰最先拿到它（无 hash 的真正空闲块走这里）。"""
        self._list_link(self._SENTINEL_HEAD, block_idx,
                        self.block_next[self._SENTINEL_HEAD])

    def _list_append(self, block_idx):
        """插到队尾：最后才被淘汰（带 hash 的闲置缓存块走这里，FIFO/LRU）。"""
        self._list_link(self.block_prev[self._SENTINEL_TAIL], block_idx,
                        self._SENTINEL_TAIL)

    def _list_remove(self, block_idx):
        """按块自己的前后指针摘掉，O(1)，不需要位置索引。"""
        prev_idx = self.block_prev[block_idx]
        next_idx = self.block_next[block_idx]
        self.block_next[prev_idx] = next_idx
        self.block_prev[next_idx] = prev_idx
        self.block_prev[block_idx] = -1
        self.block_next[block_idx] = -1
        self.num_allocatable -= 1

    def _in_alloc_list(self, block_idx):
        return self.block_prev[block_idx] != -1

    def _peek_allocatable(self, k):
        """只读地取队首前 k 个可分配块，**不改链表**。

        计划阶段不能真的取走——`_plan_block_growth()` 失败时池状态必须原样不动。
        沿 next 指针走 k 步，代价 O(k)。
        """
        picked = []
        cur = self.block_next[self._SENTINEL_HEAD]
        while cur != self._SENTINEL_TAIL and len(picked) < k:
            picked.append(cur)
            cur = self.block_next[cur]
        return picked

    def _release_block(self, block_idx):
        """引用数降到 0：无 hash 的放队首（优先复用），带 hash 的放队尾（后淘汰）。"""
        if block_idx in self.block_to_hash:
            self._list_append(block_idx)      # 闲置缓存：留着，最后才淘汰
        else:
            self._list_prepend(block_idx)     # 真正空闲：下次优先复用

    def _allocatable_block_indices(self):
        """**扫全池**算出可分配块，与链表互相独立。

        两个用途，都**不在正常分配路径上**：
        - 出错信息里报池子现状；
        - 校验链表的不变量（测试用它当基准，不能拿链表自己校验自己）。

        正常分配走 `_peek_allocatable()`（提交时按块自己的指针摘掉）——池子大了，
        这份 O(num_kv_blocks) 的扫描就是这两关要消掉的瓶颈。
        """
        return [i for i in range(self.num_kv_blocks) if self.block_usage[i] == 0]

    def _free_block_indices(self):
        """扫全池算出「真正空闲」（没有引用也没有 hash）——只给错误信息与校验用。"""
        return [i for i in range(self.num_kv_blocks)
                if self.block_usage[i] == 0 and i not in self.block_to_hash]

    def _evictable_block_indices(self, exclude=()):
        """扫全池算出「闲置缓存」——只给错误信息与校验用。"""
        return [i for i in range(self.num_kv_blocks)
                if self.block_usage[i] == 0 and i in self.block_to_hash and i not in exclude]

    def _evict_block(self, block_idx):
        # 删除 key 与物理块的关联，之后这个块才能被重新分配。
        # 链表位置由调用方处理：补块时它是被取走的，准入时它是被借走的。
        del self.hash_to_block[self.block_to_hash.pop(block_idx)]

    def _evict_hash_if_cached(self, block_idx):
        """取走一个块之前，如果它带着 prefix hash，就地清掉缓存条目
        （vLLM 的 `_maybe_evict_cached_block`）——这个块马上要被覆写。"""
        if block_idx in self.block_to_hash:
            self._evict_block(block_idx)

    # ---- 按需分配 ----
    #
    # 分两层，别混在一起：
    #
    #   准入（allocate_block）  只判「这条请求**单独跑**装不装得下」——装不下就抛
    #                           InfeasibleRequest，那是谁也帮不了它的请求。
    #                           准入**不锁未来容量**：一条刚进来的请求占用块数是 0。
    #   生长（ensure_blocks）   真正要写 KV 之前，按这一轮要写的 token 数补物理块。
    #
    # 为什么不锁：块是稀缺资源，未来会不会真的用到那么长谁也说不准。按最坏情况预留在
    # 账面，短请求会替长请求白占容量；池子不够时也不是死路——让调度器选一条犧牲者
    # 把块腾出来就行（vLLM V1 就是这么做的，它连「抢占模式」这个开关都没有）。
    # 代价是池子可能真的不够，那时 `ensure_blocks()` 返回 False，由 `Scheduler` 决定
    # 谁让路——这个决定依赖优先级，只有调度器做得了。

    # ---- 准入：先计划、后提交 ----

    def allocate_block(self, seq: SequenceConfig):
        """准入。失败时返回 False，且不留下任何副作用。"""
        plan = self._plan_admission(seq)
        if plan is None:
            return False        # 暂时不够，等别人让出来
        self._commit_admission(seq, plan)
        return True

    def _plan_admission(self, seq: SequenceConfig):
        """只读地判断这条请求能不能准入。

        - 不改块引用、不改 LRU、不改块表 / cache.length / reused_tokens；
        - 返回 None 表示「暂时不够」；
        - 不可能完成时抛 InfeasibleRequest（单独跑也装不下，谁也帮不了它）。
        """
        max_request_blocks = math.ceil(
            (len(seq.prompt_ids) + seq.max_new_tokens - 1) / self.block_size)
        matched_block_ids, matched_hashes = [], []
        if self.enable_prefix_caching:
            # 命中的是**完整已知历史**（prompt + 已生成）里最长的完整块前缀。
            # 上限取 len(all_token_ids) - 1：本实现没有缓存 logits，
            # 至少要留一个历史 token 重新进模型，才能产生下一 token 的分数。
            #
            # 首次准入时 all_token_ids == prompt_ids，命中的块数与旧实现的
            # find_matched_prefix_blocks(prompt_ids[:-1]) 完全一致；
            # 被抢占后重新准入时，这段历史里还包含已经生成过的 token。
            all_ids = seq.all_token_ids
            matched_block_ids, matched_hashes = self.find_matched_prefix_blocks(
                all_ids, max_tokens=len(all_ids) - 1)
        required_new_blocks = max_request_blocks - len(matched_block_ids)

        if max_request_blocks > self.num_kv_blocks:
            # 可行性看的是**最坏逻辑块数**，不能用「碰巧命中了几块」来放宽：
            # 命中的块也占物理块——这条请求要同时持有 max_request_blocks 块才能跑完
            # （命中的块被引用期间不可淘汰），所以池子装不下它就是要拒绝。
            raise InfeasibleRequest(
                f"请求 {seq.request_id!r} 永远无法完成：它最多需要 {max_request_blocks} 块"
                f"（prompt {len(seq.prompt_ids)} + 输出上限 {seq.max_new_tokens} 个 token，"
                f"block_size={self.block_size}），而整个 KV 池只有 {self.num_kv_blocks} 块。"
                f"当前池子：空闲 {len(self._free_block_indices())} 块、"
                f"闲置缓存 {len(self._evictable_block_indices())} 块；"
                f"命中的前缀块 {len(matched_block_ids)} 块"
                f"（命中不减需求：命中的块同样占物理块）。"
                f"请调大 num_kv_blocks，或调小这条请求的 max_new_tokens")

        return AdmissionPlan(matched_block_ids, matched_hashes, required_new_blocks)

    def _commit_admission(self, seq: SequenceConfig, plan: AdmissionPlan):
        """准入提交：从这里才开始改引用计数与请求状态。"""
        for block_idx in plan.matched_block_ids:
            if self._in_alloc_list(block_idx):
                # 借的是「闲置缓存」块（引用还是 0）：先从可分配链上摘掉，再改引用。
                # 命中已被其他请求持有的块时它不在链上，这里自然跳过。
                self._list_remove(block_idx)
            self.block_usage[block_idx] += 1
        seq.cache.block_table = list(plan.matched_block_ids)
        seq.cache.length = len(plan.matched_block_ids) * self.block_size
        seq.block_hashes = list(plan.matched_hashes)
        # 到这里才真的借到
        seq.reused_tokens += len(plan.matched_block_ids) * self.block_size

    # ---- 本轮补块：先计划、后提交 ----

    def ensure_blocks(self, seq: SequenceConfig, num_new_tokens: int):
        """按需补齐：保证块表能覆盖 [0, cache.length + num_new_tokens)。只补差额。"""
        return self.ensure_blocks_for(seq.cache, num_new_tokens)

    def ensure_blocks_for(self, cache, num_new_tokens: int):
        """`ensure_blocks()` 的**按 cache 操作**版本（第五十五关）。

        块管理只有一份实现：target 池与 draft 池是同一个类的两个实例，差别只在
        「往哪个 CacheConfig 上记账」。draft 池走的正是这个入口——它没有准入、
        没有前缀缓存，也不需要认识 `SequenceConfig`。
        """
        needed_blocks = math.ceil((cache.length + num_new_tokens) / self.block_size)
        # draft 的 CacheConfig 一开始就是空的（block_table 为 None），按空表算
        num_missing_blocks = needed_blocks - len(cache.block_table or ())
        if num_missing_blocks <= 0:
            return True

        plan = self._plan_block_growth(num_missing_blocks)
        if plan is None:
            # 本次就是分不到。一个字节都不改：不追加 block table、不动引用计数、
            # 不改 cache.length。谁该让路由 Scheduler 决定，池子不认识请求优先级。
            return False
        self._commit_block_growth(cache, plan)
        return True

    def _plan_block_growth(self, num_missing_blocks: int):
        """只读地选出本轮要取走的物理块。

        直接从可分配链的**队首**取前 k 个——不扫全池，这是本关要消掉的瓶颈。
        队首是「真正空闲」块（无 hash，直接可用），只有不够时才会取到带 hash 的
        闲置缓存块，那些在提交阶段就地清掉 hash。

        **绝不为了「试一试」先淘汰**：计划阶段只读，不改链表、不删 hash、不动引用。
        不够就返回 None，让调度器决定谁让路。
        """
        block_ids = self._peek_allocatable(num_missing_blocks)
        if len(block_ids) < num_missing_blocks:
            return None
        return BlockGrowthPlan(block_ids)

    def _commit_block_growth(self, cache, plan: BlockGrowthPlan):
        for block_idx in plan.new_block_ids:
            # 计划里选的这些块此刻仍在链上——计划与提交之间没有任何东西改链表，
            # 所以直接按块自己的前后指针摘掉即可，O(1)，不必再从队首走一遍。
            self._list_remove(block_idx)
            # 取到带 hash 的块就地清掉缓存条目（它马上要被覆写）
            self._evict_hash_if_cached(block_idx)
            self.block_usage[block_idx] = 1
        if cache.block_table is None:
            cache.block_table = []      # draft 的 cache 从空表开始
        cache.block_table.extend(plan.new_block_ids)

    def can_grow(self, seq: SequenceConfig, num_new_tokens: int):
        """只读地问一句：现在能不能补出这么多块。**不改任何状态**。

        给「计划阶段」用（第五十四关）：投机解码的草稿是可选加速，容量不够就该
        在计划里缩短它，而不是等 `ensure_blocks()` 走到提交阶段才发现。只沿可分配链
        看前 k 个够不够，不摘链、不清 hash、不动引用。
        """
        return self.can_grow_for(seq.cache, num_new_tokens)

    def can_grow_for(self, cache, num_new_tokens: int):
        """`can_grow()` 的按 cache 操作版本（第五十五关，draft 池用）。"""
        needed_blocks = math.ceil((cache.length + num_new_tokens) / self.block_size)
        # draft 的 CacheConfig 一开始就是空的（block_table 为 None），按空表算
        num_missing_blocks = needed_blocks - len(cache.block_table or ())
        if num_missing_blocks <= 0:
            return True
        return len(self._peek_allocatable(num_missing_blocks)) >= num_missing_blocks

    def truncate(self, seq: SequenceConfig, new_length: int):
        """把请求的 KV 进度退回 `new_length`（投机解码拒绝了草稿之后）。

        只动两样：`cache.length` 与**多占的整块**。仍在用的完整块留在块表里不动；
        不完整的那个尾块以后会被新内容覆盖，不需要清内容——所以这里既不写 KV，
        也不碰 hash 条目（block_size 对齐的位置本来就不该被发布过）。

        调用时机有硬约束：**必须在 `Scheduler.post_step()` 之前**。被拒绝草稿的 KV
        是随本轮输入一起写进物理块的，`cache.length` 先退回去，post_step 里的
        `publish_computed_blocks()` 读到的才是真实进度，拒绝的部分不会被登记成
        可复用前缀。
        """
        if new_length < 0 or new_length > seq.cache.length:
            raise ValueError(f"回滚目标 {new_length} 不在 [0, {seq.cache.length}] 内"
                             f"（请求 {seq.request_id!r}）")
        self.truncate_cache(seq.cache, new_length)

    def truncate_cache(self, cache, new_length: int):
        """`truncate()` 的按 cache 操作版本（第五十五关，draft 池用）。

        与 target 侧同一份回滚逻辑：只动 `length` 与多占的整块。
        """
        if new_length < 0 or new_length > cache.length:
            raise ValueError(f"回滚目标 {new_length} 不在 [0, {cache.length}] 内")
        keep_blocks = math.ceil(new_length / self.block_size)
        for block_idx in cache.block_table[keep_blocks:]:
            self.block_usage[block_idx] -= 1
            if self.block_usage[block_idx] == 0:
                # 和释放一样：无 hash 的回队首，带 hash 的留作闲置缓存
                self._release_block(block_idx)
        del cache.block_table[keep_blocks:]
        cache.length = new_length

    def deallocate_block(self, seq: SequenceConfig):
        # 只释放本请求持有的全部活动引用；已登记的缓存条目继续保留为闲置缓存
        self.release_cache(seq.cache)

    def release_cache(self, cache):
        """`deallocate_block()` 的按 cache 操作版本（第五十五关，draft 池用）。

        调用方负责把 cache 复位成 `CacheConfig()`（target 侧就是 `seq.cache = ...`）。
        """
        for block_idx in cache.block_table:
            self.block_usage[block_idx] -= 1
            if self.block_usage[block_idx] == 0:
                # 引用降到 0：无 hash 的放队首（优先复用），带 hash 的放队尾（后淘汰）
                self._release_block(block_idx)

    def publish_computed_blocks(self, seq: SequenceConfig):
        """把「已经算完、KV 确实写进了物理块」的完整块登记为可复用。

        这是 prompt 与生成 token 共用的**唯一**发布入口，不再按来源分两套 hash 逻辑。

        覆盖范围完全由 `cache.length` 决定——它表示已经进入模型并写入 KV 的 token 数：

        - 只有整块都被 `cache.length` 覆盖的块才登记，不完整的尾块不登记；
        - 刚采样出来、还没进模型的 token 不在 `cache.length` 里，所以**不可能**
          被当成命中（block_size=4、prompt 长 3 时，采样出 y0 后 cache.length 仍是 3，
          第 0 块还差一个 token，不会发布；下一轮 y0 真的算完，cache.length=4，才发布）；
        - hash 链按完整 `all_token_ids` 的块顺序计算，前块 hash 参与后块 hash，
          所以生成 token 落进的块也有稳定身份。

        已登记过的块不会被后续写入改动：`cache.length` 只增不减，这些位置不会再被写。
        同 hash 的条目已存在时保留原条目（另一个物理块内容相同，不必重复登记）。
        """
        if not self.enable_prefix_caching:
            return

        all_ids = seq.all_token_ids
        # 不需要 min(len(all_ids), cache.length)：cache.length 恒**严格小于**
        # len(all_token_ids)——刚采样的那个 token 已经在历史里、但还没进模型
        # （本实现不缓存 logits，至少留一个历史 token 走模型），所以永远差至少 1 个。
        # 实测 54 组负载 747 次调用，差值最小为 1、相等 0 次。
        full_blocks = seq.cache.length // self.block_size
        if full_blocks <= len(seq.block_hashes):
            # 本轮没有新算满的完整块——最常见的情况（每步只有 1 个 token 进模型，
            # 块很久才满一次）。注意：**下面的循环这时本来就是空的**，所以这一句
            # 不是正确性需要的，只是省掉为每个请求构造 range 并进入一次迭代器
            # （实测 16 请求约 0.3 μs）。hash 链与切片由循环的空转本身保证不动。
            return

        for i in range(len(seq.block_hashes), full_blocks):
            block = tuple(all_ids[i * self.block_size:(i + 1) * self.block_size])
            previous_hash = seq.block_hashes[-1] if seq.block_hashes else b""
            hash_value = _stable_hash(previous_hash, block)
            seq.block_hashes.append(hash_value)

            if hash_value in self.hash_to_block:
                continue

            block_idx = seq.cache.block_table[i]
            self.hash_to_block[hash_value] = block_idx
            self.block_to_hash[block_idx] = hash_value

    def _slots_of_range(self, block_table, start, count):
        # 请求内逻辑位置 [start, start+count) 对应的物理槽位：块编号 * block_size + 块内偏移。
        #
        # 纯 CPU 整数运算。区间按块切开，每一块对应一段**连续**的槽位，用 range 展开
        # （C 层），既不逐位置走 Python 循环，也不建任何临时张量。
        #
        # 之前这里是「每个请求两次 H2D」：torch.arange(..., device=cuda) 传起点、
        # torch.tensor(block_table, device=cuda) 把 Python 列表搬上 GPU。decode 时
        # count=1、数据只有一个，传的全是发射与 pageable H2D 的开销——实测 1.0 ms/步。
        bs = self.block_size
        slots = []
        for blk in range(start // bs, (start + count - 1) // bs + 1):
            lo = max(start, blk * bs)
            hi = min(start + count, (blk + 1) * bs)
            base = block_table[blk] * bs
            slots.extend(range(base + lo - blk * bs, base + hi - blk * bs))
        return slots

    def build_slot_mapping(self, caches, counts):
        # 本轮打包输入中第 i 个 token 的 K/V 应写到哪个 slot；用写入前的 length 算地址。
        # 返回 CPU 上的 long 张量（对外接口不变），整批一次搬上去，而不是每个请求一次。
        slots = []
        for cache, count in zip(caches, counts):
            slots.extend(self._slots_of_range(cache.block_table, cache.length, count))
        return torch.tensor(slots, dtype=torch.long)

    def block_view(self, cache: CacheConfig, logical_block, count, layer_idx=0):
        # 直接给出池里这个物理块在某一层上的有效切片；仍是池存储的视图，不复制
        block_idx = cache.block_table[logical_block]
        return self.k_cache[layer_idx, block_idx][:count], self.v_cache[layer_idx, block_idx][:count]

    def gather(self, cache: CacheConfig, layer_idx=0):
        # 按请求逻辑位置 0..length-1 一次选出某一层的 K 和 V，按逻辑顺序返回
        # 保留作参考/调试；attention 路径不再调用它

        if cache.length == 0:
            empty = (0, self.num_kv_heads, self.head_dim)
            return self.k_cache.new_empty(empty), self.v_cache.new_empty(empty)

        slots = torch.tensor(self._slots_of_range(cache.block_table, 0, cache.length),
                             dtype=torch.long, device=self.device)
        return self.k_flat[layer_idx].index_select(0, slots), self.v_flat[layer_idx].index_select(0, slots)


    def find_matched_prefix_blocks(self, token_ids, max_tokens=None):
        """从块 0 开始，找最长、连续且仍在缓存里的完整块前缀。

        `token_ids` 是请求的完整已知历史（all_token_ids = prompt + output），
        不只是 prompt：生成出来的 token 也可能落在某个完整块里。

        `max_tokens` 是允许命中的 token 上限，默认整段 `token_ids`。
        这个参数只是把「最多能命中多少 token」交回给调用方——**本函数不认识
        「要留一个 token 出 logits」这条策略**，那是准入的事：准入传
        `len(all_token_ids) - 1`，这样最后一个历史 token 一定还会进模型。

        返回 (物理块, hash) 两个列表；一块都没命中时返回两个空列表。
        """
        limit = len(token_ids) if max_tokens is None else max_tokens

        matched_blocks = []
        matched_hashes = []
        block_num = max(limit, 0) // self.block_size
        current_hash = b""
        for i in range(0, block_num):
            block = tuple(token_ids[i * self.block_size:(i + 1) * self.block_size])
            current_hash = _stable_hash(current_hash, block)
            if current_hash not in self.hash_to_block:
                break
            matched_blocks.append(self.hash_to_block[current_hash])
            matched_hashes.append(current_hash)

        return matched_blocks, matched_hashes
