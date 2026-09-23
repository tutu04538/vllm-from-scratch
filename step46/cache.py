"""KV 缓存池：分块存储、slot 寻址、前缀缓存与 LRU 淘汰。

这一层管的是「物理块放在哪、谁还在引用它」，不管 attention 怎么算。
"""

import hashlib
import json
import math
from dataclasses import dataclass

import torch


class InfeasibleRequest(Exception):
    """请求在任何情况下都不可能完成（例如它自己要的块数超过整个池子）。

    和「暂时不够」不是一回事：暂时不够会返回 False 让它继续排队，这个必须让调用方看见。
    """


def _stable_hash(previous_hash: bytes, block: tuple[int, ...]) -> bytes:
    # token ID 不限于 0~255，先编码成文本，再交给 sha256
    data = json.dumps((previous_hash.hex(), block)).encode("utf-8")
    return hashlib.sha256(data).digest()


@dataclass
class CacheConfig:
    block_table: list[int] = None # List of block indices in the KV cache
    length: int = 0


class SequenceConfig:
    def __init__(self, request_id, prompt_ids, max_new_tokens, block_size,
                 sampling_params=None, sampling_state=None):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.max_new_tokens = max_new_tokens
        self.output_ids = []
        self.cache = CacheConfig()
        self.block_size = block_size  # Size of each block in the KV cache
        self.block_hashes = []  # 本请求已确定的前缀块 hash 链，命中时从缓存里的前缀接上
        # 准入时按最坏情况承诺、还没分配出去的块数；随 ensure_blocks 递减、随释放归还
        self.promised_blocks = 0
        # 采样参数与状态跟着请求走，不跟着 batch 行号走
        self.sampling_params = sampling_params
        self.sampling_state = sampling_state
        # 抢占：本请求被抢占的次数；真正重新进入模型的历史 token 数；曾计算到的最大位置
        self.num_preemptions = 0
        self.recomputed_tokens = 0
        self.high_water = 0
        # 准入时真正从 prefix cache 借到的完整块换来的 token 数（累加）。
        # 命中失败/被撤销的块不计入——只有最终成功借到才算。
        self.reused_tokens = 0
        # 是谁迫使本请求让出 KV（阻塞者）。保存**请求对象引用**而不是 request_id
        # 字符串——本引擎没有建立「ID 唯一」的强约束，比字符串可能认错人。
        # 只在 recompute 模式被赋值；阻塞者结束后会被清掉，不残留引用。
        self.resume_blocker = None

    @property
    def all_token_ids(self):
        """模型应当「看过」的完整历史：prompt + 已生成的 token。

        最后一个已生成的 token 通常还没进模型——它就是下一轮的输入——
        所以 all_token_ids 总是比 KV 里已有的内容多一个。
        """
        return self.prompt_ids + self.output_ids

    @property
    def num_uncomputed_tokens(self):
        """还没进入模型、没写进 KV 的 token 数。

        「已计算长度」的唯一真相是 cache.length，这里只算差值，不再另存一份
        可能和它互相矛盾的计数。
        """
        return max(len(self.prompt_ids) + len(self.output_ids) - self.cache.length, 0)

    @property
    def is_ready_for_next_token(self):
        """历史只差最后一个 token 没算：本轮再算 1 个就能采样。

        被抢占过的请求重算到这一步时同样成立，所以重算结束后的第一次采样
        走的是和 decode 完全一样的记账，不需要另开一条路径。
        """
        return (self.num_uncomputed_tokens == 1
                and self.cache.length >= len(self.prompt_ids))

    @property
    def prefill_len(self):
        # 旧名字，等价于 num_uncomputed_tokens；无抢占时 clamp 前后的值相同
        return self.num_uncomputed_tokens


class KVCachePool:

    def __init__(self, block_size, num_kv_blocks, num_kv_heads, head_dim, device,
                 enable_prefix_caching=True, num_layers=1, dtype=torch.float32,
                 over_subscribe=False):
        # over_subscribe：允许容量超卖（recompute 模式）。关掉按最坏情况承诺未来块，
        # 准入只判断「这条请求单独跑是否可行」，块在真正排到 token 时才补。
        # 代价是池子可能不够——那时由 Scheduler 选犧牲者，池子本身不做这个决定。
        self.over_subscribe = over_subscribe
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
        self.enable_prefix_caching = enable_prefix_caching
        self.block_hash = {}  # 前缀 hash -> 该块物理块编号
        self.block_to_hash = {}  # 物理块编号 -> 仍保留它的缓存条目 hash
        self.block_last_used = [0] * self.num_kv_blocks  # LRU 序号
        self.lru_seq = 0
        # 所有活动请求已承诺、但还没真正分配出去的块数之和。
        # 准入要扣掉它，否则多条请求会各自按「当前空闲」判断，合起来超额承诺。
        self.promised_blocks = 0

    def _free_block_indices(self):
        # 真正空闲：没有活动请求引用，也没有被前缀缓存保留
        return [i for i in range(self.num_kv_blocks)
                if self.block_usage[i] == 0 and i not in self.block_to_hash]

    def _mark_used(self, block_idx):
        self.lru_seq += 1
        self.block_last_used[block_idx] = self.lru_seq

    def _evict_block(self, block_idx):
        # 先删除 key 与物理块的关联，之后这个块才能被重新分配
        del self.block_hash[self.block_to_hash.pop(block_idx)]

    # ---- 按需分配 ----
    #
    # 分两层，别混在一起：
    #
    #   准入（allocate_block）  用**最坏情况**判断这条请求能不能跑完，并按块数「承诺」额度。
    #                           承诺是记账，不占物理块——所以一条刚进来的请求占用块数是 0。
    #   生长（ensure_blocks）   真正要写 KV 之前，按这一轮要写的 token 数补物理块。
    #
    # 为什么要保留「准入时按最坏情况记账」：只按需分配、准入不做总量控制的话，
    # 池子 4 块、两条各需 3 块的请求会各自拿 1 块起步，然后同时卡在长第 3 块上——
    # 零进展且无错误。现在这套「承诺额度」把总量卡住，任何已接纳的请求都能长到它承诺的量。

    def _evictable_block_indices(self, exclude=()):
        # 活动引用为 0、但仍被前缀缓存保留的块；这些可以淘汰回收
        idx = [i for i in range(self.num_kv_blocks)
               if self.block_usage[i] == 0 and i in self.block_to_hash and i not in exclude]
        idx.sort(key=lambda i: self.block_last_used[i])
        return idx

    def _available_blocks(self, exclude=()):
        # 现在能拿到的块数：直接空闲 + 可淘汰的闲置缓存，再扣掉别人已承诺还没用的。
        # exclude 是本条请求「本次就要借用的命中前缀块」——它们的引用计数要等检查通过
        # 才加上去，此刻看起来还是「活动引用为 0 的闲置缓存」，不排掉就会把
        # 马上要用的块算成可用，准入因此偏松。
        return (len(self._free_block_indices())
                + len(self._evictable_block_indices(exclude=exclude))
                - self.promised_blocks)

    def allocate_block(self, seq: SequenceConfig):
        # 准入：借用命中的前缀块 + 按最坏情况承诺额度；**不一次占满物理块**。
        # 失败时返回 False，且不留下任何副作用。

        worst_case = math.ceil((len(seq.prompt_ids) + seq.max_new_tokens - 1) / self.block_size)
        matched_blocks, matched_hashes = [], []
        if self.enable_prefix_caching:
            # 命中的是**完整已知历史**（prompt + 已生成）里最长的完整块前缀。
            # 上限取 len(all_token_ids) - 1：本实现没有缓存 logits，
            # 至少要留一个历史 token 重新进模型，才能产生下一 token 的分数。
            #
            # 首次准入时 all_token_ids == prompt_ids，命中的块数与旧实现的
            # find_matched_prefix_blocks(prompt_ids[:-1]) 完全一致；
            # 被抢占后重新准入时，这段历史里还包含已经生成过的 token。
            all_ids = seq.all_token_ids
            matched_blocks, matched_hashes = self.find_matched_prefix_blocks(
                all_ids, max_tokens=len(all_ids) - 1)
        # 命中的前缀块是复用的，不占新的块；剩下这些才是它最终可能需要的
        need = worst_case - len(matched_blocks)

        if worst_case > self.num_kv_blocks:
            # 可行性看的是**最坏逻辑块数**，不能用「碰巧命中了几块」来放宽：
            # 命中的块也占物理块——这条请求要同时持有 worst_case 块才能跑完
            # （命中的块被引用期间不可淘汰），所以池子装不下它就是要拒绝。
            # 整个池子给它一个人都不够，永远不可能跑完 -> 明确拒绝（不留在队列里空转）
            raise InfeasibleRequest(
                f"请求 {seq.request_id!r} 永远无法完成：它最多需要 {worst_case} 块"
                f"（prompt {len(seq.prompt_ids)} + 输出上限 {seq.max_new_tokens} 个 token，"
                f"block_size={self.block_size}），而整个 KV 池只有 {self.num_kv_blocks} 块。"
                f"当前池子：空闲 {len(self._free_block_indices())} 块、"
                f"闲置缓存 {len(self._evictable_block_indices())} 块、"
                f"他人已承诺 {self.promised_blocks} 块；命中的前缀块 "
                f"{len(matched_blocks)} 块（命中不减需求：命中的块同样占物理块）。"
                f"请调大 num_kv_blocks，或调小这条请求的 max_new_tokens")
        if not self.over_subscribe and need > self._available_blocks(exclude=set(matched_blocks)):
            return False        # 暂时不够，等别人让出来

        for block_idx in matched_blocks:
            self.block_usage[block_idx] += 1
            self._mark_used(block_idx)
        seq.cache.block_table = list(matched_blocks)
        seq.cache.length = len(matched_blocks) * self.block_size
        seq.block_hashes = list(matched_hashes)
        # 到这里才真的借到：失败路径（返回 False / 抛 InfeasibleRequest）都不计
        seq.reused_tokens += len(matched_blocks) * self.block_size
        if not self.over_subscribe:
            # 超卖模式下什么都不锁：未来容量靠抢占腾，不靠账面预留
            seq.promised_blocks = need
            self.promised_blocks += need
        return True

    def ensure_blocks(self, seq: SequenceConfig, num_new_tokens: int):
        """按需补齐：保证块表能覆盖 [0, cache.length + num_new_tokens)。只补差额。"""
        needed = math.ceil((seq.cache.length + num_new_tokens) / self.block_size)
        extra = needed - len(seq.cache.block_table)
        if extra <= 0:
            return True
        # 从这里往下要么补齐返回 True，要么一个字节都不动地返回 False（超卖模式）

        free_blocks = self._free_block_indices()
        evict_blocks = []
        if len(free_blocks) < extra:
            # 只能淘汰闲置缓存。不需要排掉本请求正在用的块：它们 block_usage >= 1，
            # 而 _evictable_block_indices 的第一项就是 block_usage == 0，本来就选不中。
            # （准入那边不同：命中的前缀块此时引用计数还是 0，必须显式排掉。）
            idle = self._evictable_block_indices()
            if len(free_blocks) + len(idle) < extra:
                if self.over_subscribe:
                    # 本次就是分不到。一个字节都不改：不追加 block table、不动引用计数、
                    # 不改 cache.length。谁该让路由 Scheduler 决定，池子不认识请求优先级。
                    return False
                # 承诺过的额度必然拿得到；走到这里说明记账被破坏了，宁可报出来
                raise RuntimeError(
                    f"请求 {seq.request_id!r} 想补 {extra} 块但只找到 "
                    f"{len(free_blocks)} 空闲 + {len(idle)} 可淘汰；"
                    f"它承诺了 {seq.promised_blocks} 块，池子 {self.num_kv_blocks} 块，"
                    f"全局已承诺 {self.promised_blocks} 块——准入记账出错了")
            evict_blocks = idle[:extra - len(free_blocks)]

        for block_idx in evict_blocks:
            self._evict_block(block_idx)
        new_blocks = (free_blocks + evict_blocks)[:extra]
        for block_idx in new_blocks:
            self.block_usage[block_idx] = 1
            self._mark_used(block_idx)
        seq.cache.block_table.extend(new_blocks)
        if not self.over_subscribe:
            # 承诺式：「已承诺额度」在这里换成真实块，两个计数同步递减。
            # 超卖模式准入时根本没承诺过，这里减 extra 会把账本减成负数——
            # 结束时减去负数恰好回到 0，所以只有逐步检查才发现得了。
            seq.promised_blocks -= extra
            self.promised_blocks -= extra
        return True

    def deallocate_block(self, seq: SequenceConfig):
        # 只释放本请求持有的全部活动引用；已登记的缓存条目继续保留为闲置缓存
        # 还没用掉的承诺额度也要一并还回去，否则池子会被账面占满

        for block_idx in seq.cache.block_table:
            self.block_usage[block_idx] -= 1
        # 还没用掉的承诺额度一并还回去。超卖模式下它恒为 0，这里是空操作。
        self.promised_blocks -= seq.promised_blocks
        seq.promised_blocks = 0

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
        full_blocks = min(len(all_ids), seq.cache.length) // self.block_size

        for i in range(len(seq.block_hashes), full_blocks):
            block = tuple(all_ids[i * self.block_size:(i + 1) * self.block_size])
            previous_hash = seq.block_hashes[-1] if seq.block_hashes else b""
            hash_value = _stable_hash(previous_hash, block)
            seq.block_hashes.append(hash_value)

            if hash_value in self.block_hash:
                continue

            block_idx = seq.cache.block_table[i]
            self.block_hash[hash_value] = block_idx
            self.block_to_hash[block_idx] = hash_value
            self._mark_used(block_idx)

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
            if current_hash not in self.block_hash:
                break
            matched_blocks.append(self.block_hash[current_hash])
            matched_hashes.append(current_hash)

        return matched_blocks, matched_hashes
