"""KV 缓存池：分块存储、slot 寻址、前缀缓存与 LRU 淘汰。

这一层管的是「物理块放在哪、谁还在引用它」，不管 attention 怎么算。
"""

import hashlib
import json
import math
from dataclasses import dataclass

import torch


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
        # 采样参数与状态跟着请求走，不跟着 batch 行号走
        self.sampling_params = sampling_params
        self.sampling_state = sampling_state

    @property
    def prefill_len(self):
        return max(len(self.prompt_ids) - self.cache.length, 0)


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
        self.enable_prefix_caching = enable_prefix_caching
        self.block_hash = {}  # 前缀 hash -> 该块物理块编号
        self.block_to_hash = {}  # 物理块编号 -> 仍保留它的缓存条目 hash
        self.block_last_used = [0] * self.num_kv_blocks  # LRU 序号
        self.lru_seq = 0

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

    def allocate_block(self, seq: SequenceConfig):
        # 先借用命中的前缀块，再补齐私有块；失败时返回 False，且不留下任何副作用

        total_blocks_needed = math.ceil((len(seq.prompt_ids) + seq.max_new_tokens - 1) / self.block_size)

        matched_blocks, matched_hashes = [], []
        if self.enable_prefix_caching:
            # 至少留下最后一个 prompt token 重新计算：本关不缓存 logits
            matched_blocks, matched_hashes = self.find_matched_prefix_blocks(seq.prompt_ids[:-1])
        new_blocks_needed = total_blocks_needed - len(matched_blocks)

        free_blocks = self._free_block_indices()
        evict_blocks = []
        if len(free_blocks) < new_blocks_needed:
            # 只能淘汰闲置缓存（活动引用为 0），且不能淘汰本次要借用的命中块
            idle_cached = [i for i in range(self.num_kv_blocks)
                           if self.block_usage[i] == 0
                           and i in self.block_to_hash
                           and i not in matched_blocks]
            idle_cached.sort(key=lambda i: self.block_last_used[i])
            if len(free_blocks) + len(idle_cached) < new_blocks_needed:
                return False
            evict_blocks = idle_cached[:new_blocks_needed - len(free_blocks)]

        # 容量已经确认足够，从这里开始改动状态
        for block_idx in matched_blocks:
            self.block_usage[block_idx] += 1
            self._mark_used(block_idx)
        for block_idx in evict_blocks:
            self._evict_block(block_idx)
        new_blocks = (free_blocks + evict_blocks)[:new_blocks_needed]
        for block_idx in new_blocks:
            self.block_usage[block_idx] = 1

        seq.cache.block_table = matched_blocks + new_blocks
        seq.cache.length = len(matched_blocks) * self.block_size
        seq.block_hashes = list(matched_hashes)
        return True

    def deallocate_block(self, seq: SequenceConfig):
        # 只释放本请求持有的全部活动引用；已登记的缓存条目继续保留为闲置缓存

        for block_idx in seq.cache.block_table:
            self.block_usage[block_idx] -= 1

    def publish_completed_prompt_blocks(self, seq: SequenceConfig):
        # 登记请求已经写完 KV 的完整 prompt 块；已有 key 保留原条目，私有块随请求释放

        full_blocks = min(len(seq.prompt_ids), seq.cache.length) // self.block_size

        for i in range(len(seq.block_hashes), full_blocks):
            block = tuple(seq.prompt_ids[i * self.block_size:(i + 1) * self.block_size])
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
        # 请求内逻辑位置 [start, start+count) 对应的物理槽位：块编号 * block_size + 块内偏移
        positions = torch.arange(start, start + count, device=self.device)
        blocks = torch.tensor(block_table, device=self.device, dtype=torch.long)
        return blocks[positions // self.block_size] * self.block_size + positions % self.block_size

    def build_slot_mapping(self, caches, counts):
        # 本轮打包输入中第 i 个 token 的 K/V 应写到哪个 slot；用写入前的 length 算地址
        return torch.cat([self._slots_of_range(cache.block_table, cache.length, count)
                          for cache, count in zip(caches, counts)])

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

        slots = self._slots_of_range(cache.block_table, 0, cache.length)
        return self.k_flat[layer_idx].index_select(0, slots), self.v_flat[layer_idx].index_select(0, slots)


    def find_matched_prefix_blocks(self, prompt_ids):

        # 从第一块开始连续匹配，遇到缺失就停止；返回 (物理块, hash) 两个列表
        matched_blocks = []
        matched_hashes = []
        block_num = len(prompt_ids) // self.block_size
        current_hash = b""
        for i in range(0, block_num):
            block = tuple(prompt_ids[i * self.block_size:(i + 1) * self.block_size])
            current_hash = _stable_hash(current_hash, block)
            if current_hash not in self.block_hash:
                break
            matched_blocks.append(self.block_hash[current_hash])
            matched_hashes.append(current_hash)

        return matched_blocks, matched_hashes
