'''
Prefill and decode
'''

from dataclasses import dataclass
import math

from more_itertools import last
import torch
from torch import ceil, nn

import hashlib
import json


def _stable_hash(previous_hash: bytes, block: tuple[int, ...]) -> bytes:
    # token ID 不限于 0~255，先编码成文本，再交给 sha256
    data = json.dumps((previous_hash.hex(), block)).encode("utf-8")
    return hashlib.sha256(data).digest()


@dataclass
class CacheConfig:  
    block_table: list[int] = None # List of block indices in the KV cache
    length: int = 0


class SequenceConfig:
    def __init__(self, request_id, prompt_ids, max_new_tokens, block_size):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.max_new_tokens = max_new_tokens
        self.output_ids = []
        self.cache = CacheConfig()
        self.block_size = block_size  # Size of each block in the KV cache
        self.block_hashes = []  # 本请求已确定的前缀块 hash 链，命中时从缓存里的前缀接上

    @property
    def prefill_len(self):
        return max(len(self.prompt_ids) - self.cache.length, 0)


class KVCachePool:

    def __init__(self, block_size, num_kv_blocks, d_model, device, enable_prefix_caching=True):
        self.block_size = block_size
        self.num_kv_blocks = num_kv_blocks
        self.d_model = d_model
        self.device = device
        self.k_cache = torch.zeros(num_kv_blocks, block_size, d_model, device=device)
        self.v_cache = torch.zeros(num_kv_blocks, block_size, d_model, device=device)
        # 前两维看成一排 token 槽位，与底层存储共享，不是副本
        self.k_flat = self.k_cache.view(-1, d_model)
        self.v_flat = self.v_cache.view(-1, d_model)
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

    def append_batch(self, caches, counts, new_k, new_v):
        # 整批写入本轮真实 token，K/V 各一次批量索引写入；写完再各自增加 length
        slot_mapping = self.build_slot_mapping(caches, counts)

        self.k_flat.index_copy_(0, slot_mapping, new_k)
        self.v_flat.index_copy_(0, slot_mapping, new_v)

        for cache, count in zip(caches, counts):
            cache.length += count

    def gather(self, cache: CacheConfig):
        # 按请求逻辑位置 0..length-1 一次选出 K 和 V，按逻辑顺序返回

        if cache.length == 0:
            return self.k_cache.new_empty((0, self.d_model)), self.v_cache.new_empty((0, self.d_model))

        slots = self._slots_of_range(cache.block_table, 0, cache.length)
        return self.k_flat.index_select(0, slots), self.v_flat.index_select(0, slots)


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


class Sampler:
    
    def __init__(self):
        pass
    
    def sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)


class DummyModel:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = 5
        self.next_token_logits = torch.tensor([[0, 10, 0, 0, 0],
                                               [0, 0, 10, 0, 0],
                                               [0, 0, 0, 10, 0],
                                               [0, 0, 0, 0, 10],
                                               [0, 0, 0, 0, 10]], device=self.device, dtype=torch.float32)
        
    def forward(self, last_token_ids):
        # last_token_ids: Tensor of shape (batch_size,)
        # last_token_ids = last_token_ids.to(device=self.device, dtype=torch.long)
        return self.next_token_logits[last_token_ids]


class TinyCausalLM(nn.Module):
    
    def __init__(self, vocab_size=5, d_model=8, max_seq_len=32):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model).to(self.device)
        self.position_embedding = nn.Embedding(max_seq_len, d_model).to(self.device)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False).to(self.device)
        self.k_proj = nn.Linear(d_model, d_model, bias=False).to(self.device)
        self.v_proj = nn.Linear(d_model, d_model, bias=False).to(self.device)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(self.device)
        
    
    def forward(self, input_ids: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)
        position_embeds = self.position_embedding(position_ids)
        
        inputs_embeds = token_embeds + position_embeds
        
        q = self.q_proj(inputs_embeds)
        k = self.k_proj(inputs_embeds)
        v = self.v_proj(inputs_embeds)
        
        score = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)
        
        mask = torch.triu(torch.ones((seq_len, seq_len), device=input_ids.device), diagonal=1)
        score = score.masked_fill(mask == 1, float('-inf'))
        weights = torch.softmax(score, dim=-1)
        
        out = torch.matmul(weights, v)
        
        logits = self.lm_head(out)
        
        past_kv = [(_k, _v) for _k, _v in zip(k, v)]
        
        return logits, past_kv
    
    def _forward_append(self, input_ids: torch.Tensor, num_scheduled_tokens: list[int], past_kv: list[CacheConfig], kv_cache_pool: KVCachePool):
        # input_ids: (N,) 一维，只有真实 token，N = sum(num_scheduled_tokens)
        # past_kv: 与 num_scheduled_tokens 同序的 CacheConfig
        # 返回 logits (N, vocab_size)，行顺序与 input_ids 相同

        # 每个请求在扁平输入里的片段边界；打包数组的下标不是模型的位置
        offsets = [0]
        for num_tokens in num_scheduled_tokens:
            offsets.append(offsets[-1] + num_tokens)

        position_ids = torch.cat([
            torch.arange(_cache.length, _cache.length + num_tokens, device=input_ids.device)
            for _cache, num_tokens in zip(past_kv, num_scheduled_tokens)
        ])
        position_ids = torch.clamp(position_ids, max=self.max_seq_len - 1)

        inputs_embeds = self.token_embedding(input_ids) + self.position_embedding(position_ids)

        # 整个 [N] 输入一次投影，没有补齐长度的假 token
        q = self.q_proj(inputs_embeds)
        k = self.k_proj(inputs_embeds)
        v = self.v_proj(inputs_embeds)

        # 整批写入本轮真实 token，长度增量等于各自的 count
        kv_cache_pool.append_batch(past_kv, num_scheduled_tokens, k, v)

        # 每个请求只看自己的历史 KV 和自己的本轮 token
        logits_list = []
        for _cache, start, end in zip(past_kv, offsets[:-1], offsets[1:]):
            past_k, past_v = kv_cache_pool.gather(_cache)
            score = torch.matmul(q[start:end], past_k.transpose(-1, -2)) / (self.d_model ** 0.5)
            query_pos = torch.arange(_cache.length - (end - start), _cache.length, device=input_ids.device).unsqueeze(-1)
            key_pos = torch.arange(_cache.length, device=input_ids.device).unsqueeze(0)
            score = score.masked_fill(key_pos > query_pos, float('-inf'))
            weights = torch.softmax(score, dim=-1)
            logits_list.append(self.lm_head(torch.matmul(weights, past_v)))

        return torch.cat(logits_list, dim=0)


class Scheduler:
    
    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4, block_size=4, enable_prefix_caching=True, on_finished=None, kv_cache_pool: KVCachePool=None):
        
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.running : list[SequenceConfig] = []
        self.waiting : list[SequenceConfig] = []
        self.step_done = []
        self.enable_prefix_caching = enable_prefix_caching
        self.block_size = block_size
        self.on_finished = on_finished
        self.num_scheduled_tokens = []
        self.scheduled_items = []  # 本轮计划：prefill 与 decode 合成一份
        self.kv_cache_pool = kv_cache_pool

    def add_request(self, request):
        seq = SequenceConfig(request["request_id"], request["prompt_ids"], request["max_new_tokens"], self.block_size)
        self.waiting.append(seq)
    
    def has_unfinished_requests(self):
        return len(self.running) + len(self.waiting) > 0
    
    def schedule(self):
        # Fill running with waiting sequences if there's space
        self.scheduled_items = []
        self.step_done = []
        
        for seq in self.waiting:
            if seq.max_new_tokens == 0:
                # Zero budget requests are processed immediately
                if self.on_finished:
                    self.on_finished({"request_id": seq.request_id, "output_ids": []})
                self.step_done.append({"request_id": seq.request_id, "output_ids": []})
        
        self.waiting = [seq for seq in self.waiting if seq.max_new_tokens > 0]
        # Limit the number of batched tokens
        while len(self.running) < self.max_num_seqs and self.waiting:
            next_seq = self.waiting[0]
            if self.kv_cache_pool.allocate_block(next_seq):
                self.waiting.pop(0)
                self.running.append(next_seq)
            else:
                break  # No more blocks available, cannot schedule more sequences
        
        decode_req = [req for req in self.running if req.prefill_len == 0]
        decode_token_budget = len(decode_req)
        assert decode_token_budget <= self.max_num_batched_tokens, "Decode token budget exceeds max_num_batched_tokens"
        prefill_token_budget = self.max_num_batched_tokens - decode_token_budget

        for seq in self.running:
            if seq.prefill_len > 0:
                if prefill_token_budget == 0:
                    continue

                num_scheduled_tokens = min(seq.prefill_len, prefill_token_budget)
                prefill_token_budget -= num_scheduled_tokens
                self.scheduled_items.append({
                    "request": seq,
                    "input_ids": seq.prompt_ids[seq.cache.length:seq.cache.length + num_scheduled_tokens],
                    "num_scheduled_tokens": num_scheduled_tokens,
                    "can_sample": num_scheduled_tokens == seq.prefill_len
                })
            else:
                self.scheduled_items.append({
                    "request": seq,
                    "input_ids": [seq.output_ids[-1]],
                    "num_scheduled_tokens": 1,
                    "can_sample": True
                })

        return self.scheduled_items
    
    def post_step(self):

        for seq in self.running:

            if seq.prefill_len > 0:
                continue  # Prefill not finished yet

            if seq.output_ids[-1] == 4 or len(seq.output_ids) >= seq.max_new_tokens:
                if self.enable_prefix_caching:
                    # 先把可复用的完整 prompt 块登记为缓存，再释放本请求的活动引用
                    self.kv_cache_pool.publish_completed_prompt_blocks(seq)
                if self.on_finished:
                    self.on_finished({"request_id": seq.request_id, "output_ids": seq.output_ids})
                self.step_done.append({"request_id": seq.request_id, "output_ids": seq.output_ids})
                self.kv_cache_pool.deallocate_block(seq)
                seq.cache = None  # Reset past_kv for completed sequences

        self.running = [seq for seq in self.running if len(seq.output_ids) < seq.max_new_tokens and (not seq.output_ids or seq.output_ids[-1] != 4)]

class Engine:
    
    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4,block_size=4, num_kv_blocks=8, vocab_size=5, d_model=8, max_seq_len=32, on_finished=None, enable_prefix_caching=True):
        self.model = TinyCausalLM(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
        self.model.eval()
        self.sampler = Sampler()
        self.enable_prefix_caching = enable_prefix_caching
        self.kv_cache_pool = KVCachePool(block_size, num_kv_blocks, d_model, self.model.device, self.enable_prefix_caching)
        self.scheduler = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, block_size=block_size, enable_prefix_caching=enable_prefix_caching, on_finished=on_finished, kv_cache_pool=self.kv_cache_pool)
        
    def add_request(self, request):
        self.scheduler.add_request(request)
    
    def has_unfinished_requests(self):
        return self.scheduler.has_unfinished_requests()
    
    def _sample(self, logits, scheduled_items):
        # 就绪请求取自己片段 [start:end) 的最后一行 logits，行与请求从同一份计划里对应

        last_rows = []
        offset = 0
        for i, item in enumerate(scheduled_items):
            offset += item["num_scheduled_tokens"]
            if item["can_sample"]:
                last_rows.append((i, offset - 1))

        if not last_rows:
            return None

        rows = torch.tensor([row for _, row in last_rows], device=logits.device, dtype=torch.long)
        output_ids = self.sampler.sample(logits[rows, :])
        for (i, _), output_id in zip(last_rows, output_ids):
            scheduled_items[i]["request"].output_ids.append(output_id.item())
        return None

    def step(self):

        with torch.inference_mode():

            self.scheduler.schedule()

            if not self.scheduler.has_unfinished_requests():
                return self.scheduler.step_done

            scheduled_items = self.scheduler.scheduled_items

            if scheduled_items:
                # 本轮所有真实 token 拼成一维，prefill 与 decode 共用一次模型调用
                input_ids = torch.tensor([token for item in scheduled_items for token in item["input_ids"]], device=self.model.device)
                num_scheduled_tokens = [item["num_scheduled_tokens"] for item in scheduled_items]
                past_kv = [item["request"].cache for item in scheduled_items]

                logits = self.model._forward_append(input_ids, num_scheduled_tokens, past_kv, self.kv_cache_pool)
                self._sample(logits, scheduled_items)

            self.scheduler.post_step()

        return self.scheduler.step_done


if __name__ == "__main__":
    def on_finished(result):
        print("完成通知：", result)

    engine = Engine(max_num_seqs=2, vocab_size=100, d_model=8, max_seq_len=32, on_finished=on_finished)

    # 先收到 A，只让它执行一轮，不要在这里把 A 跑到结束。
    print("提交 A B C")
    engine.add_request({"request_id": "A", "prompt_ids": [0, 1, 2, 3, 4], "max_new_tokens": 4})
    engine.add_request({"request_id": "B", "prompt_ids": [3], "max_new_tokens": 2})
    engine.add_request({"request_id": "C", "prompt_ids": [0, 1], "max_new_tokens": 2})
    
    round_id = 0
    while engine.has_unfinished_requests():
        print(f"step {round_id} 返回：", engine.step())
        round_id += 1

    print("所有请求完成，当前还有未完成请求吗？", engine.has_unfinished_requests())
