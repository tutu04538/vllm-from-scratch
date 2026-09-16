'''
Prefill and decode
'''

from dataclasses import dataclass
import math

from more_itertools import last
import torch
from torch import ceil, nn

import hashlib


def _stable_hash(previous_hash: bytes, block: tuple[int, ...]) -> bytes:
    data = previous_hash + bytes(block)
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
        self.all_ids = prompt_ids
        self.cache = CacheConfig()
        self.block_size = block_size  # Size of each block in the KV cache
        self.block_hashes = []  # List of hashes for each block in the sequence
        self.new_hashes = []  # List of new hashes generated during the current step

    @property
    def prefill_len(self):
        return max(len(self.prompt_ids) - self.cache.length, 0)

    def generate_new_block_hashes(self, num_scheduled_tokens):
        previous_hash = self.block_hashes[-1] if self.block_hashes else b""
        block_start = self.cache.length // self.block_size
        block_end = (self.cache.length + num_scheduled_tokens) // self.block_size
        for i in range(block_start, block_end):
            block = tuple(self.all_ids[i * self.block_size:(i + 1) * self.block_size])
            previous_hash = _stable_hash(previous_hash, block)
            self.new_hashes.append((previous_hash, self.cache.block_table[:i+1]))
    
    def update_block_hashes(self):
        self.block_hashes.extend([hash_tuple[0] for hash_tuple in self.new_hashes])
        self.new_hashes = []


class KVCachePool:
    
    def __init__(self, block_size, num_kv_blocks, d_model, device, enable_prefix_caching=True):
        self.block_size = block_size
        self.num_kv_blocks = num_kv_blocks
        self.d_model = d_model
        self.device = device
        self.k_cache = torch.zeros(num_kv_blocks, block_size, d_model, device=device)
        self.v_cache = torch.zeros(num_kv_blocks, block_size, d_model, device=device)
        self.block_usage = [0] * self.num_kv_blocks
        self.enable_prefix_caching = enable_prefix_caching
        if self.enable_prefix_caching:
            self.block_hash = {}  

    def allocate_block(self, seq: SequenceConfig):
        
        _allocated_blocks = []
        num_blocks_needed = ceil((len(seq.prompt_ids) + seq.max_new_tokens - 1) / self.block_size)
        
        if self.enable_prefix_caching:
            matched_prefix_blocks = self.find_matched_prefix_blocks(seq.prompt_ids[:-1])
            if len(matched_prefix_blocks) > 0:
                _allocated_blocks.extend(matched_prefix_blocks)
                for idx in matched_prefix_blocks:
                    self.block_usage[idx] += 1
                num_blocks_needed -= len(matched_prefix_blocks)
                seq.cache.length = len(matched_prefix_blocks) * self.block_size
                
        for i in range(self.num_kv_blocks):
            if self.block_usage[i] == 0:
                self.block_usage[i] = 1
                _allocated_blocks.append(i)
                if len(_allocated_blocks) == num_blocks_needed:
                    seq.cache.block_table = _allocated_blocks
                    return True
        
        # If we reach here, it means we couldn't allocate enough blocks
        for idx in _allocated_blocks:
            self.block_usage[idx] -= 1
        seq.cache.block_table = None
        seq.cache.length = 0
        return False

    def deallocate_block(self, block_table):
        # Deallocate the blocks in the block_table
        for idx in block_table:
            self.block_usage[idx] -= 1
            
    def append(self, cache: CacheConfig, new_k, new_v):
        # Append new_k and new_v to the KV cache based on the block_table in the CacheConfig
        
        for i in range(new_k.shape[0]):
            position = cache.length + i
            block_idx = position // self.block_size
            block_offset = position % self.block_size
            self.k_cache[cache.block_table[block_idx]][block_offset] = new_k[i]
            self.v_cache[cache.block_table[block_idx]][block_offset] = new_v[i]
        
        cache.length += new_k.shape[0]
        
    def gather(self, cache: CacheConfig):
        # Gather the k and v tensors from the KV cache based on the block_table in the CacheConfig
        
        if cache.length == 0:
            return torch.empty((0, self.d_model), device=self.device), torch.empty((0, self.d_model), device=self.device)
        
        k_list = []
        v_list = []
        for i in range(cache.length):
            block_idx = i // self.block_size
            block_offset = i % self.block_size
            k_list.append(self.k_cache[cache.block_table[block_idx]][block_offset])
            v_list.append(self.v_cache[cache.block_table[block_idx]][block_offset])
        return torch.stack(k_list, dim=0), torch.stack(v_list, dim=0)


    def find_matched_prefix_blocks(self, prompt_ids):
        
        matched_blocks = []
        block_num = len(prompt_ids) // self.block_size
        current_hash = b""
        for i in range(0, block_num):
            block = tuple(prompt_ids[i * self.block_size:(i + 1) * self.block_size])
            current_hash = self._stable_hash(current_hash, block)
            if current_hash not in self.block_hash:
                break
            matched_blocks = self.block_hash[current_hash]

        return matched_blocks

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
        # input_ids: (B, T)
        # past_kv: list of CacheConfig for each sequence in the batch
        batch_size, seq_len = input_ids.shape
        
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.clamp(torch.tensor([[_cache.length + i for i in range(seq_len)] for _cache in past_kv], device=input_ids.device), max=self.max_seq_len - 1)
        position_embeds = self.position_embedding(position_ids)
        
        inputs_embeds = token_embeds + position_embeds
        
        q = self.q_proj(inputs_embeds)
        k = self.k_proj(inputs_embeds)
        v = self.v_proj(inputs_embeds)
        
        kv_max_seq_len = position_ids.max().item() + 1  # The new max sequence length after adding the new token
        
        new_k = torch.zeros((batch_size, kv_max_seq_len, self.d_model), device=self.device)
        new_v = torch.zeros((batch_size, kv_max_seq_len, self.d_model), device=self.device)
        
        for batch_idx, (_cache, _k, _v) in enumerate(zip(past_kv, k, v)):
            past_k, past_v = kv_cache_pool.gather(_cache)
            new_k[batch_idx, :, :] = torch.cat([past_k, _k[:num_scheduled_tokens[batch_idx]], torch.zeros([kv_max_seq_len - past_k.shape[0] - num_scheduled_tokens[batch_idx], self.d_model], device=self.device)], dim=0)
            new_v[batch_idx, :, :] = torch.cat([past_v, _v[:num_scheduled_tokens[batch_idx]], torch.zeros([kv_max_seq_len - past_v.shape[0] - num_scheduled_tokens[batch_idx], self.d_model], device=self.device)], dim=0)

        # Update the KV cache pool with the new k and v values for each sequence in the batch
        for batch_idx, (_cache, _k, _v) in enumerate(zip(past_kv, k, v)):
            kv_cache_pool.append(_cache, _k[:num_scheduled_tokens[batch_idx]], _v[:num_scheduled_tokens[batch_idx]])
            
        score = torch.matmul(q, new_k.transpose(-1, -2)) / (self.d_model ** 0.5)
        
        mask = torch.ones((batch_size, seq_len, kv_max_seq_len), device=input_ids.device)
        for i in range(batch_size):
            mask[i] = torch.triu(mask[i], diagonal=(position_ids[i, 0] + 1))
    
        score = score.masked_fill(mask == 1, float('-inf'))
        weights = torch.softmax(score, dim=-1)
        out = torch.matmul(weights, new_v)
        
        logits = self.lm_head(out)
            
        return logits
    

class Scheduler:
    
    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4, block_size=4, enable_prefix_caching=True, on_finished=None):
        
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.running : list[SequenceConfig] = []
        self.waiting : list[SequenceConfig] = []
        self.step_done = []
        self.enable_prefix_caching = enable_prefix_caching
        self.block_size = block_size
        self.on_finished = on_finished
        self.num_scheduled_tokens = []

    def add_request(self, request):
        seq = SequenceConfig(request["request_id"], request["prompt_ids"], request["max_new_tokens"], self.block_size)
        self.waiting.append(seq)
    
    def has_unfinished_requests(self):
        return len(self.running) + len(self.waiting) > 0
    
    def schedule(self, kv_cache_pool: KVCachePool):
        # Fill running with waiting sequences if there's space
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
            if kv_cache_pool.allocate_block(next_seq):
                self.waiting.pop(0)
                self.running.append(next_seq)
            else:
                break  # No more blocks available, cannot schedule more sequences
            
        prefill_scheduled_items = []
        decode_scheduled_items = []
        
        decode_req = [req for req in self.running if req.prefill_len == 0]
        decode_token_budget = len(decode_req)
        assert decode_token_budget <= self.max_num_batched_tokens, "Decode token budget exceeds max_num_batched_tokens"
        prefill_token_budget = self.max_num_batched_tokens - decode_token_budget

        for i, req in enumerate(self.running):
            if req.prefill_len > 0:
                if prefill_token_budget == 0:
                    continue
                
                if prefill_token_budget >= req.prefill_len:
                    num_scheduled_tokens = req.prefill_len
                    prefill_scheduled_items.append({
                        "request": req,
                        "input_ids": req.prompt_ids[req.cache.length:],
                        "num_scheduled_tokens": num_scheduled_tokens,
                        "can_sample": True
                    })
                    prefill_token_budget -= num_scheduled_tokens
                    req.generate_new_block_hashes(num_scheduled_tokens)
                else:
                    num_scheduled_tokens = prefill_token_budget
                    prefill_scheduled_items.append({
                        "request": req,
                        "input_ids": req.prompt_ids[req.cache.length:req.cache.length + prefill_token_budget],
                        "num_scheduled_tokens": num_scheduled_tokens,
                        "can_sample": False
                    })
                    prefill_token_budget = 0
                    req.generate_new_block_hashes(num_scheduled_tokens)
            else:
                decode_scheduled_items.append({
                    "request": req,
                    "input_ids": [req.output_ids[-1]],
                    "num_scheduled_tokens": 1,
                    "can_sample": True
                })
                req.generate_new_block_hashes(1)
        
        return prefill_scheduled_items, decode_scheduled_items

class Engine:
    
    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4,block_size=4, num_kv_blocks=8, vocab_size=5, d_model=8, max_seq_len=32, on_finished=None, enable_prefix_caching=True):
        self.model = TinyCausalLM(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
        self.model.eval()
        self.sampler = Sampler()
        self.enable_prefix_caching = enable_prefix_caching
        self.kv_cache_pool = KVCachePool(block_size, num_kv_blocks, d_model, self.model.device, self.enable_prefix_caching)
        self.scheduler = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, block_size=block_size, enable_prefix_caching=enable_prefix_caching, on_finished=on_finished)
        
    def add_request(self, request):
        self.scheduler.add_request(request)
    
    def step(self):
        

        with torch.inference_mode():
            
            prefill_scheduled_items, decode_scheduled_items = self.scheduler.schedule()
            
            if not self.scheduler.has_unfinished_requests():
                return self.scheduler.step_done
            
            def _forward_and_sample(scheduled_items):
                num_scheduled_tokens = [item["num_scheduled_tokens"] for item in scheduled_items]
                max_scheduled_tokens = max(num_scheduled_tokens) if num_scheduled_tokens else 0
                input_ids = [item["input_ids"] + [0] * (max_scheduled_tokens - len(item["input_ids"])) for item in scheduled_items]
                past_kv = [item["request"].cache for item in scheduled_items]
                
                _logits = self.model._forward_append(torch.tensor(input_ids, device=self.model.device), num_scheduled_tokens, past_kv, self.kv_cache_pool)
                
                ready_sample_idx = torch.tensor([i for i, item in enumerate(scheduled_items) if item["can_sample"]], device=self.model.device, dtype=torch.long)
                
                if ready_sample_idx.shape[0] == 0:
                    return None
                
                ready_sample_items = [scheduled_items[i] for i in ready_sample_idx.tolist()]
                last_token_positions = torch.tensor([item["num_scheduled_tokens"] - 1 for item in ready_sample_items], device=self.model.device, dtype=torch.long)
                logits = _logits[ready_sample_idx, last_token_positions, :]
                output_ids = self.sampler.sample(logits)
                for item, output_id in zip(ready_sample_items, output_ids):
                    item["request"].output_ids.append(output_id.item())
                return None

            if prefill_scheduled_items:
                _ = _forward_and_sample(prefill_scheduled_items)
            if decode_scheduled_items:
                _ = _forward_and_sample(decode_scheduled_items)
        
        for i, req in enumerate(self.running):
                
            request_id = req.request_id
            max_new_tokens = req.max_new_tokens
            
            if req.prefill_len > 0:
                continue  # Prefill not finished yet
            
            if req.output_ids[-1] == 4 or len(req.output_ids) >= max_new_tokens:
                if self.on_finished:
                    self.on_finished({"request_id": request_id, "output_ids": req.output_ids})
                self.step_done.append({"request_id": request_id, "output_ids": req.output_ids})
                self.kv_cache_pool.deallocate_block(req.cache.block_table)
                req.cache = None  # Reset past_kv for completed sequences
    
        self.running = [seq for seq in self.running if len(seq.output_ids) < seq.max_new_tokens and (not seq.output_ids or seq.output_ids[-1] != 4)]
        
        return self.step_done


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
