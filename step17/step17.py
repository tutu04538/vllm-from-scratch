'''
Prefill and decode
'''

from dataclasses import dataclass
import math

from more_itertools import last
import torch
from torch import ceil, nn


@dataclass
class CacheConfig:  
    block_table: list[int] = None # List of block indices in the KV cache
    length: int = 0


class KVCachePool:
    
    def __init__(self, block_size, num_kv_blocks, d_model, device):
        self.block_size = block_size
        self.num_kv_blocks = num_kv_blocks
        self.d_model = d_model
        self.device = device
        self.k_cache = torch.zeros(num_kv_blocks, block_size, d_model, device=device)
        self.v_cache = torch.zeros(num_kv_blocks, block_size, d_model, device=device)
        self.block_usage = [False] * self.num_kv_blocks

    def allocate_block(self, num_blocks_needed=1):
        
        _allocated_blocks = []
        
        for i in range(self.num_kv_blocks):
            if not self.block_usage[i]:
                self.block_usage[i] = True
                _allocated_blocks.append(i)
                if len(_allocated_blocks) == num_blocks_needed:
                    return _allocated_blocks
                
        for idx in _allocated_blocks:
            self.block_usage[idx] = False
        return None

    def deallocate_block(self, block_table):
        
        for idx in block_table:
            self.block_usage[idx] = False

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
    
    def forward_prefill(self, input_ids: torch.Tensor, prefill_lengths: list, past_kv: list[CacheConfig], kv_cache_pool: KVCachePool):
        # inputs_ids: (B, T)
        # prefill_lengths: list of length B, indicating the length of each prefill in the batch
        batch_size, q_max_seq_len = input_ids.shape
        
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(q_max_seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, q_max_seq_len).to(self.device)
        
        start_token_position = torch.tensor([past_kv[i].length for i in range(batch_size)], device=input_ids.device).unsqueeze(-1)
        position_ids = position_ids + start_token_position
        position_ids = torch.clamp(position_ids, max=self.max_seq_len - 1)
        
        position_embeds = self.position_embedding(position_ids)
        
        inputs_embeds = token_embeds + position_embeds
        
        q = self.q_proj(inputs_embeds)
        k = self.k_proj(inputs_embeds)
        v = self.v_proj(inputs_embeds)
        
        # Update the KV cache pool with the new k and v values for each sequence in the batch
        for i in range(batch_size):
            prefill_len = prefill_lengths[i]
            for j in range(prefill_len):
                position = past_kv[i].length + j
                block_idx = position // kv_cache_pool.block_size
                block_offset = position % kv_cache_pool.block_size
                kv_cache_pool.k_cache[past_kv[i].block_table[block_idx]][block_offset] = k[i][j]
                kv_cache_pool.v_cache[past_kv[i].block_table[block_idx]][block_offset] = v[i][j]
            past_kv[i].length += prefill_len
        
        # Construct new_k and new_v tensors for attention computation
        kv_max_seq_len = position_ids.max().item() + 1
        new_k = torch.zeros((batch_size, kv_max_seq_len, self.d_model), device=self.device)
        new_v = torch.zeros((batch_size, kv_max_seq_len, self.d_model), device=self.device)
        
        for i, cache in enumerate(past_kv):
            
            # Retrieve the past key and value tensors from the KV cache pool based on the block_table in the CacheConfig
            last_block_idx = (cache.length - 1) // kv_cache_pool.block_size
            last_block_offset = (cache.length - 1) % kv_cache_pool.block_size
            
            if last_block_idx == 0:
                past_k = kv_cache_pool.k_cache[cache.block_table[0]][:last_block_offset + 1]
                past_v = kv_cache_pool.v_cache[cache.block_table[0]][:last_block_offset + 1]
            else:
                past_k = torch.cat([kv_cache_pool.k_cache[block_idx] for block_idx in cache.block_table[:last_block_idx]], dim=0)
                past_k = torch.cat((past_k, kv_cache_pool.k_cache[cache.block_table[last_block_idx]][:last_block_offset + 1]), dim=0)
                past_v = torch.cat([kv_cache_pool.v_cache[block_idx] for block_idx in cache.block_table[:last_block_idx]], dim=0)
                past_v = torch.cat((past_v, kv_cache_pool.v_cache[cache.block_table[last_block_idx]][:last_block_offset + 1]), dim=0)

            # print(f"past_k shape: {past_k.shape}")
            # print(f"new_k shape: {new_k.shape}")
            new_k[i, :past_k.shape[0], :] = past_k
            new_v[i, :past_v.shape[0], :] = past_v
        
        # Compute the attention scores
        score = torch.matmul(q, new_k.transpose(-1, -2)) / (self.d_model ** 0.5)
        
        # Create a causal mask
        # mask = torch.triu(torch.ones((q_max_seq_len, kv_max_seq_len), device=input_ids.device), diagonal=(kv_max_seq_len - q_max_seq_len + 1))
        mask = torch.ones((batch_size, q_max_seq_len, kv_max_seq_len), device=input_ids.device)
        for i in range(batch_size):
            mask[i] = torch.triu(mask[i], diagonal=(start_token_position[i].item() + 1))
        score = score.masked_fill(mask == 1, float('-inf'))
        weights = torch.softmax(score, dim=-1)
        
        out = torch.matmul(weights, new_v)
        
        # logits: (B, T, vocab_size)
        logits = self.lm_head(out)
    
        return logits
    
    def forward_decode(self, input_ids: torch.Tensor, past_kv: list[CacheConfig], kv_cache_pool: KVCachePool):
        
        assert past_kv is not None, "past_kv must be provided for forward_decode"
        
        batch_size = input_ids.shape[0]
        
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.tensor([[_cache.length] for _cache in past_kv], device=input_ids.device)
        position_embeds = self.position_embedding(position_ids)
        
        inputs_embeds = token_embeds + position_embeds
        
        q = self.q_proj(inputs_embeds)
        k = self.k_proj(inputs_embeds)
        v = self.v_proj(inputs_embeds)

        kv_max_seq_len = position_ids.max().item() + 1  # The new max sequence length after adding the new token
        
        # Update the KV cache pool with the new k and v values for each sequence in the batch
        for i in range(len(past_kv)):
            block_idx = past_kv[i].length // kv_cache_pool.block_size
            block_offset = past_kv[i].length % kv_cache_pool.block_size
            kv_cache_pool.k_cache[past_kv[i].block_table[block_idx]][block_offset] = k[i][0]
            kv_cache_pool.v_cache[past_kv[i].block_table[block_idx]][block_offset] = v[i][0]
            past_kv[i].length += 1
        
        new_k = torch.zeros((batch_size, kv_max_seq_len, self.d_model), device=self.device)
        new_v = torch.zeros((batch_size, kv_max_seq_len, self.d_model), device=self.device)
        
        for i, cache in enumerate(past_kv):
            
            last_block_idx = (cache.length - 1) // kv_cache_pool.block_size
            last_block_offset = (cache.length - 1) % kv_cache_pool.block_size
            
            if last_block_idx == 0:
                past_k = kv_cache_pool.k_cache[cache.block_table[0]][:last_block_offset + 1]
                past_v = kv_cache_pool.v_cache[cache.block_table[0]][:last_block_offset + 1]
            else:
                past_k = torch.cat([kv_cache_pool.k_cache[block_idx] for block_idx in cache.block_table[:last_block_idx]], dim=0)
                past_k = torch.cat((past_k, kv_cache_pool.k_cache[cache.block_table[last_block_idx]][:last_block_offset + 1]), dim=0)
                past_v = torch.cat([kv_cache_pool.v_cache[block_idx] for block_idx in cache.block_table[:last_block_idx]], dim=0)
                past_v = torch.cat((past_v, kv_cache_pool.v_cache[cache.block_table[last_block_idx]][:last_block_offset + 1]), dim=0)

            new_k[i, :past_k.shape[0], :] = past_k
            new_v[i, :past_v.shape[0], :] = past_v
     
        score = torch.matmul(q, new_k.transpose(-1, -2)) / (self.d_model ** 0.5)
        
        positions = torch.arange(
            kv_max_seq_len,
            device=input_ids.device,
        ).view(1, 1, kv_max_seq_len)
        mask = positions > position_ids.unsqueeze(-1)
        
        score = score.masked_fill(mask, float('-inf'))

        weights = torch.softmax(score, dim=-1)
        out = torch.matmul(weights, new_v)

        logits = self.lm_head(out)

        return logits
    

class SequenceConfig:
    def __init__(self, request_id, prompt_ids, max_new_tokens):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.max_new_tokens = max_new_tokens
        self.output_ids = []
        self.cache = CacheConfig()

    @property
    def prefill_len(self):
        return max(len(self.prompt_ids) - self.cache.length, 0)


class Engine:
    
    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4,block_size=4, num_kv_blocks=8, vocab_size=5, d_model=8, max_seq_len=32, on_finished=None):
        self.model = TinyCausalLM(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
        self.model.eval()
        self.sampler = Sampler()
        self.on_finished = on_finished
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.running = []
        self.waiting = []
        self.step_done = []
        self.kv_cache_pool = KVCachePool(block_size, num_kv_blocks, d_model, self.model.device)
        
    def add_request(self, request):
        seq = SequenceConfig(request["request_id"], request["prompt_ids"], request["max_new_tokens"])
        self.waiting.append(seq)

    def has_unfinished_requests(self):
        return len(self.running) + len(self.waiting) > 0
    
    def _schedule(self):
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
            next_seq.cache.block_table = self.kv_cache_pool.allocate_block(math.ceil((len(next_seq.prompt_ids) + next_seq.max_new_tokens - 1) / self.kv_cache_pool.block_size))
            if next_seq.cache.block_table is not None:
                self.waiting.pop(0)
                self.running.append(next_seq)
            else:
                break  # No more blocks available, cannot schedule more sequences
    
    def step(self):
        
        self.step_done = []
        self._schedule()
        
        if not self.running:
            return self.step_done
        
        with torch.inference_mode():
        
            # logits = torch.empty(len(self.running), self.model.vocab_size, device=self.model.device)
            ready_sample_reqs_num = 0

            prefill_req = []
            decode_req = []

            for i, req in enumerate(self.running):
                if req.prefill_len > 0:
                    prefill_req.append(req)
                else:
                    decode_req.append(req)
                    ready_sample_reqs_num += 1

            decode_token_budget = len(decode_req)
            assert decode_token_budget <= self.max_num_batched_tokens, "Decode token budget exceeds max_num_batched_tokens"
            prefill_token_budget = self.max_num_batched_tokens - decode_token_budget
            
            # Prefill
            prefill_input_ids = []
            prefill_past_kv = []
            prefill_idx = []
            for i, req in enumerate(prefill_req):
                if prefill_token_budget == 0:
                    break
                if prefill_token_budget >= req.prefill_len:
                    prefill_input_ids.append(req.prompt_ids[req.cache.length:])
                    prefill_past_kv.append(req.cache)
                    prefill_token_budget -= req.prefill_len
                    prefill_idx.append(i)
                else:
                    prefill_input_ids.append(req.prompt_ids[req.cache.length:req.cache.length + prefill_token_budget])
                    prefill_past_kv.append(req.cache)
                    prefill_token_budget = 0
            
            ready_sample_reqs_num += len(prefill_idx)

            # Decode
            decode_input_ids = []
            decode_past_kv = []
            
            for req in decode_req:
                current_result = req.output_ids
                past_kv = req.cache
                decode_past_kv.append(past_kv)
                decode_input_ids.append([current_result[-1]])
            
            logits = torch.empty(ready_sample_reqs_num, self.model.vocab_size, device=self.model.device)

            # prefill generation
            if prefill_input_ids:
                prefill_lengths = [len(ids) for ids in prefill_input_ids]
                max_prefill_len = max(len(ids) for ids in prefill_input_ids)
                padded_input_ids = [ids + [0] * (max_prefill_len - len(ids)) for ids in prefill_input_ids]
                padded_input_ids = torch.tensor(padded_input_ids, device=self.model.device)
                position = torch.tensor([prefill_lengths[idx] for idx in prefill_idx], device=self.model.device, dtype=torch.long) - 1
                _logits = self._generate_prefill(padded_input_ids, prefill_lengths, prefill_past_kv)
                logits[len(decode_input_ids):] = _logits[torch.tensor(prefill_idx, device=self.model.device, dtype=torch.long), position, :]
            
            # decode generation
            if decode_input_ids:
                decode_input_ids = torch.tensor(decode_input_ids, device=self.model.device)
                print("decode_input_ids:", decode_input_ids)
                _logits = self._generate_decode(decode_input_ids, decode_past_kv)
                logits[:decode_input_ids.shape[0]] = _logits[:, -1, :]
            
            output_ids = []
            if ready_sample_reqs_num > 0:
                output_ids = self.sampler.sample(logits)
        
        for i, req in enumerate(self.running):
                
            request_id = req.request_id
            max_new_tokens = req.max_new_tokens
            
            if req.prefill_len > 0:
                continue  # Prefill not finished yet
            output_id = output_ids[i]
            req.output_ids.append(output_id.item())
            self.running[i] = req
        
            if output_id.item() == 4 or len(req.output_ids) >= max_new_tokens:
                if self.on_finished:
                    self.on_finished({"request_id": request_id, "output_ids": req.output_ids})
                self.step_done.append({"request_id": request_id, "output_ids": req.output_ids})
                self.kv_cache_pool.deallocate_block(req.cache.block_table)
                req.cache = None  # Reset past_kv for completed sequences
    
        self.running = [seq for seq in self.running if len(seq.output_ids) < seq.max_new_tokens and (not seq.output_ids or seq.output_ids[-1] != 4)]
        
        return self.step_done

    def _generate_prefill(self, input_ids, prefill_lengths, past_kv):
        if type(input_ids) != torch.Tensor:
            input_ids = torch.tensor(input_ids, device=self.model.device)
        else:
            input_ids = input_ids.clone().to(self.model.device)

        with torch.inference_mode():
            logits = self.model.forward_prefill(input_ids, prefill_lengths=prefill_lengths, past_kv=past_kv, kv_cache_pool=self.kv_cache_pool)
            return logits

    def _generate_decode(self, input_ids, past_kv):
        if type(input_ids) != torch.Tensor:
            input_ids = torch.tensor(input_ids, device=self.model.device)
        else:
            input_ids = input_ids.clone().to(self.model.device)

        with torch.inference_mode():
            logits = self.model.forward_decode(input_ids, past_kv=past_kv, kv_cache_pool=self.kv_cache_pool)
            return logits



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
