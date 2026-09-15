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
    
    def forward_prefill(self, input_ids: torch.Tensor, prompt_lengths: list, past_kv: list[CacheConfig], kv_cache_pool: KVCachePool):
        # inputs_ids: (B, T)
        # prompt_lengths: list of length B, indicating the length of each prompt in the batch
        batch_size, max_seq_len = input_ids.shape
        
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(max_seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, max_seq_len).to(self.device)
        position_embeds = self.position_embedding(position_ids)
        
        inputs_embeds = token_embeds + position_embeds
        
        q = self.q_proj(inputs_embeds)
        k = self.k_proj(inputs_embeds)
        v = self.v_proj(inputs_embeds)
        
        score = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)
        
        mask = torch.triu(torch.ones((max_seq_len, max_seq_len), device=input_ids.device), diagonal=1)
        score = score.masked_fill(mask == 1, float('-inf'))
        weights = torch.softmax(score, dim=-1)
        
        out = torch.matmul(weights, v)
        
        # logits: (B, T, vocab_size)
        logits = self.lm_head(out)
        
        for i in range(batch_size):
            prompt_len = prompt_lengths[i]
            for j in range(prompt_len):
                block_idx = j // kv_cache_pool.block_size
                block_offset = j % kv_cache_pool.block_size
                kv_cache_pool.k_cache[past_kv[i].block_table[block_idx]][block_offset] = k[i][j]
                kv_cache_pool.v_cache[past_kv[i].block_table[block_idx]][block_offset] = v[i][j]
            past_kv[i].length = prompt_len
                
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

        current_max_seq_len = position_ids.max().item() + 1  # The new max sequence length after adding the new token
        
        for i in range(len(past_kv)):
            block_idx = past_kv[i].length // kv_cache_pool.block_size
            block_offset = past_kv[i].length % kv_cache_pool.block_size
            kv_cache_pool.k_cache[past_kv[i].block_table[block_idx]][block_offset] = k[i][0]
            kv_cache_pool.v_cache[past_kv[i].block_table[block_idx]][block_offset] = v[i][0]
            past_kv[i].length += 1
        
        new_k = torch.zeros((batch_size, current_max_seq_len, self.d_model), device=self.device)
        new_v = torch.zeros((batch_size, current_max_seq_len, self.d_model), device=self.device)
        
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
            current_max_seq_len,
            device=input_ids.device,
        ).view(1, 1, current_max_seq_len)
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

    
class Engine:
    
    def __init__(self, max_num_seqs=1, block_size=4, num_kv_blocks=8, vocab_size=5, d_model=8, max_seq_len=32, on_finished=None):
        self.model = TinyCausalLM(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
        self.model.eval()
        self.sampler = Sampler()
        self.on_finished = on_finished
        self.max_num_seqs = max_num_seqs
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
        
            logits = torch.empty(len(self.running), self.model.vocab_size, device=self.model.device)

            prefill_req = []
            decode_req = []

            for i, req in enumerate(self.running):
                if req.cache.length == 0:
                    prefill_req.append(req)
                else:
                    decode_req.append(req)

            # Prefill
            input_ids = []
            prefill_past_kv = []
            for req in prefill_req:
                input_ids.append(req.prompt_ids)
                prefill_past_kv.append(req.cache)

            if input_ids:
                max_prompt_len = max(len(ids) for ids in input_ids)
                padded_input_ids = [ids + [0] * (max_prompt_len - len(ids)) for ids in input_ids]
                padded_input_ids = torch.tensor(padded_input_ids, device=self.model.device)
                prompt_lengths = [len(ids) for ids in input_ids]
                position = torch.tensor(prompt_lengths, device=self.model.device) - 1
                _logits = self._generate_prefill(padded_input_ids, prompt_lengths, prefill_past_kv)
                logits[-len(input_ids):] = _logits[torch.arange(_logits.size(0), device=self.model.device), position, :]
            
            # Decode
            decode_input_ids = []
            decode_past_kv = []
            
            for req in decode_req:
                current_result = req.output_ids
                past_kv = req.cache
                decode_past_kv.append(past_kv)
                decode_input_ids.append([current_result[-1]])

            if decode_input_ids:
                decode_input_ids = torch.tensor(decode_input_ids, device=self.model.device)
                print("decode_input_ids:", decode_input_ids)
                _logits = self._generate_decode(decode_input_ids, decode_past_kv)
                logits[:decode_input_ids.shape[0]] = _logits[:, -1, :]
            
            output_ids = self.sampler.sample(logits)
        
        for i, req in enumerate(self.running):
                
            request_id = req.request_id
            max_new_tokens = req.max_new_tokens
            
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

    def _generate_prefill(self, input_ids, prompt_lengths, past_kv):
        if type(input_ids) != torch.Tensor:
            input_ids = torch.tensor(input_ids, device=self.model.device)
        else:
            input_ids = input_ids.clone().to(self.model.device)

        with torch.inference_mode():
            logits = self.model.forward_prefill(input_ids, prompt_lengths=prompt_lengths, past_kv=past_kv, kv_cache_pool=self.kv_cache_pool)
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
    engine.add_request({"request_id": "A", "prompt_ids": [0, 1, 2], "max_new_tokens": 4})
    engine.add_request({"request_id": "B", "prompt_ids": [3], "max_new_tokens": 2})
    engine.add_request({"request_id": "C", "prompt_ids": [0, 1], "max_new_tokens": 2})
    
    round_id = 0
    while engine.has_unfinished_requests():
        print(f"step {round_id} 返回：", engine.step())
        round_id += 1

    print("所有请求完成，当前还有未完成请求吗？", engine.has_unfinished_requests())
