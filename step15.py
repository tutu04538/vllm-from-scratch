'''
Prefill and decode
'''

import torch
from torch import nn

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
    
    def forward_prefill(self, input_ids: torch.Tensor, prompt_lengths: list, past_kv):
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
            past_kv[i]['k'][:prompt_len] = k[i, :prompt_len]
            past_kv[i]['v'][:prompt_len] = v[i, :prompt_len]
            past_kv[i]['length'] = prompt_len
        
        return logits
    
    def forward_decode(self, input_ids: torch.Tensor, past_kv):
        
        assert past_kv is not None, "past_kv must be provided for forward_decode"
        
        batch_size = input_ids.shape[0]
        
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.tensor([[_cache['length']] for _cache in past_kv], device=input_ids.device)
        position_embeds = self.position_embedding(position_ids)
        
        inputs_embeds = token_embeds + position_embeds
        
        q = self.q_proj(inputs_embeds)
        k = self.k_proj(inputs_embeds)
        v = self.v_proj(inputs_embeds)

        current_max_seq_len = position_ids.max().item() + 1  # The new max sequence length after adding the new token
        
        for i in range(len(past_kv)):
            past_kv[i]['k'][past_kv[i]['length']] = k[i][0]
            past_kv[i]['v'][past_kv[i]['length']] = v[i][0]
            past_kv[i]['length'] += 1  
        
        new_k = torch.stack([torch.cat((_cache['k'][:_cache['length']], torch.zeros(current_max_seq_len - _cache['length'], self.d_model, device=self.device)), dim=0) for _cache in past_kv], dim=0)
        new_v = torch.stack([torch.cat((_cache['v'][:_cache['length']], torch.zeros(current_max_seq_len - _cache['length'], self.d_model, device=self.device)), dim=0) for _cache in past_kv], dim=0)
        
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
        self.cache = None  # To store past_kv for this sequence

class Engine:
    
    def __init__(self, max_num_seqs=1, vocab_size=5, d_model=8, max_seq_len=32, on_finished=None):
        self.model = TinyCausalLM(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
        self.model.eval()
        self.sampler = Sampler()
        self.on_finished = on_finished
        self.max_num_seqs = max_num_seqs
        self.running = []
        self.waiting = []
        self.step_done = []
        
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
            next_seq = self.waiting.pop(0)
            if next_seq.max_new_tokens > 0:
                self.running.append(next_seq)        
    
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
                if req.cache is None:
                    prefill_req.append(req)
                else:
                    decode_req.append(req)

            # Prefill
            input_ids = []
            prefill_past_kv = []
            for req in prefill_req:
                req.cache = {
                    "k": torch.empty(self.model.max_seq_len, self.model.d_model, device=self.model.device),
                    "v": torch.empty(self.model.max_seq_len, self.model.d_model, device=self.model.device),
                    "length": 0
                }
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
                req.cache = None  # Reset past_kv for completed sequences
    
        self.running = [seq for seq in self.running if len(seq.output_ids) < seq.max_new_tokens and (not seq.output_ids or seq.output_ids[-1] != 4)]
        
        return self.step_done

    def _generate_prefill(self, input_ids, prompt_lengths, past_kv):
        if type(input_ids) != torch.Tensor:
            input_ids = torch.tensor(input_ids, device=self.model.device)
        else:
            input_ids = input_ids.clone().to(self.model.device)

        with torch.inference_mode():
            logits = self.model.forward_prefill(input_ids, prompt_lengths=prompt_lengths, past_kv=past_kv)
            return logits

    def _generate_decode(self, input_ids, past_kv):
        if type(input_ids) != torch.Tensor:
            input_ids = torch.tensor(input_ids, device=self.model.device)
        else:
            input_ids = input_ids.clone().to(self.model.device)

        with torch.inference_mode():
            logits = self.model.forward_decode(input_ids, past_kv=past_kv)
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
