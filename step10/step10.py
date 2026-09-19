'''
model = TinyCausalLM(vocab_size=5, d_model=8, max_seq_len=32)
logits = model(input_ids)
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
        
    
    def forward(self, input_ids: torch.Tensor, past_kv=None):
        batch_size, seq_len = input_ids.shape
        
        token_embeds = self.token_embedding(input_ids)
        past_length = past_kv[0].shape[1] if past_kv is not None else 0
        position_ids = torch.arange(seq_len, device=input_ids.device) + past_length
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)
        position_embeds = self.position_embedding(position_ids)
        
        inputs_embeds = token_embeds + position_embeds
        
        q = self.q_proj(inputs_embeds)
        k = self.k_proj(inputs_embeds)
        v = self.v_proj(inputs_embeds)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        score = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)
        
        mask = torch.triu(torch.ones((seq_len, seq_len), device=input_ids.device), diagonal=1) if past_kv is None else torch.zeros((1, seq_len), device=input_ids.device)
        score = score.masked_fill(mask == 1, float('-inf'))
        weights = torch.softmax(score, dim=-1)
        
        out = torch.matmul(weights, v)
        
        logits = self.lm_head(out)
        
        return logits, (k, v)

    

class SequenceConfig:
    def __init__(self, request_id, prompt_ids, max_new_tokens):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.max_new_tokens = max_new_tokens
        self.output_ids = []

class Engine:
    
    def __init__(self, max_num_seqs=2, vocab_size=5, d_model=8, max_seq_len=32, on_finished=None):
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
    
    def _post_step(self):
        self.running = [seq for seq in self.running if len(seq.output_ids) < seq.max_new_tokens and (not seq.output_ids or seq.output_ids[-1] != 4)]
    
    def step(self):
        
        self.step_done = []
        
        self._schedule()
        
        if not self.running:
            return self.step_done
        
        input_ids = []
        
        for i, req in enumerate(self.running):
            
            request_id = req.request_id          
            prompt_ids = req.prompt_ids
            max_new_tokens = req.max_new_tokens
            current_result = req.output_ids
    
            input_ids.append(prompt_ids + current_result)
        
        max_seq_len = max(len(ids) for ids in input_ids)
        attention_mask = [[1] * len(ids) + [0] * (max_seq_len - len(ids)) for ids in input_ids]
        input_ids = [ids + [0] * (max_seq_len - len(ids)) for ids in input_ids]
        
        output_ids = self._generate(input_ids, attention_mask)
        
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
    
        self._post_step()
        
        return self.step_done

    def _generate(self, input_ids, attention_mask):
        if type(input_ids) != torch.Tensor:
            input_ids = torch.tensor(input_ids, device=self.model.device)
        else:
            input_ids = input_ids.clone().to(self.model.device)
        
        if type(attention_mask) != torch.Tensor:
            attention_mask = torch.tensor(attention_mask, device=self.model.device)
        else:
            attention_mask = attention_mask.clone().to(self.model.device)

        with torch.inference_mode():
            logits = self.model.forward(input_ids)
            last_token_position = attention_mask.sum(dim=-1) - 1
            batch_indices = torch.arange(logits.size(0), device=self.model.device)
            last_token_logits = logits[batch_indices, last_token_position]
            next_token_id = self.sampler.sample(last_token_logits)
        
            return next_token_id

if __name__ == "__main__":
    def on_finished(result):
        print("完成通知：", result)

    engine = Engine(on_finished=on_finished)

    # 先收到 A，只让它执行一轮，不要在这里把 A 跑到结束。
    print("提交 A")
    engine.add_request({"request_id": "A", "prompt_ids": [0], "max_new_tokens": 10})
    print("step 1 返回：", engine.step())

    # 现在控制权回到了 main。模拟 B 此时才到来。
    print("A 尚未完成，现在提交 B")
    engine.add_request({"request_id": "B", "prompt_ids": [3], "max_new_tokens": 10})

    # 本次演示不再有新请求，继续推进已提交的请求，全部完成后退出。
    round_id = 2
    while engine.has_unfinished_requests():
        print(f"step {round_id} 返回：", engine.step())
        round_id += 1

    print("所有请求完成，当前还有未完成请求吗？", engine.has_unfinished_requests())
