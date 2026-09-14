'''
  新增 step06.py：

  engine = Engine(max_num_seqs=2, on_finished=show_result)

  同时提交 A、B、C：

  step 1：运行 A、B；C 等待。B 完成。
  step 2：运行 A、C；C 补位。

  核心要求：

  - 运行请求保留名额，直到完成。
  - 等待请求按提交顺序补位。
  - 本轮释放的名额，下一轮才补。
  - 零预算请求不占名额，仍优先交付。

  这次要解决的问题是：

  > C 已经提交，却暂时不能执行，应把它保存在哪里？
'''

import torch


class Sampler:
    
    def __init__(self):
        pass
    
    def sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1).unsqueeze(0)


class DummyModel:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = 5
        self.next_token_logits = torch.tensor([[0, 10, 0, 0, 0],
                                               [0, 0, 10, 0, 0],
                                               [0, 0, 0, 10, 0],
                                               [0, 0, 0, 0, 10],
                                               [0, 0, 0, 0, 10]], device=self.device, dtype=torch.float32)
        
    def forward(self, input_ids):
        return self.next_token_logits[input_ids[-1]]
        

class SequenceConfig:
    def __init__(self, request_id, prompt_ids, max_new_tokens):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.max_new_tokens = max_new_tokens
        self.output_ids = []

class Engine:
    
    def __init__(self, max_num_seqs=2, on_finished=None):
        self.model = DummyModel()
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
        
        for i, req in enumerate(self.running):
            
            if req is None:
                continue
            
            request_id = req.request_id          
            prompt_ids = req.prompt_ids
            max_new_tokens = req.max_new_tokens
            current_result = req.output_ids
            
            if current_result:
                prompt_ids = prompt_ids + current_result
                
            output_id = self._generate_one_token(prompt_ids)
            req.output_ids.append(output_id.item())
            self.running[i] = req
            
            if output_id.item() == 4 or len(req.output_ids) >= max_new_tokens:
                if self.on_finished:
                    self.on_finished({"request_id": request_id, "output_ids": req.output_ids})
                self.step_done.append({"request_id": request_id, "output_ids": req.output_ids})
        
        self._post_step()
        
        return self.step_done

    def _generate_one_token(self, prompt_ids):
        if type(prompt_ids) != torch.Tensor:
            generated_ids = torch.tensor(prompt_ids, device=self.model.device)
        else:
            generated_ids = prompt_ids.clone().to(self.model.device)
        
        logits = self.model.forward(generated_ids)
        next_token_id = self.sampler.sample(logits)
        
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
