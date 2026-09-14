'''
engine = Engine(on_finished=show_result)

engine.add_request(request_a)
engine.step()  # 只推进一轮，然后返回控制权

engine.add_request(request_b)
engine.step()  # A、B 各推进一步

另提供 has_unfinished_requests()，查询是否还有未完成工作。
'''

'''
input:
[
    {"request_id": "A", "prompt_ids": [0], "max_new_tokens": 10},
    {"request_id": "B", "prompt_ids": [3], "max_new_tokens": 10},
    {"request_id": "C", "prompt_ids": [1], "max_new_tokens": 10},
]

output:
[
    {"request_id": "A", "output_ids": [1, 2, 3, 4]},
    {"request_id": "B", "output_ids": [4]},
    {"request_id": "C", "output_ids": [2, 3, 4]},
]
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
        
        
class Engine:
    
    def __init__(self, on_finished=None):
        self.model = DummyModel()
        self.sampler = Sampler()
        self.requests = []
        self.results = []
        self.done = set()
        self.on_finished = on_finished
    
    def add_request(self, request):
        self.requests.append(request)
        self.results.append({"request_id": request["request_id"], "output_ids": []})
    
    def has_unfinished_requests(self):
        return len(self.done) < len(self.requests)
    
    def step(self):
        
        new_done = list()
        
        for i, (req, result) in enumerate(zip(self.requests, self.results)):
            
            if req["request_id"] in self.done:
                continue
            
            if req["max_new_tokens"] == 0:
                if self.on_finished:
                    self.on_finished(result)
                self.done.add(req["request_id"])
                new_done.append(result)
        
        for i, (req, result) in enumerate(zip(self.requests, self.results)):
            request_id = req["request_id"]
            if request_id in self.done:
                continue
            
            prompt_ids = req["prompt_ids"]
            max_new_tokens = req["max_new_tokens"]
            current_result = result["output_ids"]
            
            if current_result:
                prompt_ids = prompt_ids + current_result
                
            output_id = self._generate_one_token(prompt_ids)
            self.results[i]["output_ids"].append(output_id.item())
                
            if output_id.item() == 4 or len(self.results[i]["output_ids"]) >= max_new_tokens:
                if self.on_finished:
                    self.on_finished(self.results[i])
                self.done.add(request_id)
                new_done.append(self.results[i])
        
        return new_done

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
