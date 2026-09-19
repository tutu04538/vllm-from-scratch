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
        
        

model = DummyModel()
sampler = Sampler()

def generate_one_token(prompt_ids):
    
    if type(prompt_ids) != torch.Tensor:
        generated_ids = torch.tensor(prompt_ids, device=model.device)
    else:
        generated_ids = prompt_ids.clone().to(model.device)
    
    logits = model.forward(generated_ids)
    next_token_id = sampler.sample(logits)
    
    return next_token_id


def generate_many(requests):
    results = []
    done = set()
    
    for req in requests:
        results.append({"request_id": req["request_id"], "output_ids": []})
    
    while True:
        for i, (req, result) in enumerate(zip(requests, results)):
            max_new_tokens = req["max_new_tokens"]
            request_id = req["request_id"]
            
            if len(result["output_ids"]) >= max_new_tokens:
                done.add(req["request_id"])
                continue
            if request_id in done:
                continue
            
            # print(f"Processing request {request_id}...")
            prompt_ids = req["prompt_ids"]
            
            current_result = result["output_ids"]
            if current_result:
                prompt_ids = prompt_ids + current_result
            output_id = generate_one_token(prompt_ids)
            results[i]["output_ids"].append(output_id.item())
            if output_id.item() == 4:
                done.add(request_id)
        if len(done) == len(requests):
            break
    return results

if __name__ == "__main__":
    requests = [
        {"request_id": "A", "prompt_ids": [0], "max_new_tokens": 10},
        {"request_id": "B", "prompt_ids": [3], "max_new_tokens": 10},
        {"request_id": "C", "prompt_ids": [1], "max_new_tokens": 10},
        {"request_id": "D", "prompt_ids": [1], "max_new_tokens": 0},
    ]
    
    results = generate_many(requests)
    for result in results:
        print(f"Request ID: {result['request_id']}, Generated token IDs: {result['output_ids']}")