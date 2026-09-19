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

def generate(prompt_ids, max_new_tokens):
    
    if type(prompt_ids) != torch.Tensor:
        generated_ids = torch.tensor(prompt_ids, device=model.device)
    else:
        generated_ids = prompt_ids.clone().to(model.device)
    
    for _ in range(max_new_tokens):
        logits = model.forward(generated_ids)
        next_token_id = sampler.sample(logits)
        generated_ids = torch.cat((generated_ids, next_token_id), dim=0)
        if next_token_id.item() == 4:  # Assuming token ID 4 is the end-of-sequence token
            break
    
    return generated_ids[len(prompt_ids):].tolist()


def generate_many(requests):
    results = []
    for req in requests:
        request_id = req["request_id"]
        prompt_ids = req["prompt_ids"]
        max_new_tokens = req["max_new_tokens"]
        output_ids = generate(prompt_ids, max_new_tokens)
        results.append({"request_id": request_id, "output_ids": output_ids})
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