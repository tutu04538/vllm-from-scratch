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


if __name__ == "__main__":
    prompt_ids = [0]
    max_new_tokens = 10
    generated_ids = generate(prompt_ids, max_new_tokens)
    print("Generated token IDs:", generated_ids)