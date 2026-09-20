"""采样：本关只要贪心。"""

import torch


class Sampler:

    def __init__(self):
        pass

    def sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)
