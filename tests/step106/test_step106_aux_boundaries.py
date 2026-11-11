"""EAGLE 辅助层编号是模型边界，而不是 decoder 的零基下标。"""
import torch
from torch import nn

from minivllm.models.qwen3 import Qwen3Model
from minivllm.testing.tiny_models import tiny_qwen3_config


class Increment(nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, positions, hidden, residual):
        return hidden + self.amount, torch.zeros_like(hidden)


def test_embedding_and_post_layer_boundaries():
    config = tiny_qwen3_config('tiny_gqa')
    model = Qwen3Model(config)
    model.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
    with torch.no_grad(): model.embed_tokens.weight.copy_(torch.arange(config['vocab_size'] * config['hidden_size']).reshape(config['vocab_size'], -1))
    model.layers = nn.ModuleList([Increment(10), Increment(20)])
    ids = torch.tensor([1, 2]); positions = torch.tensor([0, 1])
    embedding = model.embed_tokens(ids)
    model.set_aux_hidden_state_layers((0, 1, 2))
    _, aux = model(ids, positions)
    assert len(aux) == 3
    torch.testing.assert_close(aux[0], embedding, rtol=0, atol=0)
    torch.testing.assert_close(aux[1], embedding + 10, rtol=0, atol=0)
    torch.testing.assert_close(aux[2], embedding + 30, rtol=0, atol=0)
