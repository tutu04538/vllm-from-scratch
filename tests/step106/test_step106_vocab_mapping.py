import torch

from minivllm.models.qwen3_eagle3 import Eagle3ForCausalLM
from minivllm.testing.tiny_models import tiny_qwen3_config


def test_nonzero_offsets_map_before_softmax():
    config = tiny_qwen3_config('tiny_gqa')
    config.update(draft_vocab_size=4, num_hidden_layers=1)
    model = Eagle3ForCausalLM(config)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[:4, 0].copy_(torch.tensor([1., 2., 3., 4.]))
    model.d2t = torch.tensor([2, 3, 5, 6])
    hidden = torch.zeros(1, config['hidden_size']); hidden[0, 0] = 1
    raw = model.draft_vocab_logits(hidden)
    logits = model.compute_logits(hidden)
    ids = torch.tensor([2, 4, 7, 9])
    torch.testing.assert_close(logits[0, ids], raw[0], atol=0, rtol=0)
    valid = torch.zeros(config['vocab_size'], dtype=torch.bool); valid[ids] = True
    assert torch.isneginf(logits[0, ~valid]).all()
    probabilities = logits.softmax(dim=-1)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))
    assert probabilities[0, ~valid].count_nonzero() == 0
    assert int(raw.argmax()) == 3 and int(logits.argmax()) == 9
